import os
import time

import numpy as np
import streamlit as st
from MQTT_Scripts.helpers import byte_array_to_pil_image, get_config, get_now_string, pil_image_to_byte_array
from MQTT_Scripts.mqtt import get_mqtt_client
from paho.mqtt import client as mqtt
from PIL import Image

from darknet import Darknet
from utils import *

model = Darknet("cfg/yolo-pose.cfg")
model.load_weights("backup/model.weights")

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC = CONFIG["camera"]["mqtt_topic"]
MQTT_TOPIC_NEW = CONFIG["publisher_1"]["mqtt_topic"]

FPS = CONFIG["camera"]["fps"]

VIEWER_WIDTH = 640


def test():

    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(topic=MQTT_TOPIC)
    time.sleep(4)
    client.loop_forever()

def get_random_numpy():
    """Return a dummy frame."""
    return np.random.randint(0, 100, size=(32, 32))

def on_connect(client, userdata, flags, rc):
    st.write(
        f"Connected with result code {str(rc)} to MQTT broker on {MQTT_BROKER}"
    )

# The callback for when a PUBLISH message is received from the server.
def on_message( client, userdata, msg):
    now = get_now_string()
    print("message on " + str(msg.topic) + f" at {now}")
    try:
        image = byte_array_to_pil_image(msg.payload)  # PIL image
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        output_image = np.copy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (544, 544))
        image = np.transpose(image, (2, 0, 1))
        image = image[None, :, :, :]
        image = torch.Tensor(image)

        new_image = image[0].permute(1, 2, 0).cpu().numpy()
        data = torch.from_numpy(new_image[None, :, :, :]).permute(0, 3, 1, 2)
        output = model(data).data

        all_boxes = get_region_boxes(output, 0.1, 1)

        match_thresh = 0.7
        save = True
        for i in range(output.size(0)):
            boxes = all_boxes[i]

            best_conf_est = 0.6
            for j in range(len(boxes)):
                if boxes[j][18] > best_conf_est:
                    box_pr = boxes[j]
                    best_conf_est = boxes[j][18]

            if best_conf_est > match_thresh:
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

                for idx, point in enumerate(corners2D_pr):
                    x = int(float(point[0]))
                    y = int(float(point[1]))
                    print("X{}: {}, Y{}: {}".format(idx + 1, x, idx + 1, y))
                    cv2.circle(output_image, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(output_image, str(idx + 1), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                1)
                print("\n")
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)
        output_image = pil_image_to_byte_array(output_image)

        client.publish(MQTT_TOPIC_NEW, output_image, MQTT_QOS)
        now = get_now_string()
        print(f"published frame on topic: {MQTT_TOPIC_NEW} at {now}")
        time.sleep(1 / FPS)

    except Exception as exc:
        print(exc)

def get_region_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False):
    # Parameters
    anchor_dim = 1
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (19 + num_classes) * anchor_dim)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -100000
    output = output.view(batch * anchor_dim, 19 + num_classes, h * w).transpose(0, 1).contiguous().view(
        19 + num_classes, batch * anchor_dim * h * w)
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * anchor_dim, 1, 1).view(
        batch * anchor_dim * h * w)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * anchor_dim, 1, 1).view(
        batch * anchor_dim * h * w)
    xs0 = torch.sigmoid(output[0]) + grid_x
    ys0 = torch.sigmoid(output[1]) + grid_y
    xs1 = output[2] + grid_x
    ys1 = output[3] + grid_y
    xs2 = output[4] + grid_x
    ys2 = output[5] + grid_y
    xs3 = output[6] + grid_x
    ys3 = output[7] + grid_y
    xs4 = output[8] + grid_x
    ys4 = output[9] + grid_y
    xs5 = output[10] + grid_x
    ys5 = output[11] + grid_y
    xs6 = output[12] + grid_x
    ys6 = output[13] + grid_y
    xs7 = output[14] + grid_x
    ys7 = output[15] + grid_y
    xs8 = output[16] + grid_x
    ys8 = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    # GPU to CPU
    sz_hw = h * w
    sz_hwa = sz_hw * anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > max_conf:
                        max_conf = conf
                        max_ind = ind

                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
                               bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h,
                               det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
            if len(boxes) == 0:
                bcx0 = xs0[max_ind]
                bcy0 = ys0[max_ind]
                bcx1 = xs1[max_ind]
                bcy1 = ys1[max_ind]
                bcx2 = xs2[max_ind]
                bcy2 = ys2[max_ind]
                bcx3 = xs3[max_ind]
                bcy3 = ys3[max_ind]
                bcx4 = xs4[max_ind]
                bcy4 = ys4[max_ind]
                bcx5 = xs5[max_ind]
                bcy5 = ys5[max_ind]
                bcx6 = xs6[max_ind]
                bcy6 = ys6[max_ind]
                bcx7 = xs7[max_ind]
                bcy7 = ys7[max_ind]
                bcx8 = xs8[max_ind]
                bcy8 = ys8[max_ind]
                cls_max_conf = cls_max_confs[max_ind]
                cls_max_id = cls_max_ids[max_ind]
                det_conf = det_confs[max_ind]
                box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
                       bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h,
                       det_conf, cls_max_conf, cls_max_id]
                boxes.append(box)
                all_boxes.append(boxes)
            else:
                all_boxes.append(boxes)

        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

# def process(input_image, model):

# def main():
#     model = Darknet("cfg/yolo-pose.cfg")
#     model.load_weights("backup/model.weights")
#
#     client = get_mqtt_client()
#     client.on_message = on_message
#     client.connect(MQTT_BROKER, port=MQTT_PORT)
#     client.subscribe(topic=MQTT_TOPIC)
#     time.sleep(4)
#     client.loop_forever()

if __name__ == '__main__':
    test()
