import argparse
import warnings

warnings.filterwarnings("ignore")

from darknet import Darknet
from utils import *


from MQTT_Scripts.helpers import get_config, get_now_string, pil_image_to_byte_array
from MQTT_Scripts.mqtt import get_mqtt_client
from PIL import Image


CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC_CAMERA = CONFIG["camera"]["mqtt_topic"]
VIDEO_SOURCE = CONFIG["camera"]["video_source"]
FPS = CONFIG["camera"]["fps"]


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=544, help="The common width and height for all images")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="backup/model.weights")
    # parser.add_argument("--input", type=str, default="LINEMOD/ape/JPEGImages/000000.jpg")
    # parser.add_argument("--output", type=str, default="test_videos/output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    client = get_mqtt_client()
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4)
    client.loop_start()

    input_video = cv2.VideoCapture(0)
    # input_video = cv2.VideoCapture("sample_video_box_10.mp4")

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        flag, image = input_video.read()
        output_image = np.copy(image)

        # Additional code for MQTT
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)
        output_image = pil_image_to_byte_array(output_image)
        client.publish(topic=MQTT_TOPIC_CAMERA, payload=output_image, qos=MQTT_QOS)

        now = get_now_string()
        print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now}")
        time.sleep(1 / FPS)


if __name__ == '__main__':
    opt = get_args()
    test(opt)