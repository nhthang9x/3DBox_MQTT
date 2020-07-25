# [PYTHON] 3D Object Detection based on SSPE (Webcam)

## Introduction

Here is the visualization for 3D object detection/localization using camera/webcam.

## How to run my code
In the very first step, in order to use my code, you have to install Docker via this link (https://docs.docker.com/get-docker/)

Step 1: Clone this repository to your device, then cd 3Dbox

Step 2: Download pre-trained model file via link (https://drive.google.com/file/d/1dVNPRsPFnQG5PQy_0Lmi2Jl9drsetXVu/view?usp=sharing) and put this file into /backup folder

Step 3: `sudo docker build -t 3dbox .`

Step 4: `xhost +`

Step 5: `sudo docker run -it --rm --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="DISPLAY" -e QT_X11_NO_MITSHM=1 --device=/dev/video0:/dev/video0 3dbox /bin/bash`   

Step 6: Running my code by python/python3 webcam.py

After Step 5, you can see that webcam/camera in your device is operated, and can detect blue box if it appeared in front of your camera/device
