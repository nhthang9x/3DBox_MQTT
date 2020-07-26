# [PYTHON] 3D Object Detection based on SSPE (Webcam)

## Introduction

Here is the visualization for 3D object detection/localization using camera/webcam.

## How to run my code
In the very first step, in order to use my code, you have to install Docker via this link (https://docs.docker.com/get-docker/)

Step 1: Clone this repository to your device, then cd 3Dbox

Step 2: Download pre-trained model file via link (https://drive.google.com/file/d/1dVNPRsPFnQG5PQy_0Lmi2Jl9drsetXVu/view?usp=sharing) and put this file into /backup folder

Step 3: `sudo docker build -t 3dbox .`

Step 4: `xhost +`

Step 5: `sudo docker run -it --rm --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="DISPLAY" -e QT_X11_NO_MITSHM=1 --device=/dev/video0:/dev/video0 --network=host 3dbox /bin/bash`

Note: In Step 5, if there is no webcam in your device or your laptop is macbook, and you only want to test with video instead of webcam, run `sudo docker run -it --rm --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="DISPLAY" -e QT_X11_NO_MITSHM=1 --network=host 3dbox /bin/bash`   

Step 6: Running the publisher 1 by `python/python3 publisher_1_video.py` (video) or `python/python3 publisher_1_webcam.py` (webcam)

Step 7: Opening other terminal, finding out docker container ID by `sudo docker ps`

Step 8: Running into docker container: `sudo docker exec -it CONTAINER_ID bash`

Step 9: Running the publisher 2 by `python/python3 publisher_2.py`

Repeat Step 7 and 8 once

Step 10: Running the subscriber by `streamlit run subscriber.py`

Step 11: Clicking the link appearing in this terminal. Done

There are some issues can be happened when running in different operating system such as IOS. 
