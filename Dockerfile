FROM pytorch/pytorch:latest
MAINTAINER Hung Thang Nguyen <nhthang1009@gmail.com>

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip install opencv-python scipy paho-mqtt streamlit

COPY . /workspace/3Dbox_Test
WORKDIR /workspace/3Dbox_Test