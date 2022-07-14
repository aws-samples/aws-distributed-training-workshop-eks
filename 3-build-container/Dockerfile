FROM ubuntu:20.04

RUN apt-get update && apt-get install -y wget unzip python3 python3-pip htop

RUN pip3 install python-etcd 
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install tensorboard debugpy

RUN mkdir -p /workspace/
ADD cifar10-model-train.py /workspace/
ADD cifar10-model-test.py /workspace/
ADD cnn_model.py /workspace/
ADD utils.py /workspace/
ADD data-prep.sh /workspace/
