FROM python:3.8

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip3 install --upgrade pip && pip install --upgrade pip && pip3 install torch numpy torchvision matplotlib networkx zmq jsonlines pillow smallworld localconfig PyWavelets pandas crudini scikit-learn lz4 fpzip && apt update -y && apt upgrade -y && apt install -y dnsutils net-tools

# TO COMPLETE : PATH OF DATASET IN ORDER TO COPY TO CONTAINERS
# COPY PATH/CIFAR /CIFAR
# TO COMPLETE : PATH OF SCRIPTS IN ORDER TO COPY TO CONTAINERS
# COPY PATH/scripts /scripts

WORKDIR /
ENTRYPOINT ["/bin/bash", "/scripts/example_client.sh"] 
