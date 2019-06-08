FROM nvidia/cuda:8.0-devel-ubuntu16.04

USER root

RUN apt-get update && \
    apt-get install -y python3 python3-pip graphviz libgraphviz-dev

WORKDIR /job
COPY . /job

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
