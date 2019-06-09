FROM ubuntu:18.10
FROM python:stretch

USER root

RUN apt-get update && \
    apt-get install -y graphviz libgraphviz-dev

WORKDIR /job
COPY . /job

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
