FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        tzdata \
        vim \
        git \
        lsof \
        python3 \
        libopencv-dev \
        p7zip-full \
        python3-pip \
        python3-dev \
        sqlite3
 
WORKDIR home/
 
COPY . .
 
RUN pip3 install -r requirments.txt
