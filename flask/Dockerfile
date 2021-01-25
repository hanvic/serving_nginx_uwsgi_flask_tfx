FROM ubuntu:18.04

MAINTAINER Dongyul Lee "leedongyull@gmail.com"

RUN apt-get update -y \
    && apt-get install -y \
    gunicorn \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y \
    && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

COPY . /www/src/

WORKDIR /www/src/

#EXPOSE 5000
CMD ["uwsgi","uwsgi.ini"]