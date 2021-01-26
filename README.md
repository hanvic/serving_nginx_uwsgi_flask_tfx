# NGINX - uWsgi - Flask with Docker

Nginx + uWsgi + Flask + TFX serving with docker-compose


## Project Structure
```bash
├── docker-compose-uwsgi.yml
├── flask
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
│   ├── route
│   ├── templates
│   └── uwsgi.ini
├── nginx
│   ├── Dockerfile
│   └── nginx.conf
└── venv
```

## Set Up with..?

- 1) Dockerfile를 사용해서 Flask와 uWsgi 서버 올리기
- 2) Dockerfile로 Nginx서버로 만들어주기
- 3) docker-compose를 사용해서 생성된 Docker파일들을 하나로 묶어주기

## Docker로 Flask와 uWsgi 세팅하기
tfx는 docker image로부터 model 위치와 환경 변수를 같이 넘겨주면 실행이 가능하기에 docker-compose내에 서비스 설정만으로 실행이 가능하다.  
이후 실행되면 REST API만으로 test가 가능하다.
만약 단일, 도커 tfx serving 서비스를 실행하고자 하면 다음의 command로 실행이 가능하다.
```bash
docker run --runtime=nvidia -p 3434:8501 --rm -v /home/serving/:/models/ --mount type=bind,source=/home/serving/eye,target=/models/eye -e MODEL_NAME=eye  -t tensorflow/serving:2.3.0-gpu --model_config_file_poll_wait_seconds=60 --model_config_file=/models/models.config --per_process_gpu_memory_fraction=0.15 --allow_growth=True
```

## Docker로 Flask와 uWsgi 세팅하기

### Project Structure

```bash
flask
├── Dockerfile
├── app.py
├── requirements.txt
├── route
│   ├── __init__.py
│   └── app_route.py
├── templates
│   └── index.html
└── uwsgi.ini
```

### app.py

```python
from flask import Flask
from route.app_route import app_route

app = Flask(__name__)

app.register_blueprint(app_route)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

### uwsgi.ini

```
[uwsgi]
wsgi-file = {working dir}/app.py
callable = app
socket = :5000
processes = 4
threads = 2
master = true
vacum = true
chmod-socket = 666
die-on-term = true
```

### route/app_route.py

```python
from flask import Blueprint, render_template

app_route = Blueprint('first_route',__name__)

@app_route.route('/')
def index():
    return render_template('index.html')
```

### requirements.txt

```
click==7.1.2
Flask==1.1.2
itsdangerous==1.1.0
Jinja2==2.11.2
MarkupSafe==1.1.1
uWSGI==2.0.19.1
Werkzeug==1.0.1
pillow
numpy
opencv-python
requests
```

### templates/index.html

```html
<html>
    <head>
        <title>flask Docker</title>
    </head>
</html>
<body>
    <h5>
        Welcome to flask server
    </h5>
</body>
```

### Docker

```Dockerfile
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

COPY . /www/src/
WORKDIR /www/src/
RUN pip3 install -r requirements.txt

CMD ["uwsgi","uwsgi.ini"]
```

## Docker-Nginx


### Project Structure

```bash
nginx
├── Dockerfile
└── nginx.conf
```

### nginx.conf

```
upstream up_uwsgi {
    server 192.168.80.6:5000;
}

server {
	listen 8080;
    server_name 192.168.80.7;
    charset     utf-8;

    client_max_body_size 75M;   # adjust to taste

	location / {
		include uwsgi_params;
		uwsgi_pass up_uwsgi;
	}
}
```

### Dockerfile

```Dockerfile
FROM nginx:1.17.4

RUN rm /etc/nginx/conf.d/default.conf

COPY default.conf /etc/nginx/conf.d/default.conf
```


### docker-compose-uwsgi.yml

```docker
version: "3.3"

services:
    tfserving:
        image: tensorflow/serving:2.3.0-gpu
        container_name: tfs
        volumes:
            - /home/serving/:/models/
            - type: bind
              source: /home/serving/eye
              target: /models/eye
        ports:
            - "8501:8501"
        environment:
            - NVIDIA_VISIBLE_DEVICES=0
            - MODEL_NAME=eye
        command:
            - '--model_config_file_poll_wait_seconds=60'
            - '--model_config_file=/models/models.config'
            - '--per_process_gpu_memory_fraction=0.15'
            - '--allow_growth=True'
        expose:
            - 8501
        networks:
            wire1:
              ipv4_address: 192.168.80.5

    flask:
        image: flask:serving
        build: ./flask
        container_name: flask
        restart: always
        ports:
            - "5000:5000"
        environment:
            - APP_NAME=FlaskTest
#            - /docker/fmsnas/10.13.88.51/logs:/www/logs
        expose:
            - 5000
        networks:
            wire1:
              ipv4_address: 192.168.80.6

    nginx:
        image: nginx:0.0.1
        build: ./nginx
        container_name: nginx
        restart: always
        depends_on:
            - flask
        ports:
            - "8080:8080"

        expose:
            - 8080
        networks:
            wire1:
                ipv4_address: 192.168.80.7


networks:
  wire1:
      driver: bridge
      ipam:
          config:
              - subnet: 192.168.0.0/16
```

## Run docker-compose file

```bash
$ docker-compose -f docker-compose-uwsgi.yml up -d --build
```

## test
test 폴더안에 testAWS.py를 통해 테스트가 가능하다.
다음과 같이 이미지 파일을 바이트 array로 read한 후 base64로 스트링으로 변화하여 application/json content-type의 post 메쏘드를 날려주면 된다.

```python
with open("/home/serving/alpha/data/AlphadoPhoto_2020-10-15 13_33_51_931.JPG", "rb") as fh:
    buf = io.BytesIO(fh.read())

bb=buf.read()
data = {'data':base64.b64encode(bb)}
json_response = requests.post('http://115.71.48.60:8080/predict',data=data)
```

## to do list

- [x] uwsgi의 프로세스 thread 최적화
- [x] tfx 서빙 서버의 최적화 - resolution size
- [ ] tfx 서빙 서버 warm-up
- [ ] docker compose에 log 폴더 volume시키기