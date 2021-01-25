from flask import Flask, jsonify, request, make_response
from PIL import Image
import base64
from werkzeug.utils import secure_filename
# import config
# import demo
import os
import io
import numpy as np
import cv2
import json
import time
import requests
from route.app_route import app_route

app = Flask(__name__)
# app.register_blueprint(app_route)

@app.route('/predict', methods=['POST'])
def predict():
    # f = request.files['file']
    # in_memory_file = io.BytesIO()
    # f.save(in_memory_file)
    # data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    a=request
    data=request.form['data']
    imgdata = base64.b64decode(str(data))
    image = Image.open(io.BytesIO(imgdata))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # data = np.fromstring(data, dtype=np.uint8)
    # img = cv2.imdecode(data, color_image_flag)
    cv2.imwrite('/home/serving/alpha/data/test_saved.png',img)
    img = cv2.resize(img, (512,512))
    tmp_images = []
    tmp_images.append(img)
    test_images = np.array(tmp_images, dtype=np.float32)

    # Make a request
    data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
    print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))
    headers = {"content-type": "application/json"}
    start = time.time()
    json_response = requests.post('http://192.168.80.5:8501/v1/models/eye:predict', data=data, headers=headers)
    print("elapsed time : ", time.time() - start)
    predictions = json.loads(json_response.text)['predictions']
    predictions = np.array(predictions)
    print(predictions)
    result=dict()
    result['predict']=predictions

    def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, datetime.datetime):
        #     return obj.__str__()

    json_object = json.dumps(result, indent=4, default=myconverter, ensure_ascii=False)
    return json_object

@app.route('/',methods=['GET'])
def index():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7014)