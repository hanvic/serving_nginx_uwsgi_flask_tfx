import requests
import io
import base64


with open("/home/serving/alpha/data/AlphadoPhoto_2020-10-15 13_33_51_931.JPG", "rb") as fh:
    buf = io.BytesIO(fh.read())


bb=buf.read()
# bb2 = base64.b64encode(bb).decode()
# print(bb2)

data = {'data':base64.b64encode(bb)}
json_response = requests.post('http://115.71.48.60:8080/predict',data=data)