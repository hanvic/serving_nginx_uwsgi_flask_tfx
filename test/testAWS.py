import requests
import io
import base64


with open("/home/serve/data/_2737476_orig[1].jpg", "rb") as fh:
    buf = io.BytesIO(fh.read())

bb=buf.read()
data = {'data':base64.b64encode(bb)}
# json_response = requests.post('http://115.71.48.60:8080/predict',data=data)
json_response = requests.post('http://0.0.0.0:7014/predict',data=data)
print("json_response",json_response.text)