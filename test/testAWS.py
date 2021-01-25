import requests
import io
import base64


with open("/home/serving/alpha/data/_2737476_orig[1].jpg", "rb") as fh:
    buf = io.BytesIO(fh.read())


bb=buf.read()
# bb2 = base64.b64encode(bb).decode()
# print(bb2)

data = {'data':base64.b64encode(bb)}
json_response = requests.post('http://0.0.0.0:7014/predict',data=data)