import requests


json_response = requests.get('http://220.118.0.29:7014/')
print("json_response : ", json_response.text)