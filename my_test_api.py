from requests import *
import json
import uuid
import threading
import base64

def convertImgToBase64(img_path):
	with open(img_path, 'rb') as f:
		data = base64.b64encode(f.read())
	return data


r = post(url='http://192.168.1.9:2112/', data={'id': uuid.uuid4(), 'data':'con ngoan trò giỏi'})
# print(json.loads(r.text))
a = json.loads(r.text)
imgdata = base64.b64decode(a['image'])
# imgdata = a['image'].decode('base64')
filename = 'get_api.png'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(imgdata)

