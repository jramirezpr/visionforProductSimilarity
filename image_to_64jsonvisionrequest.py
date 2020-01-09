# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:28:05 2020

@author: cenic
"""
import base64
import json
def encode_image(image):
  image_content = image.read()
  return base64.b64encode(image_content)
with open("donuts.jpeg", 'rb') as image:
    encoded=encode_image(image)
content = encoded.decode('ascii')
jsonstr = {
  "requests":[
    {
      "image":{
        "content": content
      },
      "features": [
        {
          "type":"LABEL_DETECTION",
          "maxResults":1
        }
      ]
    }
  ]
}
with open('request.json', 'w') as f:
    json.dump(jsonstr,f)