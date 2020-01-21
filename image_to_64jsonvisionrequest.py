# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:28:05 2020

@author: cenic
"""
from requests.auth import HTTPBasicAuth
import base64
import json
import requests
import glob
import os


def encode_image(image):
    image_content = image.read()
    return base64.b64encode(image_content)


def createjson(filename):
    with open(filename, 'rb') as image:
        encoded = encode_image(image)
    content = encoded.decode('ascii')
    jsonstr = {"requests": [
            {"image": {"content": content},
             "features": [{"type": "LABEL_DETECTION",
                           "maxResults": 10
                           }]
             }
            ]
        }
    return jsonstr


def makeVisionRequestforImg(jsonobj, apikey):
    authorization = HTTPBasicAuth("key", apikey)
    url = "https://vision.googleapis.com/v1/images:annotate?key="
    flag_retry = "y"
    while(flag_retry == "y"):
        flag_retry = False
        try:
            r = requests.post(url + apikey,
                              json=jsonobj,
                              auth=authorization)
        except requests.exceptions.RequestException as e:
            r = e
            # catastrophic error. bail.
            print(e)
            flag_retry = input("continue? (y/n)")
    return r


def main(apikey, alreadyprocessed):
    lineList = [line.rstrip('\n') for line in open(alreadyprocessed)]
    filelist = glob.glob("Images/*")
    filelist = [x for x in filelist if x not in lineList]
    fileresps = glob.glob("response/*")
    resplistnums = [
        os.path.splitext(filename)[0][9:].split("_")[0]
        for filename in fileresps]
    filelist = [
        x for x in filelist if x[7:].split(".")[0] not in resplistnums]
    jsonlist = []
    for filename in filelist:
        print(filename)
        jsonobj = createjson(filename)
        filepref = os.path.splitext(filename)[0][7:]
        with open('request/{}_req.json'.format(filepref), 'w') as f:
            json.dump(jsonobj, f)
        r = makeVisionRequestforImg(jsonobj, apikey)
        with open('response/{}_resp.json'.format(filepref), 'w') as f:
            json.dump(r.json(), f)
            jsonlist.append(r.json())
    return jsonlist
