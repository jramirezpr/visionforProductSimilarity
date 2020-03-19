# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:04:23 2020

@author: cenic
"""

import http.client, urllib.parse
import json
import os
import os.path
from azure.cognitiveservices.search.visualsearch import VisualSearchClient
from azure.cognitiveservices.search.visualsearch.models import (
    VisualSearchRequest,
    CropArea,
    ImageInfo,
    Filters,
    KnowledgeRequest,
)
from msrest.authentication import CognitiveServicesCredentials
IMG_PATH = os.path.join(os.getcwd(), "imgprodtest1firstretriev.jpg")

def visualsearch(subscription_key, image_path):
    creds = CognitiveServicesCredentials(subscription_key)
    endp = "https://westus.api.cognitive.microsoft.com/"
    client = VisualSearchClient(endpoint=endp, credentials=creds)
    with open(image_path, "rb") as image_fd:
        knowledge_request = json.dumps(VisualSearchRequest().serialize())
        print("Bing Visual Search")
        result = client.images.visual_search(image=image_fd, knowledge_request=knowledge_request)
        return result
    if not result:
        print("No visual search result data.")

        # Visual Search results
    if result.image.image_insights_token:
        print("Uploaded image insights token: {}".format(result.image.image_insights_token))
    else:
        print("Couldn't find image insights token!")

    # List of tags
    if result.tags:
        first_tag = result.tags[0]
        print("Visual search tag count: {}".format(len(result.tags)))

        # List of actions in first tag
        if first_tag.actions:
            first_tag_action = first_tag.actions[0]
            print("First tag action count: {}".format(len(first_tag.actions)))
            print("First tag action type: {}".format(first_tag_action.action_type))
        else:
            print("Couldn't find tag actions!")
    else:
        print("Couldn't find image tags!")