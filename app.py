# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings('ignore')

from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
from matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import detect
import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re
import tensorflow_hub as hub
import tensorflow as tf

tf.keras.backend.set_floatx('float64')



import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

from utils import * 



app = Flask(__name__)



""" Taken from the official tensorflow documentation, see
https://www.tensorflow.org/hub/tutorials/object_detection
"""


def draw_detections_on_image(image, detections, labels=''):
    image_with_detections = image
    width, height, channels = image_with_detections.shape
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    label_padding = 5
    
    num_detections = detections['num_detections']
    if num_detections > 0:
        for detection_index in range(num_detections):
            detection_score = detections['detection_scores'][detection_index]
            detection_box = detections['detection_boxes'][detection_index]
            detection_class = detections['detection_classes'][detection_index]
            #detection_label = labels[detection_class]
            #detection_label_full = detection_label + ' ' + str(math.floor(100 * detection_score)) + '%'
            
            y1 = int(width * detection_box[0])
            x1 = int(height * detection_box[1])
            y2 = int(width * detection_box[2])
            x2 = int(height * detection_box[3])
                        
            # Detection rectangle.    
            image_with_detections = cv2.rectangle(
                image_with_detections,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

    # not writing labels here
    return image_with_detections


def detect_objects_on_image(image, model):

    start_time = time.time()


    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    # Adding one more dimension since model expect a batch of images.
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)

    num_detections = int(output_dict['num_detections'])
    output_dict = {
        key:value[0, :num_detections].numpy() 
        for key,value in output_dict.items()
        if key != 'num_detections'
    }
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    end_time = time.time()

    return output_dict, end_time-start_time 






def run_detector(data_input, detector):

    path = data_input

    if '.' in path :
       # analyzing a single image  
       files = [ path ]

    else:
       # analyzing the entire directory 
        try:
           files = [path + '/' + f for f in os.listdir(path) if f.split('.')[-1] in ['jpg','jpeg','png'] ]
        except:
           files = []

    out = open('DETECTING_doing_smt.txt', 'w') 

    '''
    for f in files:
        try:
           image_with_boxes, inference_time = run_detector(detector, f)
        except:
           image_with_boxes, inference_time = run_detector(detector, f)
           inference_time = '-1' 

        out.write(f + '_' + str(inference_time) + '\n')
    '''
    for f in files:
        fname = f.split('.')[0]
        print(f)
        image_np = np.array(Image.open(f))
        detections, inference_time = detect_objects_on_image(image_np, detector)
        image_with_detections = draw_detections_on_image(image_np, detections, '')
        #return(detections)
        out.write(f + '_' + str(inference_time) + '\n')
        cv2.imwrite( fname + '_detected.jpg',  cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR))

        #plt.figure(figsize=(8, 6))
        #plt.imshow(image_with_detections)

    out.close()


""" Here below: Implementation of the LOOP """
#initializing the flask app
app = Flask(__name__)



#routing http posts to this method
#@app.route('/api/detect', methods=['POST', 'GET'])
#@app.route('/api/detect', methods=['POST', 'GET'])




from flask import jsonify

'''
if request.method == 'POST':
    return jsonify(**request.json)
'''

#detector = hub.load(module_handle).signatures['default']

# THIS ONE IS VERY SLOW
#module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
#detector = hub.load(module_handle)
#detector.signature['default']


# Apply image detector on a single image. -> FAST BUT DOES NOT WORK WITH IMAGE SIZE...

module_handle= "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = hub.load(module_handle)



# Loads the module from internet, unpacks it and initializes a Tensorflow saved model.
import pathlib

def load_model(model_name):
    model_url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'
    
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=model_url,
        untar=True,
        cache_dir=pathlib.Path('.tmp').absolute()
    )
    model = tf.saved_model.load(model_dir + '/saved_model')
    
    return model

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
saved_model = load_model(MODEL_NAME)
detector = saved_model.signatures['serving_default']


### EXAMPLE HERE https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1 
#detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
#detector_output = detector(image_tensor)
#class_ids = detector_output["detection_classes"]


### FOR TESTING

#http://131.130.157.123:1111/api/detect?input_image=test1.jpg
#http://131.130.157.123:1111/?input_image=test1.jpg


@app.route('/', methods=['POST', 'GET'])
def main():

    #img = request.files["image"].read()
    #image = Image.open(io.BytesIO(img))
    #data_input = request.args['input']
    #data_input = request.values.get('input')
    #data_input = request.values

    data_input = request.values['input_image']

    #a = TESTONE(data_input)  ### -> Processing stuff around 
    a = run_detector(data_input, detector)  ### -> Processing stuff around 

    status_code = Response(status = 200)
    return status_code


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port='1111')


