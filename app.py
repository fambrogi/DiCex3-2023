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




app = Flask(__name__)



""" Taken from the official tensorflow documentation, see
https://www.tensorflow.org/hub/tutorials/object_detection
"""



def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):


    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename



def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                         int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
              image_pil,
              ymin,
              xmin,
              ymax,
              xmax,
              color,
              font,
              display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)


    return img



def run_detector(detector, path):

    """ Needed??? 
    def prepare(filepath):
	    IMG_SIZE = 50
	    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	    img_array = img_array/255.0
	    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    """


    img = load_img(path)

    #converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...] # ORIGINAL

    converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...] ### => WORKS

    ### TRY THIS https://stackoverflow.com/questions/66472843/what-does-mean-python-inputs-incompatible-with-input-signature
    # it says that the dimensions of images should be within range::: "The input tensor is a tf.uint8 tensor with shape [1, height, width, 3] with values in [0, 255]."
    #converted_img = converted_img[:, :, :, :3]

    #converted_img = rescale(converted_img)
    start_time = time.time()
    tf.cast(converted_img, tf.uint8)
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    inference_time = end_time - start_time
    print("Inference time: ", end_time-start_time)

    image_with_boxes = 0
    """
    image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

    display_image(image_with_boxes)
    """
    return image_with_boxes, inference_time

def detect_img(image_url):
    start_time = time.time()
    image_path = download_and_resize_image(image_url, 640, 480)
    run_detector(detector, image_path)
    end_time = time.time()
    print("Inference time:",end_time-start_time)




""" Here below: Implementation of the LOOP """

def detection_loop(filename_image, path, output):
    
    inference_times = []
    with open('inference_times.txt', 'w') as f:
        for filename, image in filename_image.items():
            image_location = os.path.join(path, filename)
            print(str(image_location))
            image_with_boxes, inference_time = run_detector(detector, image_location)
            inference_times.append(inference_time)
            #cv2.imwrite(f"images/out/with_boxes_{filename}",  cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
            cv2.imwrite("images/out/with_boxes_{filename}",  cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
            inference_time = round(inference_time, 3)
            #f.write(f"image: {filename}, inference_time: {inference_time}s\n")
            f.write("image: {filename}, inference_time: {inference_time}s\n")

        f.write(80*"-")
        average_inference_time = round(np.mean(np.array(inference_times)),3)
        #f.write(f"\n\naverage inference time: {average_inference_time}s\n")
        f.write("\n\naverage inference time: {average_inference_time}s\n")
        f.close()




#initializing the flask app
app = Flask(__name__)


def load_img(path):
    print("LOADING IMAGE :::")
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


#routing http posts to this method
#@app.route('/api/detect', methods=['POST', 'GET'])

def TESTONE(data_input):

    """ Check if a single image is given 
        Handling cases:
        - if input_image is given, analyze only this picture
	- if input_directory is given, extract all files in directory

    """

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


    out = open('NOW_doing_smt.txt', 'w') 
    for f in files:
        try:
           image_with_boxes, inference_time = run_detector(detector, f)
        except:
           image_with_boxes, inference_time = run_detector(detector, f)
           inference_time = '-1' 

        out.write(f + '_' + str(inference_time) + '\n')

    out.close()


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


    a = TESTONE(data_input)  ### -> Processing stuff around 



    status_code = Response(status = 200)
    return status_code


    #return(data_input)





    #return(i)

    #http://131.130.157.123:1111/api/detect?input_image=test1.jpg
    #http://131.130.157.123:1111/?input_image=test1.jpg

    '''
    output = request.values.get('output')
    #output = request.form.get('output')

    path = data_input
    filename_image = {}

    input_format = ["jpg", "png", "jpeg"]

    if data_input.find(".") != -1:

        print(data_input + " is a file")
        split_data_input = data_input.split(".", 1)

        if data_input.endswith(tuple(input_format)):
            print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
            path_splitted = []
            path_splitted = re.split('/', data_input)
            filename = path_splitted[len(path_splitted)-1]
            filename_image[filename] = Image.open(data_input)
            path = os.path.dirname(data_input)+"/"
        else:
            print(data_input + " is a path with the following files: ")
            for filename in os.listdir(data_input):
                image_path = data_input + filename
                filename_image[filename] = Image.open(image_path)
                print("  " + filename)
    
    #detection_loop(filename_image, path, output)
    

    status_code = Response(status = 200)
    return status_code
    '''

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port='1111')
