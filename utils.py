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


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)


    return img



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

def detect_img(image_url):
    start_time = time.time()
    image_path = download_and_resize_image(image_url, 640, 480)
    run_detector(detector, image_path)
    end_time = time.time()
    print("Inference time:",end_time-start_time)



def run_detector_(detector, path):

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

def load_img(path):
    print("LOADING IMAGE :::")
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def TESTONE_old(data_input):

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




