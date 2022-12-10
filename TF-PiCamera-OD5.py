#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Pi Camera) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
#import telepot
import pyautogui
#import requests
#from picamera import PiCamera
from PIL import ImageGrab
#import sys
from threading import Thread
import requests
import time
import os




tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
class VideoStream:
    """Camera object that controls video streaming 6from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


class Api:
    # url to send image for upload
    baseUrl = "https://production-vohbr3y-fxlqssehfqz5e.au.platformsh.site"
    image = any
    type = ""
    start = time.time()
    
    # default constructor
    def __init__(self, image, type, start):
        self.image = image
        self.type = type
        self.start = start
    
    def Send(self):
        statusCode, imageUrl = self.uploadImage()
        if statusCode == 200:
            self.createData(imageUrl=imageUrl)
        end = time.time()
        print('Delay Web =', end - self.start,'s')
    
    def uploadImage(self):
        files = {'myfile': (self.image.replace("/home/pi/tensorflow/snapshot/", ""), open(self.image, 'rb'), 'multipart/form-data')}
        response = requests.post(self.baseUrl+"/images/upload",files=files)
        statusCode = response.status_code
        if statusCode == 200:
            print("SUCCESS UPLOAD IMAGE")
            responseJson = response.json()
            return statusCode, responseJson["url"]
        else:
            print("FAILED UPLOAD IMAGE")
            print("Error Response :", response.json())
            return statusCode, ""
    
    def getType(self):
        if self.type == "proper":
            return 1
        elif self.type == "improper":
            return 2
        else:
            return 3
            
    def createData(self, imageUrl):
        payload = {
            'mask_type': self.getType(),
            'image_url': imageUrl
        }
        response = requests.post(self.baseUrl+"/datas/", data=payload)
        statusCode = response.status_code
        if statusCode == 200:
            print("SUCCESS SEND DATA")
        else:
            print("FAILED SEND DATA")
            print(response)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='od-models/my_savedmodel')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='/home/pi/tensorflow/od-models/my_savedmodel/label_map.pbtxt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
                    
args = parser.parse_args()


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# Load the model
# ~~~~~~~~~~~~~~
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()


#variabel aktif condition
aktif = 1

print('Running inference for PiCamera')
videostream = VideoStream(resolution=(640,480),framerate=30).start()
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = videostream.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    imH, imW, _ = frame.shape

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    
    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
    #objects2 = []
    objects = []
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    count = 0
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            #increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
           
            #Draw label
            object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            #definisi nama object dari array menjai list
            #objects2.append(label)
            objects.append(object_name)
            
    #mengubah list array menjadi string
    #test2 = str(objects2)
    test = str(objects)
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    # Draw count of object detected in corner of frame
    cv2.putText (frame,'Objects Detected : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
   
    maskType = ""
    imageFileName = ""
   
    #If a person is detected
    if 'improper' in test:
   
        maskType = "improper"
        
        #aktif condition
        aktif = 0
        
        #notif terminal improper
        print('\nImproper')
        
        # date time string for file name
        path = "/home/pi/tensorflow/snapshot/"
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        imageFileName = path+"img.jpg"
        
        #screenshoot with pyautogui
        pyautogui.screenshot("/home/pi/tensorflow/snapshot/object.jpg")
        ss = cv2.imread('/home/pi/tensorflow/snapshot/object.jpg')
        cropped_image = ss[0:556, 0:640]
        cv2.imwrite(imageFileName, cropped_image)
 

        
    #If a person is detected
    elif 'nomask' in test:
   
        maskType = "nomask"
        
        #aktif condition
        aktif = 0
        
        #notif terminal No Mask
        print('\nNo Mask')
        
        # date time string for file name
        path = "/home/pi/tensorflow/snapshot/"
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        imageFileName = path+"img2.jpg"
        
        #screenshoot with pyautogui
        pyautogui.screenshot("/home/pi/tensorflow/snapshot/object2.jpg")
        ss = cv2.imread('/home/pi/tensorflow/snapshot/object2.jpg')
        cropped_image = ss[0:556, 0:640]
        cv2.imwrite(imageFileName, cropped_image)

    
        
    #If a person is detected
    elif 'proper' in test:
    
        maskType = "proper"
        
        #aktif condition
        aktif = 0
        
        #notif terminal Proper
        print('\nProper')
        
        # date time string for file name
        path = "/home/pi/tensorflow/snapshot/"
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        imageFileName = path+"img3.jpg"
        
        #screenshoot with pyautogui
        pyautogui.screenshot("/home/pi/tensorflow/snapshot/object3.jpg")
        ss = cv2.imread('/home/pi/tensorflow/snapshot/object3.jpg')
        cropped_image = ss[0:556, 0:640]
        cv2.imwrite(imageFileName, cropped_image)
        
        
        print(imageFileName)
        print(os.path.exists(imageFileName))
        api = Api(image=imageFileName, type=maskType, start=time.time())
        api.Send()

      
       
   

        
    #show the window
    title = "Object Detector"
    cv2.namedWindow(title)
    cv2.moveWindow(title, 0, 0)
    cv2.imshow(title, frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    


    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print("Done")
        