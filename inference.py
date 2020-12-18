# python codevector.py --weights ./data/weights/yolov4.weights --size 320 --video ./data/input/1.mkv --show True  --location telibandha
import datetime
import time
import cv2
import os
import numpy as np
from modules.utils import *

# The main function
def start_inference(**args):

    print("STARTING INFERENCE", args)
    
    # Loading model
    # model, input_size, ANCHORS, STRIDES, XYSCALE, classes, NUM_CLASS = model_initialize(args)
    # print("Model Loaded Successfully")  

    # Getting image
    image = cv2.imread(args['image_path'])

    # Start inferencing
    h, w = image.shape[0], image.shape[1]
        
    # # detection
    # img,img_dst,total_people, people_danger, people_warning, people_normal = preprocessing_and_detection(image, model, image_size, input_size, ANCHORS, STRIDES, XYSCALE, classes, NUM_CLASS, args["dev"])

    # img,img_dst = show_save(args,img,img_dst)
    
    caption = generateCaption(image)
    print(caption)
    #cv2.rectangle(image, (10,10), (320,220),(0,0,0),-1)
    cv2.putText(image, "Caption: " + caption, (40,h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.imwrite("./data/output/output.jpg",image)

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(0) & 0xFF == 27:            
        cv2.destroyAllWindows()

if __name__ == "__main__":  
    print("Directly Invoked")
    image_path = "./data/input/01.jpeg"
    start_inference(image_path=image_path)