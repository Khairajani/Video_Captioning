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
    
    # Reading video
    cap = cv2.VideoCapture(args['video_path'])
    ret, image = cap.read()

    # Taking shape of the video/frames
    fshape = image.shape
    fheight = fshape[0]
    fwidth = fshape[1]

    # Writing on the output file
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*"vp80")
    output_video = cv2.VideoWriter('./data/output/output.avi', fourcc, 5.0, (fwidth,fheight))
    
    #create the output file to write the content
    temp_time = time.time()
    file_ouput = open("./data/output/output_" + str(temp_time) + ".txt", "a")


    frame_count = 0
    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        frame_count+=1

        if not ret:
            print("[Exiting..]")
            break

        if frame_count%3==0:

            # Start inferencing
            # image_size = image.shape[:2]

            # detection
            # img,img_dst,total_people, people_danger, people_warning, people_normal = preprocessing_and_detection(image, model, image_size, input_size, ANCHORS, STRIDES, XYSCALE, classes, NUM_CLASS, args["dev"])

            #img,img_dst = show_save(args,img,img_dst)
            caption = generateCaption(image)
            print(caption)

            output_line=caption+'\n'
            file_ouput.write(output_line)

            #cv2.rectangle(image, (10,10), (320,220),(0,0,0),-1)
            cv2.putText(image, "Caption: " + caption, (20,fheight-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

            output_video.write(image)   

            # Display the resulting frame
            cv2.imshow('frame',image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    output_video.release()
    file_ouput.write("\n\n")
    file_ouput.close()
    cv2.destroyAllWindows()

    """
    # Start inferencing
    h, w = image.shape[0], image.shape[1]
    h_crop = int(h*((100-args["height_percentage"])/100))
    left_crop = int(w*(args['left_remove']/100))
    right_crop = w//2 + int((w/2) - ((w)*(args['right_remove']/100)))
    image = image[h_crop:, left_crop:right_crop]
    image_size = image.shape[:2]
    
    # detection
    img,img_dst,total_people, people_danger, people_warning, people_normal = preprocessing_and_detection(image, model, image_size, input_size, ANCHORS, STRIDES, XYSCALE, classes, NUM_CLASS, args["dev"])

    img,img_dst = show_save(args,img,img_dst)
    
    #return img,img_dst,total_people, people_danger, people_warning, people_normal
    """

if __name__ == "__main__":  
    print("Directly Invoked")
    video_path = "./data/input/s.mp4"
    start_inference(video_path=video_path)