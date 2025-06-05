import cv2
import numpy as np 
import imutils
from tracker import *

previous_point = None  # default value at start
img = cv2.imread('images/test9.png')

def click_event(event, x, y, flags, params):
    global previous_point   # to assing new value to external variable

    font = cv2.FONT_HERSHEY_SIMPLEX

    if event == cv2.EVENT_LBUTTONDOWN:
        text = f"{x}, {y}"

        print(text)
        cv2.putText(img, text, (x, y), font, 1, (255, 0, 0), 2)

        cv2.imshow('image', img)

        if previous_point:  # check if there is previous point (not `None`)
            x2, y2 = previous_point
            dist = ((x-x2)**2 + (y-y2)**2 )**0.5
            print('distance:', dist)
                  
        previous_point = (x, y)  # keep it for next calculation