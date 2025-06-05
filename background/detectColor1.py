import cv2
import numpy as np
import matplotlib.pyplot as plt


def fixHSVRange(h, s, v):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * h / 360, 255 * s / 100, 255 * v / 100)

#[ 92  28 137]
#[ 87  12 221]
im=cv2.imread("images/aerea1.jpg",1)
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
color1 = fixHSVRange(h=92, s=28, v=137)
color2 = fixHSVRange(h=87, s=12, v=221)
mask = cv2.inRange(im_hsv, color1, color2)
#cv2.imwrite("mask.jpg",mask)
cv2.imshow("mask.jpg",mask)
cv2.waitKey()

