import cv2

import numpy as np
from matplotlib import pyplot as plt
 
 
img = cv2.imread('images/aerea1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# define ROI of RGB image 'img'
#roi = img[5:200, 5:400]

# convert it into HSV
#hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

edges = cv2.Canny(hsv, 50, 150, apertureSize=3)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1] # threshold to binary

assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
plt.show()


cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
cv2.imshow('RGB image', img)
cv2.imshow('edges', edges)
cv2.imshow('HSV image', hsv)
cv2.waitKey()
cv2.destroyAllWindows()