# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
#image = cv2.imread('images/coins.jpg')

def nothing(pos):
    pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',0,255, nothing)
cv2.createTrackbar('hc','Thresholds',17,255, nothing)
image = cv2.imread("images/aerea2.jpg")
image = imutils.resize (image, width=700)

while (True):
    #image = cv2.imread("images/lateral1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lc = cv2.getTrackbarPos('lc','Thresholds')
    hc = cv2.getTrackbarPos('hc','Thresholds')

    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, lc, hc, 3)
    #dilated = cv2.dilate(canny, (1, 1), iterations=0)

    #(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cnt, hierarchy) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

    #print("coins : ", len(cnt))
    cv2.imshow('canny', canny) 
    cv2.imshow('image', rgb) 
    cv2.waitKey(0) 
    #plt.hist(areas, bins=100)
    #plt.show()
#cv2.destroyAllWindows()