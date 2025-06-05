import cv2
import numpy as np
import imutils

#image = cv2.imread('images/blobs.jpg')
image = cv2.imread('images/area4.jpg')
#image = cv2.resize(image,(700,700))
mask = np.zeros(image.shape, dtype=np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,100,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

blobs = 0
for c in cnts:
    cv2.waitKey()
    area = cv2.contourArea(c)
    cv2.drawContours(mask, [c], -1, (0,255,0), -1)
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.putText(mask, str(blobs), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # if area > 10:
    #     blobs += 2
    # else:
    #     blobs += 1

    
mask = imutils.resize (mask, width=1024)
opening = imutils.resize (opening, width=1024)
thresh = imutils.resize (thresh, width=1024)
image = imutils.resize (image, width=1024)
    #cv2.waitKey()
   # print('n blobs:', str(c))
    
print('blobs:', blobs)
cv2.imshow('mask', mask)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('image', image)
cv2.waitKey()