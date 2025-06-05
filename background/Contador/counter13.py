import cv2
import numpy as np

# Read image
image = cv2.imread('images/area4.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray,50,150,apertureSize=3)

def nothing(pos):
    pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',147,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

#cv2.imwrite('image1.jpg',inv_img)

while (True):
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')

	#_, thresh = cv2.threshold(imgray, 127, 255, 0)
	#_, thresh = cv2.threshold(gray, 100, 255, 0)
	res,thresh=cv2.threshold(gray,lc, hc ,cv2.THRESH_BINARY) 
	#thresh = cv2.bitwise_not(thresh) # inverse the image so that objects of interest are white

	#fgmask = object_detector.apply(image_area)
	# Find the contours and remove small ones (noise)
	contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.putText(image, str(len(contours)), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	for cnt in contours:
	# cv2.waitKey()
		#area = cv2.contourArea(c)
		cv2.drawContours(image, [cnt], -1, (0,255,0), -1)
		#(x, y, w, h) = cv2.boundingRect(c)
	# cv2.putText(mask, str(blobs), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # if area > 10:
    #cv2.imwrite('detectedLines.png',image)
	# k =  cv2.waitKey() #& 0xff
	# if k == 27:
	break
cv2.imshow('thresh',thresh)
cv2.imshow('gray',gray)
cv2.imshow('image',image)
cv2.waitKey()