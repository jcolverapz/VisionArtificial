import cv2
import imutils
import numpy as np
import os

cap = cv2.VideoCapture(0)  
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
 
kernel = np.ones((2,2),np.uint8) 

while(True): 
	ret, frame = cap.read() 
	 
	cv2.imshow('frame',frame ) 
 
	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()

