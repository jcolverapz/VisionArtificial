# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2 
import imutils

#cap = cv2.VideoCapture('/home/sourabh/Downloads/people-walking.mp4') 
cap = cv2.VideoCapture('videos/vidrio0.mp4') 
#cap = cv2.VideoCapture('videos/vidrio0.mp4') 
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorMOG2() 
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
def nothing(pos):
	pass
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel = np.ones((5,5),np.uint8)
cv2.namedWindow('Thresholds')
cv2.createTrackbar('value','Thresholds',0,255,nothing)

while(1): 
	#img, frame = cap.read() 
	_, frame = cap.read() 
	#frame = imutils.resize (frame, width=720)
	#img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
	#fgmask = fgbg.apply(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	value = cv2.getTrackbarPos('value','Thresholds')
    # Set threshold and maxValue
	thresh = 127
	maxValue = 255	
	retval, thresh = cv2.threshold(frame, thresh, maxValue, cv2.THRESH_BINARY_INV)
	#ret, thresh = cv2.threshold(gray, value, 200, 0, cv2.THRESH_BINARY)
	#fgmask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	countours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#thresh=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#_, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	cv2.drawContours(frame,countours,-1,(0,255,0),3)
	cv2.imshow("Contour",frame)
# 	for cnt in contours:
# # 		#print(cv2.contourArea(cnt))
# # 		#if cv2.contourArea(cnt) > 10000:
# # 		if cv2.contourArea(cnt) > 1000:
# 		(x, y, w, h) = cv2.boundingRect(cnt)
# # 			#band1 = True
# 		cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
# # 			number_of_white_pix = np.sum(fgmask == 255) 
	
	cv2.imshow("threshold", thresh)
	#cv2.imshow("frame", frame)
	#cv2.waitKey(0)
  
	k = cv2.waitKey(120) & 0xff
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows() 
