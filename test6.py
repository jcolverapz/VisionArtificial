import cv2
import imutils
#from  tracker  import   trackerdialog, trackerDB
#from pympler import tracker
#from  TestTracker  import  *
from funciones.tracker import *
#import EuclideanDistTracker
#ounter=0
#cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture('videos/vidrio0.mp4') 

#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, varThreshold= 40)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 100) 
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
tracker = EuclideanDistTracker() # type: ignore

#def nothing(pos):
	#pass
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


while(1): 
	ret, frame = cap.read() 
	height, weight, _ = frame.shape
	#roi = frame[340:720, 500: 800]
	roi = frame[0:500, 0: 600]
	#frame = imutils.resize #(
	mask = object_detector.apply(frame)
	fgmask = cv2.dilate(mask, None, iterations=5)
 
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	detections=[]
	for cnt in contours:
		area= cv2.contourArea(cnt)
		if area > 10000:
			cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
			x,y,w,h= cv2.boundingRect(cnt)
			cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
			detections.append([x, y, w, h])
	# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
	# screenCnt = None
	boxers_ids= tracker.update(detections)
	for box_id in boxers_ids:
		x, y, w, h, id = box_id
		cv2.putText(roi,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 3)
		cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
	print(detections)
 
 
	cv2.imshow('roi', roi ) 
	cv2.imshow('frame',frame ) 
	cv2.imshow('mask',mask ) 
     
	k = cv2.waitKey(130) & 0xff
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows() 