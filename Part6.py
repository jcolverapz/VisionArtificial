import cv2
import numpy as np

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/vidrio23.mp4')

def nothing(pos):
	pass
cv2.namedWindow('Thresholds')
cv2.createTrackbar('LS','Thresholds',0,255, nothing)
cv2.createTrackbar('LH','Thresholds',0,255, nothing)

while True:

	ret, frame  = cap.read()
	ls=cv2.getTrackbarPos('LS','Thresholds')
	lh=cv2.getTrackbarPos('LH','Thresholds')
 
	hsv = cv2.cvtColor(frame ,cv2.COLOR_BGR2HSV)

	lower = np.array([ls,200,20])
	upper = np.array([lh,250,250])

	mask = cv2.inRange(hsv, lower, upper)

	contors, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	#if contors:
		#contors = max(contors, key=cv2.contourArea)
	for cnt in contors:
		x,y,w,h = cv2.boundingRect(cnt)

		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

		cv2.drawContours(frame, cnt, -1, (0,225,0), 2)
		cv2.waitKey()
  
	cv2.imshow("frame", frame)
	cv2.imshow("hsv", hsv)
	cv2.imshow("mask", mask)

	if cv2.waitKey(100) == 27:
		cap.release()
		cv2.destroyAllWindows()
		break