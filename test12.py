import cv2
import numpy as np
 
video = cv2.VideoCapture('videos/vidrio51.mp4')

def nothing(pos):
	pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('LS','Thresholds',160,255, nothing)
cv2.createTrackbar('LH','Thresholds',255,255, nothing)

i = 0
while True:
	ret, frame = video.read()
	if ret == False: break
	ls=cv2.getTrackbarPos('LS','Thresholds')
	lh=cv2.getTrackbarPos('LH','Thresholds')
 
 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if i == 5:
		bgGray = gray
	if i > 20:
		dif = cv2.absdiff(gray, bgGray)
		_, th = cv2.threshold(dif, ls, lh, cv2.THRESH_BINARY)
		
		cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		for c in cnts:
			area = cv2.contourArea(c)
			if area > 10:
				x,y,w,h = cv2.boundingRect(c)
				cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)

		cv2.imshow('dif',dif)
		cv2.imshow('Frame',frame)

	i = i+1
	if cv2.waitKey(100) & 0xFF == ord ('q'):
		break
video.release()