import cv2
import numpy as np
#from matplotlib import pyplot as plt

video = cv2.VideoCapture('videos/vidrio23.mp4')
def nothing(pos):
	pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('LS','Thresholds',0,255, nothing)
cv2.createTrackbar('LH','Thresholds',0,255, nothing)
  

i = 0
while True:
	ret, frame = video.read()
	if ret == False: break
	# convert all to grayscale
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,127,255,0)
	kernel = np.ones((3, 3), np.float32) / 9
	dst = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
 
	contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
 
	def find_if_close(cnt1,cnt2):
		row1,row2 = cnt1.shape[0],cnt2.shape[0]
		for i in range(row1):
			for j in range(row2):
				dist = np.linalg.norm(cnt1[i]-cnt2[j])
				if abs(dist) < 50 :
					return True
				elif i==row1-1 and j==row2-1:
					return False

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,127,255,0)
	contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))

	for i,cnt1 in enumerate(contours):
		if cv2.contourArea(cnt1) > 1000:
     
			x = i    
			if i != LENGTH-1:
				for j,cnt2 in enumerate(contours[i+1:]):
					x = x+1
					dist = find_if_close(cnt1,cnt2)
					if dist == True:
						val = min(status[i],status[x])
						status[x] = status[i] = val
					else:
						if status[x]==status[i]:
							status[x] = i+1

	unified = []
	maximum = int(status.max())+1
	for i in range(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(contours[i] for i in pos)
			hull = cv2.convexHull(cont)
			unified.append(hull)

	cv2.drawContours(frame,unified,-1,(0,255,0),2)
	cv2.drawContours(thresh,unified,-1,255,-1)
 
 
	 
	cv2.imshow('gray',gray)
	cv2.imshow('thresh',thresh)
	#cv2.imshow('fgmask',fgmask)
	cv2.imshow('Frame',frame)

	i = i+1
	if cv2.waitKey(80) & 0xFF == ord ('q'):
		break
#video.release()