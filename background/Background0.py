import cv2
import numpy as np 
import cv2 
import imutils
 
cap = cv2.VideoCapture('videos/vidrio71.mp4') 
  
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
 
def nothing(pos):
	pass
  
kernel = np.ones((1,1),np.uint8)
 
while(True): 
	ret, frame = cap.read() 
	 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	 
	area_pts = np.array([[100,100], [400,100], [400,300], [100,300]])
 
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
 
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	
	fgmask = object_detector.apply(image_area)
  
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 

	k = cv2.waitKey(80) & 0xff
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows()


