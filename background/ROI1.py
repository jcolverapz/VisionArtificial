import cv2
import numpy as np 
import imutils
import math

def nothing(pos):
    pass

cap = cv2.VideoCapture('videos/vidrio51.mp4')
#cap = cv2.VideoCapture(0)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=100,nmixtures=5, backgroundRatio=0.0001) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=0,nmixtures=200, backgroundRatio=0.001, noiseSigma=10) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.001, noiseSigma=0) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.1, noiseSigma=0.001) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
object_detector = cv2.createBackgroundSubtractorKNN()
#object_detector = cv2.bgsegm.createBackgroundSubtractorGMG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=0)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel = np.ones((2,2),np.uint8)# apply morphology open with square kernel to remove small white spots

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

ret, frame = cap.read()

#area_pts = cv2.selectROI("Frame", frame, fromCenter=False,  showCrosshair=True)
  
# Select ROI 
area_pts = cv2.selectROI("select the area", frame) 

while(True): 
	ret, frame = cap.read() 
  
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')
 
	size = np.size(frame) 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	 
	
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	_,thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

	#area_pts_1 = np.array([[150,5], [650,5], [650,450], [150,450]])
	#area_pts_1 = np.array([[100,20], [600,20], [600,350], [100,350]])
	#area_pts = np.array([[200,100], [400,100], [400,200], [200,200]])

	# imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	# #imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
	# x,y,w,h = area_pts
	# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	# fgmask = object_detector.apply(image_area)
	#(x, y, w, h) = cv2.boundingRect(area_pts)
	rect=cv2.boundingRect=area_pts
	# rect1 = cv2.minAreaRect(rect)
	# centerX = rect[0][0]
	# box = cv2.boxPoints(rect)
	# box = np.int_(box)
	#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.drawContours(frame,[rect][0],0,(0,0,255),2)	  
	 
	cv2.imshow('thresh', thresh) 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 

	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()


     