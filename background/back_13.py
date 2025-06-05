import cv2
import numpy as np 
import cv2 
import imutils
#from funciones.tracker import *
#from skimage import medial_axis
#from skimage.morphology import medial_axis
#import pyodbc
import os
from datetime import datetime
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
Datos = 'objects'
if not os.path.exists(Datos):
    print('Carpeta creada: ',Datos)
    os.makedirs(Datos)
counter=0
i=0

def nothing(pos):
    pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

def process_image4(frame):  # Douglas-peucker approximation
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')
 
    # Convert to black and white threshold map
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray, (5, 5), 0)
	(thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#(thresh, bw) = cv2.threshold(gray, lc, hc, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#area_pts_1 = np.array([[100,5], [650,5], [650,400], [100,400]])

	#imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	#imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	#image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = object_detector.apply(image_area)
	#fgmask = cv2.dilate(fgmask, None, iterations=5)
	# Convert bw image back to colored so that red, green and blue contour lines are visible, draw contours
	# modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
	#modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
	contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Contour approximation
	try:  # Just to be sure it doesn't crash while testing!
		for cnt in contours:
			epsilon = 0.005 * cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, epsilon, True)
			cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)
	except:
		pass
	# try:  # Just to be sure it doesn't crash while testing!
	# 	for contour in contours:
	# 		epsilon = 0.009 * cv2.arcLength(contour, True)
	# 		#approx = cv2.approxPolyDP(contour, epsilon, closed=True)
	# 		approx = cv2.approxPolyDP(contour, epsilon, closed=False)
	# 		cv2.drawContours(frame, [approx], -1, (0, 255, 255), 3)
	# except:
	# 		pass
	
	return frame
 
cap = cv2.VideoCapture('videos/vidrio20.mp4') 

#cap = cv2.VideoCapture(0) 
object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=True)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 

kernel = np.ones((2,2),np.uint8)# apply morphology open with square kernel to remove small white spots
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#kernel = np.ones((1,1),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1000))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

while(True): 
	ret, frame = cap.read() 
 
	#frame = imutils.resize (frame, width=720)
    #frame = cv2.flip(frame, 1) # Girar horizontalmente
    
	#size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# #gray = cv2.GaussianBlur(gray, (1,1), 0) #desenfoque

	# edged = cv2.Canny(gray, 100, 255)
	# edged = cv2.dilate(edged, None, iterations=101)
	# edged = cv2.erode(edged, None, iterations=101)

	# fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	
	area_pts_1 = np.array([[100,5], [650,5], [650,350], [100,350]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	#fgmask = cv2.dilate(fgmask, None, iterations=5)
  
	image = process_image4(frame)
 
	#cv2.imshow('window', image)
	#cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
			 
	cv2.imshow('frame',frame ) 
	cv2.imshow('fgmask', fgmask) 
	
	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()

 