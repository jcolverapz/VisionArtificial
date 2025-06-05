import cv2
import numpy as np 
import cv2 
import imutils
#from tracker import *
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
# def getSkeletonIntersection(skeleton):
#     image = skeleton.copy();
#     image = image/255;
#     intersections = list();
#     for y in range(1,len(image)-1):
#         for x in range(1,len(image[y])-1):
#             if image[y][x] == 1:
#                 neighbourCount = 0;
#                 neighbours = neighbourCoords(x,y);
#                 for n in neighbours:
#                     if (image[n[1]][n[0]] == 1):
#                         neighbourCount += 1;
#                 if(neighbourCount > 2):
#                     print(neighbourCount,x,y);
#                     intersections.append((x,y));
#     return intersections;
#cap = cv2.VideoCapture(0) 
#cap.set(cv2.CAP_PROP_FPS, 1)
cap = cv2.VideoCapture('videos/vidrio51.mp4') 

#cap = cv2.VideoCapture(0) 
object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=True)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# Function to index and distance of the point closest to an array of points
# borrowed shamelessly from: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#tracker = EuclideanDistTracker() # type: ignore

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#kernel = np.ones((1,1),np.uint8)
kernel = np.ones((2,2),np.uint8)# apply morphology open with square kernel to remove small white spots
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1000))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

while(True): 
	ret, frame = cap.read() 
	frame = imutils.resize (frame, width=720)
 
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')

	 
    #frame = cv2.flip(frame, 1) # Girar horizontalmente
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#gray = cv2.GaussianBlur(gray, (1,1), 0) #desenfoque

	edged = cv2.Canny(gray, lc, hc)
	#edged = cv2.dilate(edged, None, iterations=4)
	#edged = cv2.erode(edged, None, iterations=2)

	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	
	area_pts_1 = np.array([[100,5], [650,5], [650,450], [100,450]])
	#area_pts_1 = np.array([[100,5], [600,5], [600,410], [100,410]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	#fgmask = cv2.dilate(fgmask, None, iterations=5)
 
    #cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
    #cv2.rectangle(frame,(10,10),(frame.shape[1],40),(0,0,0),-1)
    #image_area = cv2.bitwise_and(gray, gray, mask=imAux)
    #image_area = cv2.bitwise_and(fgmask, fgmask, mask=imAux)
    
    #fgmask = fgbg.apply(image_area)
    #fgmask = fgbg.apply(image_area)
    
    #mask = object_detector.apply(frame)
# 	fgmask = fgbg.apply(frame) 
    
    #fgmask = cv2.dilate(fgmask, None, iterations=4)
# threshold to binary
    #thresh = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)[1]
    #edges = cv2.Canny(erosion, lowc, maxc)

#Encontramos los contornos presentes en fgmask, para luego basándonos
    #en su área poder determina si existe movimiento
    #cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    #cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    #contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #contours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
	#f.write("Number of white pixels:"+ "\n")
	detections = []

	#convert to binary by thresholding
	#ret, binary_map = cv2.threshold(src,127,255,0)
	# do connected components processing
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= 500:   #keep
			result[labels == i + 1] = 255
 
	cnts = cv2.findContours(result , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	for i in range(len(cnts)):
		min_dist = max(result.shape[0], result.shape[1])
		cl = []
		
		ci = cnts[i]
		ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
		ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
		ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
		ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
		ci_list = [ci_bottom, ci_left, ci_right, ci_top]
    
		for j in range(i + 1, len(cnts)):
			cj = cnts[j]
			cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
			cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
			cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
			cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
			cj_list = [cj_bottom, cj_left, cj_right, cj_top]
			
			for pt1 in ci_list:
				for pt2 in cj_list:
					dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))     #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
					if dist < min_dist:
						min_dist = dist             
						cl = []
						cl.append([pt1, pt2, min_dist])
		if len(cl) > 0:
			cv2.line(result, cl[0][0], cl[0][1], (255, 255, 255), thickness = 1)

	#output = result.copy()
	contours = cv2.findContours(result , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	for cnt in contours:
		#cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
		(x, y, w, h) = cv2.boundingRect(cnt)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
		m = cv2.moments(cnt) # calculate x,y coordinate of center
		detections.append([x, y, w, h])
		cv2.waitKey()
			 
	cv2.imshow('frame',frame ) 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow("result",result) 
	#cv2.imshow('Image', output)
 
	cv2.waitKey(0)
 


	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()


#Grabar captura de pantalla
    
     