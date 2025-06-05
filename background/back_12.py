import cv2
import numpy as np 
import cv2 
import imutils
import os
from datetime import datetime
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
 
counter=0
i=0

def nothing(pos):
    pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',100,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

def process_image4(original_image):  # Douglas-peucker approximation

    # Contour approximation
	# try:  # Just to be sure it doesn't crash while testing!
	# 	for cnt in contours:
	# 		epsilon = 0.005 * cv2.arcLength(cnt, True)
	# 		approx = cv2.approxPolyDP(cnt, epsilon, True)
	# 		# cv2.drawContours(modified_image, [approx], -1, (0, 0, 255), 3)
	# except:
	# 	pass
	try:  # Just to be sure it doesn't crash while testing!
		for contour in contours:
			epsilon = 0.009 * cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, epsilon, closed=True)
			cv2.drawContours(original_image, [approx], -1, (0, 0, 255), 1)
	except:
			pass
	

	return original_image

# 	lc = cv2.getTrackbarPos('lc','Thresholds')
# 	hc = cv2.getTrackbarPos('hc','Thresholds')
 
#     # Convert to black and white threshold map
# 	gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# 	#gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 	#(thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 	#(thresh, bw) = cv2.threshold(gray, lc, hc, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 	#area_pts_1 = np.array([[100,5], [650,5], [650,400], [100,400]])

# 	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
# 	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
# 	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
# 	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
# 	#fgmask = object_detector.apply(image_area)
# 	#fgmask = cv2.dilate(fgmask, None, iterations=5)
# 	# Convert bw image back to colored so that red, green and blue contour lines are visible, draw contours
# 	# modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
# 	#modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
# 	contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 	cv2.drawContours(original_image, contours, -1, (0, 0, 255), 3)
# # def getSkeletonIntersection(skeleton):
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

while(True): 
	ret, frame = cap.read() 
	#frame = imutils.resize (frame, width=720)
    #frame = cv2.flip(frame, 1) # Girar horizontalmente
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#gray = cv2.GaussianBlur(gray, (1,1), 0) #desenfoque

	edged = cv2.Canny(gray, 100, 255)
	edged = cv2.dilate(edged, None, iterations=101)
	edged = cv2.erode(edged, None, iterations=101)

	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	
	area_pts_1 = np.array([[100,100], [300,100], [300,350], [100,350]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
 
 
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]
	result = np.zeros((labels.shape), np.uint8)
	# # display skeleton
	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= 50:   #keep
			result[labels == i + 1] = 255
 
	#fgmask = cv2.dilate(fgmask, None, iterations=5)
	contours = cv2.findContours(fgmask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#cv2.drawContours(frame, contours, -1, (0,255,0), 2)
	cv2.putText(frame, "contours: " + str(len(contours)) , (100, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
 
 
	image = process_image4(frame)
 
 
	cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
			 
	cv2.imshow('window', image)
	cv2.imshow('result',result ) 
	cv2.imshow('frame',frame ) 
	#cv2.imshow('edged', edged) 
	cv2.imshow('fgmask', fgmask) 
	
	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()

 