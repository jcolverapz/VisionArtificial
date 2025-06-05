import cv2
import numpy as np
import cv2
import imutils
#from funciones.tracker import *
#from skimage.morphology import medial_axis
#import pyodbc

def nothing(pos):
    pass

def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance
 
counter=0

#object_detector = cv2.VideoCapture('videos/vidrio40.mp4')
cap = cv2.VideoCapture('videos/vidrio51.mp4') 
#cap.set(cv2.CAP_PROP_FPS, 1)
#cap = cv2.VideoCapture('videos/vidrio23.mp4')
#object_detector = cv2.createBackgroundSubtractorMOG2()
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=False)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# Function to index and distance of the point closest to an array of points
# borrowed shamelessly from: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40)
#fgbg = cv2.createBackgroundSubtractorMOG2()

#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40)
#tracker = EuclideanDistTracker() # type: ignore

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# 	#
# cv2.namedWindow('Thresholds')
# cv2.createTrackbar('min','Thresholds',0,255, nothing)
# cv2.createTrackbar('max','Thresholds',255,255, nothing)
# cv2.createTrackbar('area','Thresholds',0,400, nothing)
# cv2.createTrackbar('lowc','Thresholds',100,400, nothing)
# cv2.createTrackbar('maxc','Thresholds',200,400, nothing)
# #
#kernel = np.ones((1,1),np.uint8)
#kernel = np.ones((2,2),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

while(True):
	ret, frame = cap.read()
	frame = imutils.resize (frame, width=720)
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')

	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# apply morphology open with square kernel to remove small white spots
	#desenfoque
	#blur_img = cv2.GaussianBlur(gray, (3,3), 0)

	edged = cv2.Canny(gray, lc, hc)
	edged = cv2.dilate(edged, None, iterations=4)
	edged = cv2.erode(edged, None, iterations=1)

	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	# do distance transform
	#dist = cv2.distanceTransform(gray, distanceType=cv2.DIST_L2, maskSize=5)

	#area_pts_1 = np.array([[50,20], [500,20], [500,400], [50,400]])
	area_pts_1 = np.array([[100,200], [400,200], [400,300], [100,300]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	fgmask = cv2.dilate(fgmask, None, iterations=3)

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

	#contours = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	# find contours.
	contours = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] 
	# Suppose this has the contours of just the car and the obstacle.

	# create an image filled with zeros, single-channel, same size as img.
	blank = np.zeros(fgmask.shape[0:2] )
	cv2.line(blank, (300, 0), (300, 300), (0, 255, 255), 3)

	# copy each of the contours (assuming there's just two) to its own image. 
	# Just fill with a '1'.
	img1 = cv2.drawContours( blank.copy(), contours, 0, 1 )
	#img2 = cv2.drawContours( blank.copy(), contours, 1, 1 )

	rows, cols = frame.shape[:2]
	pixel_counts = [0] * cols
	for cnt in contours:
		for point in cnt:
			row, col = point[0]
			pixel_counts[col] += 1
   
	# # Find pixel row with the most contour points
	# max_count = max(pixel_counts)
	# max_col = pixel_counts.index(max_count) - 50

	# # Draw line
	# cv2.line(frame, (0, max_col), (cols-1, max_col), (0, 255, 255), 3)

	# # Find the contour with the largest area (which is assumed to be the bubble)
	#cnt = max(contours, key=cv2.contourArea)

	# # Fit a polynom to the largest contour
	# epsilon = 0.0001 * cv2.arcLength(cnt, True)
	# approx = cv2.approxPolyDP(cnt, epsilon, True)

	# # Draw the polynom fit in yellow
	# cv2.polylines(frame, [approx], True, (0, 255, 255), 2)

	# Convert line points to numpy array
	line_points = np.array([(0, max_col), (cols-1, max_col)])

	# Find intersection of line and largest contour
	# if len(cnt) > 0:
	# 	intersection_points = []
	# 	for i in range(len(cnt)):
	# 		x, y = cnt[i][0]
	# 		x = int(x)
	# 		y = int(y)
	# 		dist = cv2.pointPolygonTest(line_points, (x, y), True)

	# 		if dist == 0:
	# 			intersection_points.append((x, y))
			

	# else:
	# 	print("Error: No contour points found")
	# 	exit()

	# # Draw intersection points in green
	# for point in intersection_points:
	# 	cv2.circle(frame, point, 4, (0, 255, 0), thickness=-1)	
	# # store the 2 contours
	# 	#cv2.imshow("Binary", binary_map)
		#cv2.imshow("Result", result)
		#print(cv2.contourArea(cnt))
		# if cv2.contourArea(cnt) > 1000:
        # #if cv2.contourArea(cnt) > 10000:
        # #if w > 300:
		# 	# Draw Contours
		# 	blank_mask = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
		# 	cv2.drawContours(blank_mask, contours, -1, (0, 255, 0), 1)
		# 	contours_idx = blank_mask[...,1] == 255

		# 	# Define lines coordinates
		# 	line1 = [300, 10, 300, 300]

		# 	# Draw Lines over Contours
		# 	cv2.line(blank_mask, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0), thickness=1)
		# 	lines_idx = blank_mask[...,0] == 255
		# 	overlap = np.where(contours_idx * lines_idx)
		# 	array([ 90, 110, 110, 140, 140], dtype=int64), array([ 80,  40, 141,  27, 156], dtype=int64))
		# 	list(zip(*overlap))
	#cv2.imshow('result', output)
	#cv2.imshow('img',img)
	cv2.drawContours(frame, [area_pts_1], -1, (255,0,255), 2)

	cv2.imshow('fgmask', fgmask)
	cv2.imshow('frame',frame )


	k = cv2.waitKey(100) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
