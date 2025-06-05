import cv2
import numpy as np 
import imutils
import math

def nothing(pos):
    pass

#cap = cv2.VideoCapture('videos/vidrio51.mp4')
#cap = cv2.VideoCapture(0)
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=100,nmixtures=5, backgroundRatio=0.0001) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=0,nmixtures=200, backgroundRatio=0.001, noiseSigma=10) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.001, noiseSigma=0) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.1, noiseSigma=0.001) #createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorKNN()
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

area_pts = cv2.selectROI("Frame", frame, fromCenter=False,  showCrosshair=True)

TopLeft = (area_pts[0],area_pts[1])
TopRight = ((area_pts[0]+ area_pts[2]),area_pts[1])
BotRight = ((area_pts[0]+ area_pts[2]), (area_pts[1]+ area_pts[3]))
BotLeft = (area_pts[0],(area_pts[1]+ area_pts[3]))


#extRight = tuple(area_pts[area_pts[:, :, 0].argmax()][0])
#extTop = tuple(area_pts[area_pts[:, :, 1].argmin()][0])
#extBot = tuple(area_pts[area_pts[:, :, 1].argmax()][0])
   
#area_pts_1 = np.array([area_pts[0],area_pts[1]], [area_pts[2],area_pts[1]], [area_pts[2],area_pts[3]], [area_pts[0],area_pts[3]])
#area_pts_1 = np.array([extLeft])
#area_pts_1 = np.array(extLeft, extRight, extTop, extBot)
#print(area_pts_1)
while(True): 
	ret, frame = cap.read() 
  
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')
 
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	edged = cv2.Canny(gray, lc, hc)
	#edged = cv2.dilate(edged, None, iterations=4)
	
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	_,thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

	area_pts_1 = np.array([TopLeft, TopRight, BotRight , BotLeft])
	#area_pts_1 = np.array([[100,20], [600,20], [600,350], [100,350]])
	#area_pts = np.array([[200,100], [400,100], [400,200], [200,200]])
	#area_pts_1 = np.array([[100,100], [300,100], [300,300], [100,300]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	#x,y,w,h = area_pts
	#rect=cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	#rect=cv2.boundingRect(x,y,w,h)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	#cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
    #cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

	#detections = []

	#convert to binary by thresholding
	 
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)
	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]
	result = np.zeros((labels.shape), np.uint8)
 
	lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
	#cv2.putText(frame, "lines: " + str(len(lines)) , (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
	# print(len(lines))
	if lines is not None:
		cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		for line in lines:
			x1,y1,x2,y2 = line[0]
			#cv2.line(result,(x1,y1),(x2,y2),(255),1)
			cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
   
   
	# #lineas de hough
	# lines = cv2.HoughLines(result, 1, np.pi / 180, 150, None, 100, 100)
	# print(len(lines))
	# if lines is not None:
	# 	for i in range(0, len(lines)):
	# 		rho = lines[i][0][0]
	# 		theta = lines[i][0][1]
	# 		a = math.cos(theta)
	# 		b = math.sin(theta)
	# 		x0 = a * rho
	# 		y0 = b * rho
	# 		pt1 = (int(x0 + 100*(-b)), int(y0 + 100*(a)))
	# 		pt2 = (int(x0 - 100*(-b)), int(y0 - 100*(a)))
	# 		cv2.line(frame, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= 10:   #keep
			result[labels == i + 1] = 255

	cv2.imshow("result",result) 
 
	# output = frame.copy()
	# # Add borders to prevent skeleton artifacts:
	# borderThickness = 4
	# borderColor = (0, 0, 0)
	# grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
	# 									cv2.BORDER_CONSTANT, None, borderColor)
	# #Compute the skeleton:
	# skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
	 # # Draw Contours
	#blank_mask = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
	# contours_idx = blank_mask[...,1] == 255
	# # Define lines coordinates
	#line1 = [300, 10, 300, 300]
	 
	# # Draw Lines over Contours
	#cv2.line(blank_mask, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0), thickness=1)
	#eps=0.1
	
	contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(frame, [contours], -1, (0,255,0), 4); # Draw the biggest contour
	
	# #contours = contours[0] if len(contours) == 2 else contours[1]
	# cv2.putText(frame, "contours: " + str(len(contours)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	# #biggest_contour = max(contours, key = len)
	# # for cnt in contours:
	#if len(contours)>1:
	# 	#if biggest_contour.size > 100:
	# 		cv2.drawContours(frame, [contours], -1, (0,255,0), 4); # Draw the biggest contour
		#cv2.putText(frame, "biggest: " + str(cv2.contourArea(biggest_contour)) , (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
	# 		#centerX = rect[0][0]
	# 		# box = cv2.boxPoints(rect)
	# 		(x, y, w, h) = cv2.boundingRect(cnt)
	# 		# box = np.int_(box)
	# 		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
			#cv2.drawContours(frame,[box],0,(0,0,255),2)	   
		#biggest_contour = max(contours, key = cv2.contourArea) # Retreive the biggest contour


		# rect = cv2.minAreaRect(cnt)

	# 	cv2.drawContours(frame, contour, -1, (0,255,0), 3)
	# 	print(contour)	
			#cv2.imshow("Contours", contourImg)
			#cv2.waitKey()
		# area = cv2.contourArea(cnt)
		# convex_hull = cv2.convexHull(cnt)
		# convex_hull_area = cv2.contourArea(convex_hull)
		# ratio = area / convex_hull_area
		# cv2.drawContours(frame, [convex_hull], -1, (0,255,0), 2)
  
		#print(index, area, convex_hull_area, ratio)
		#x,y,w,h = cv2.boundingRect(cntr)
		#cv2.putText(label_img, str(index), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
		# if ratio < 0.91:
		# # cluster contours in red
		# #cv2.drawContours(contour_img, [cntr], 0, (0,0,255), 2)
		# 	cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
		# 	#cluster_count = cluster_count + 1
		# else:
		# # isolated contours in green
		# 	cv2.drawContours(frame, [cnt], -1, (0,255,255), 2)
		# #cv2.drawContours(contour_img, [cntr], 0, (0,255,0), 2)
			#isolated_count = isolated_count + 1
			#index = index + 1
		
	# params = cv2.SimpleBlobDetector_Params()

	# # Change thresholds
	# params.minThreshold = 10;
	# params.maxThreshold = 255;

	# #blur = cv2.GaussianBlur(img,(5,5),0)

	# #params.filterByCircularity = True
	# params.filterByCircularity = False
	# params.minCircularity = 0.2

	# params.filterByArea = True;
	# params.minArea = 10;

	# # ver = (cv2.__version__).split('.')
	# # if int(ver[0]) < 3 :
	# #     detector = cv2.SimpleBlobDetector(params)
	# # else :
	# detector = cv2.SimpleBlobDetector_create(params) #version 4

	# Set up the detector with default parameters.
	#detector = cv2.SimpleBlobDetector()

	# # Detect blobs.
	# keypoints = detector.detect(result)

	# # Draw detected blobs as red circles.
	# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	# im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	#[x,y,w,h] = area_pts
	# x, y, w, h = cv2.boundingRect(area_pts)
	# cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
	# 	cv2.drawContours(frame, contour, -1, (0,255,0), 3)
	# 	#if cv2.contourArea(cnt) > 3000: 
	# 	#cv2.imshow("skeleton",skeleton)
	#cv2.drawContours(frame, area_pts[0], -1, (255,0,255), 2)
	cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
	#cv2.rectangle(frame, imAux, (0, 255, 0), 2)
  
	cv2.imshow('edged', edged) 
	cv2.imshow('thresh', thresh) 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 

	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()


     