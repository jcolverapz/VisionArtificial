import cv2
import numpy as np 
import cv2 
import imutils
#from funciones.tracker import *
#from skimage.morphology import medial_axis
#import pyodbc

counter=0
 
def nothing(pos):
    pass
#cap = cv2.VideoCapture(0) 
#cap.set(cv2.CAP_PROP_FPS, 1)
cap = cv2.VideoCapture('videos/vidrio51.mp4') 
#object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=False)
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 

#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#tracker = EuclideanDistTracker() # type: ignore

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#kernel = np.ones((1,1),np.uint8)
#kernel = np.ones((2,2),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
# # morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	fgmask = cv2.dilate(fgmask, None, iterations=3)
	
	#blur_img = cv2.GaussianBlur(gray, (3,3), 0) #desenfoque
	#area_pts_1 = np.array([[300,20], [400,20], [400,400], [300,400]])
	#area_pts_1 = np.array([[100,200], [400,200], [400,300], [100,300]])
	area_pts_1 = np.array([[200,110], [300,110], [300,350], [200,350]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
 
 
	# line_img = np.zeros_like(fgmask) #Imagen con Linea de referencia
	# cv2.line(line_img, (360, 10), (360, 400), (255,255,255), 3)
	# cv2.imshow("line_img", line_img)
 
	# result_img = cv2.bitwise_xor(fgmask, line_img)
	# contours1 = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	# cv2.fillPoly(frame, contours1, 255)
	# cv2.drawContours(frame, contours1, -1, (0, 255, 255), 3)
	# print(len(contours1))
	# #print(len(contours1[0]))
	#(x1, y1, w1, h1) = cv2.boundingRect(contours1)
	#)
# 	if len(contours1)>0:
# #		cv2.line(line_img, (contours1[0][0],10), (contours1[0][0], 400), (255,255,255), 3)
# 		cv2.imshow("result_img", result_img)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	# do distance transform
	#dist = cv2.distanceTransform(gray, distanceType=cv2.DIST_L2, maskSize=5)
	#cv2.fillPoly(blank.copy(), [contour1], 1)
  # channel_count = img.shape[2]
	#match_mask_color = (255,) * channel_count 
 #Fill inside the polygon
	#mask = np.zeros_like(img)
	#cv2.fillPoly(mask, vertices, match_mask_color)
    # Returning the image only where mask pixels match
	#masked_image = cv2.bitwise_and(img, mask)
	#return masked_image
	edges = cv2.Canny(gray, lc, hc)
	edges = cv2.dilate(edges, None, iterations=4)
	edges = cv2.erode(edges, None, iterations=1)

	dialation = cv2.dilate(edges, kernel, iterations=1)
   
	#convert to binary by thresholding
	#ret, binary_map = cv2.threshold(src,127,255,0)
	# do connected components processing
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

	# #get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
	# 	#if areas[i] >= 100:   #keep
		if areas[i] >= 50:   #keep
			result[labels == i + 1] = 255

	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#print(len(contours))

	output = frame.copy()
	# Add borders to prevent skeleton artifacts:
	borderThickness = 1
	borderColor = (0, 0, 0)
	grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
										cv2.BORDER_CONSTANT, None, borderColor)
	# Compute the skeleton:
	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
  
	cv2.imshow("skeleton",skeleton) 
	detections =[]
 
	lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=100,maxLineGap=50)
	if lines is not None:
		cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		for line in lines:
			x1,y1,x2,y2 = line[0]
			cv2.line(result,(x1,y1),(x2,y2),(255),1)
			cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
   
	contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	for cnt in contours:
	# store the 2 contours
		#cv2.imshow("Binary", binary_map)
		#cv2.imshow("Result", result)
		#print(cv2.contourArea(cnt))
		if cv2.contourArea(cnt) > 100:
			cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)) , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        #if cv2.contourArea(cnt) > 10000:
				# convexHull = cv2.convexHull(cnt)
			cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 1)
			# perimeter = cv2.arcLength(cnt, True)
			# approximatedShape = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
			#cv2.drawContours(output, [approximatedShape], -1, (0, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
			#cv2.drawContours(output, [convexHull], -1, (0, 255, 0), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
			#(x, y, w, h) = convexHull
			#(x, y, w, h) = cv2.boundingRect(convexHull)
			(x, y, w, h) = cv2.boundingRect(cnt)
			rect = cv2.minAreaRect(cnt)
			centerX = rect[0][0]
			box = cv2.boxPoints(rect)
			box = np.int_(box)
   
			cv2.waitKey()
				#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
				# cv2.drawContours(frame,approximatedShape,0,(0,255,0),1)
				# if w > 300:
				# 	detections.append([x, y, w, h])
		
				# 	cv2.line(cnt[1][0],cnt[1][0],(511,511),(255,0,0),5)    
			
			# extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
			# extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
			# extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
			# extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
			# cv2.circle(frame, extLeft, 4, (0, 255, 0), -1)
			# cv2.circle(frame, extRight, 4, (0, 255, 0), -1)
			# cv2.circle(frame, extTop, 4, (0, 255, 0), -1)
			# cv2.circle(frame, extBot, 4, (0, 255, 0), -1)
# show the output image
			# extLeft = tuple(c[c[:, :, 0].argmin()][0])
			# extRight = tuple(c[c[:, :, 0].argmax()][0])
			# extTop = tuple(c[c[:, :, 1].argmin()][0])
			# extBot = tuple(c[c[:, :, 1].argmax()][0])
	# draw the outline of the object, then draw each of the
	# extreme points, where the left-most is red, right-most
	# is green, top-most is blue, and bottom-most is teal
		# 	cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
		# 	cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
		# 	cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
		# 	cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
		# 	cv2.circle(frame, extBot, 8, (255, 255, 0), -1)
			
		# # Draw a diagonal blue line with thickness of 5 px
			
				#if 450 > w > 300:
			# calculate x,y coordinate of center
			m = cv2.moments(cnt)
			#m = cv2.moments(cnt)
			#cv2.waitKey()
			# set up cross for tophat skeletonization
			if m["m00"] != 0:
				cX = int(m["m10"] /  m["m00"]) 
				cY = int(m["m01"] / m["m00"])
				cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)
					#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
				cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
				#cv2.line(frame, ((cX, cY)), (endpt_x, endpt_y), 255, 2)  
					#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
						#cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)), (cX,cY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

				# boxers_ids = tracker.update(detections)
				# for box_id in boxers_ids:
				# 	x, y, w, h, id = box_id
				# 	cv2.putText(frame,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
				# 	#counter =id
				# 	#cv2.putText(frame,  str(id) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
				# 	counter_actual=id
	#cv2.floodFill(edges, fgmask, (0,0), 123)		 
	#cv2.imshow('result', result) 
	#cv2.imshow('img', img) 
	cv2.drawContours(frame, [area_pts_1], -1, (255,0,255), 2)
	 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 
	cv2.imshow('result',result ) 
	#cv2.imshow("contour_img",contour_img) 


	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()
