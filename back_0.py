import cv2
import numpy as np 
import imutils
from tracker import *

#cap = cv2.VideoCapture(0) 
#cap.set(cv2.CAP_PROP_FPS, 1)
cap = cv2.VideoCapture('videos/vidrio51.mp4') 
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=False)
def nothing(pos):
    pass
def drawlines(original_img, lines):
        try:

            img = cv2.polylines(original_img, [lines], False, (0, 255, 0), 4)
            return img
        
        except:
            return original_img
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# Function to index and distance of the point closest to an array of points
# borrowed shamelessly from: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#kernel = np.ones((1,1),np.uint8)
#kernel = np.ones((2,2),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
# morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)
tracker = EuclideanDistTracker() # type: ignore
ret, frame = cap.read() 
area_pts = cv2.selectROI("Frame", frame, fromCenter=False,  showCrosshair=True)

TopLeft = (area_pts[0],area_pts[1])
TopRight = ((area_pts[0]+ area_pts[2]),area_pts[1])
BotRight = ((area_pts[0]+ area_pts[2]), (area_pts[1]+ area_pts[3]))
BotLeft = (area_pts[0],(area_pts[1]+ area_pts[3]))

while(True): 
	ret, frame = cap.read() 
	#frame = imutils.resize (frame, width=720)
 
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')

	# Girar horizontalmente
    #frame = cv2.flip(frame, 1) 
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# apply morphology open with square kernel to remove small white spots

#desenfoque
	#blur_img = cv2.GaussianBlur(gray, (3,3), 0)

	edged = cv2.Canny(gray, lc, hc)
	#edged = cv2.dilate(edged, None, iterations=2)
	#edged = cv2.erode(edged, None, iterations=1)

	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	
	#area_pts_1 = np.array([[100,150], [320,150], [320,380], [100,380]])
	area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	#morfologia
	fgmask = cv2.dilate(fgmask, None, iterations=3)
    #fgmask = fgbg.apply(image_area)
    #fgmask = fgbg.apply(image_area)
	# 	fgmask = fgbg.apply(frame) 
 
	#fgmask = cv2.dilate(fgmask, None, iterations=4)
    #cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
    #cv2.rectangle(frame,(10,10),(frame.shape[1],40),(0,0,0),-1)
    #image_area = cv2.bitwise_and(gray, gray, mask=imAux)
    #image_area = cv2.bitwise_and(fgmask, fgmask, mask=imAux)
    
    #mask = object_detector.apply(frame)
    
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

	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#print(len(contours))

	output = frame.copy()
	# Add borders to prevent skeleton artifacts:
	borderThickness = 2
	borderColor = (0, 0, 0)
	grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
										cv2.BORDER_CONSTANT, None, borderColor)
	# Compute the skeleton:
	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
	# do connected components processing
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, None, None, None, 8, cv2.CV_32S)

	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)
	# # display skeleton
	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= 50:   #keep
			result[labels == i + 1] = 255
   
	contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
	for cnt in contours:
		if cv2.contourArea(cnt) > 100:
			cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
			#perimeter = cv2.arcLength(cnt, True)
			#perimeter = cv2.arcLength(cnt, False)
			# approximatedShape = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
			# cv2.drawContours(output, [approximatedShape], -1, (0, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
			# (x, y, w, h) = convexHull
			# (x, y, w, h) = cv2.boundingRect(convexHull)
			(x, y, w, h) = cv2.boundingRect(cnt)
			
			#approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
			#cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
  
			detections.append([x, y, w, h])

			#cv2.line(frame, ((cX, cY)), (endpt_x, endpt_y), 255, 2)  
			#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
			#cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)), (cX,cY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
			
		#Draw a diagonal blue line with thickness of 5 px
			#cv2.line(img,(centerX,centerY),(511,511),(255,0,0),5)    

			boxers_ids = tracker.update(detections)
			for box_id in boxers_ids:
				x, y, w, h, id = box_id
				cv2.putText(frame,  str(id) , (x, y),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
				#counter =id
				#cv2.putText(frame,  str(id) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
				counter_actual=id
				global counter
				#if  counter_actual>counter:
					#Guardar(counter_actual)
 
  # Create a black image
	# img = np.zeros((512,512,3), np.uint8)
	# i=0
	# eps=0.2
 
	#cv2.putText(frame, "Contours: " + str(len(contours)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
  
	# lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
	# if lines is not None:
	# 	cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	# 	for line in lines:
	# 		x1,y1,x2,y2 = line[0]
	# 		cv2.line(result,(x1,y1),(x2,y2),(255),1)
	# 		cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
   
	#contours, hier = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# for c in contours:
	# 	area = cv2.contourArea(c)
	# 	#if area > 100:
			
			#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
			#lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=40,maxLineGap=80)
 
	# for cnt in contours:
	# 	if cv2.contourArea(cnt) > 100:
	# 		print()
	# 		(x, y, w, h) = cv2.boundingRect(cnt)
	# 		rect = cv2.minAreaRect(cnt)
	# 		centerX = rect[0][0]
	# 		box = cv2.boxPoints(rect)
	# 		box = np.int_(box)
	# 		#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	# 		cv2.drawContours(frame,[box],0,(0,0,255),2)	   
			
			#cv2.waitKey()
			 
	cv2.drawContours(frame, [area_pts], -1, (0,255,0), 2)
	cv2.imshow("skeleton",skeleton)
	cv2.imshow('result', result) 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 


	k = cv2.waitKey(80) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()
					#counter = counter_actual
					
				#cv2.putText(frame,  str(counter) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
				#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
				#print(detections)					
				#cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
				#number_of_white_pix = np.sum(fgmask == 255) 
				#cv2.putText(frame, "centroid", ((int(m[0]) - 25, int(m[1]) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2))
			
	#cv2.imshow('img', img) 
	#cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0,255,0), 2)
	#cv2.putText(frame, str(counter), (500,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	#cv2.line(fgmask, (300,0), (300,500), (255, 255, 255), 2)
	#cv2.line(frame, (350,0), (350,500), (0, 255, 255), 1)
	#cv2.line(frame, (400,0), (400,500), (0, 255, 255), 1)