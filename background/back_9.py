import cv2
import numpy as np 
import cv2 
import imutils

cap = cv2.VideoCapture('videos/vidrio51.mp4') 
#object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=False)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2()
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
def nothing(pos):
	pass

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 

#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	# morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# 	#
#kernel = np.ones((1,1),np.uint8)
#kernel = np.ones((2,2),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
# Remove border


while(True): 
	ret, frame = cap.read() 
	#frame = imutils.resize (frame, width=300)
	#height, weight, _ = frame.shape
	#roi = frame[340:720, 500: 800]
	#roi = frame[0:500, 0: 600]
	#roi = frame[0:500, 100: 600]
	#frame = imutils.resize(frame, 200,200)
	#frame = imutils.resize(frame, 300,200)
	# min = cv2.getTrackbarPos('min','Thresholds')
	# max = cv2.getTrackbarPos('max','Thresholds')
	# area = cv2.getTrackbarPos('area','Thresholds')
	#kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
	kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
	temp1 = 255 - cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel_vertical)
	#horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1000,1))
	temp2 = 255 - cv2.morphologyEx(frame, cv2.MORPH_CLOSE, horizontal_kernel)
	temp3 = cv2.add(temp1, temp2)
	result = cv2.add(temp3, frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# apply morphology open with square kernel to remove small white spots
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	#area_pts = np.array([[330,216], [frame.shape[1]-80,216], [frame.shape[1]-80,271], [330,271]])
	area_pts = np.array([[100,100], [400,100], [400,350], [100,350]])

	#area_pts = np.array([[150,5], [600,5], [600,400], [150,400]])
	#area_pts = np.array([[250,10], [380,10], [380,330], [250,330]])
	#area_pts_1 = np.array([[260,10], [380,10], [380,150], [260,150]])
	#area_pts_1 = np.array([[200,5], [400,5], [400,100], [200,100]])
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)

	#cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
	#cv2.rectangle(frame,(10,10),(frame.shape[1],40),(0,0,0),-1)
	#image_area = cv2.bitwise_and(gray, gray, mask=imAux)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	#image_area = cv2.bitwise_and(fgmask, fgmask, mask=imAux)
	
	#fgmask = fgbg.apply(image_area)
	#fgmask = fgbg.apply(image_area)
	#mask = object_detector.apply(frame)
	fgmask = object_detector.apply(image_area)
# 	fgmask = fgbg.apply(frame) 
	
	#fgmask = cv2.dilate(fgmask, None, iterations=4)
	fgmask = cv2.dilate(fgmask, None, iterations=1)
	#erosion = cv2.erode(frame,kernel,iterations = 1)
# threshold to binary
	thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]
	#edges = cv2.Canny(erosion, lowc, maxc)

#Encontramos los contornos presentes en fgmask, para luego basándonos
	#en su área poder determina si existe movimiento
	#cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

	#cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
	#contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
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
		if areas[i] >= 50:   #keep
			result[labels == i + 1] = 255
   
	blank = np.zeros((frame.shape), np.uint8)

	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#cv2.imshow("Binary", binary_map)
	lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=20,maxLineGap=20)
	if lines is not None:
		cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		for line in lines:
			x1,y1,x2,y2 = line[0]
			#cv2.line(result,(x1,y1),(x2,y2),(255),1)
			#cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
			cv2.line(blank,(x1,y1),(x2,y2),(255,255,255),2)
	# 		m = cv2.moments(line)
	# #m = cv2.moments(cnt)
	# 		if m["m00"] != 0:
	# 			cX = int(m["m10"] /  m["m00"]) 
	# 			cY = int(m["m01"] / m["m00"])
	# 			cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)    
	#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		#cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		# 		#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		# 		#cv2.putText
	#contours, hier = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#cv2.putText(frame, "Contours: " + str(len(contours)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	#cv2.imshow("Result", result)
	#for cnt in contours:
     
		#print(cv2.contourArea(cnt))
		#cv2.waitKey() 
		#if cv2.contourArea(cnt) > 5000:
		#if cv2.contourArea(cnt) > 10000:
		#if w > 300:
			#convexHull = cv2.convexHull(cnt)
			#cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

			#(x, y, w, h) = convexHull
			#(x, y, w, h) = cv2.boundingRect(convexHull)
		# (x, y, w, h) = cv2.boundingRect(cnt)
		# rect = cv2.minAreaRect(cnt)
		# centerX = rect[0][0]
		# box = cv2.boxPoints(rect)
		# box = np.int_(box)
		# cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
		# #cv2.drawContours(frame,[box],0,(0,222,0),2)
		# detections.append([x, y, w, h])
		# #center= box.
		# #if 450 > w > 300:
		# # calculate x,y coordinate of center
		# m = cv2.moments(cnt)
		# #m = cv2.moments(cnt)
		# if m["m00"] != 0:
		# 	cX = int(m["m10"] /  m["m00"]) 
		# 	cY = int(m["m01"] / m["m00"])
		# 	cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)    
		# 		#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		# 	cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		# 		#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		# 		#cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)), (cX,cY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
					
	
			# boxers_ids = tracker.update(detections)
			# for box_id in boxers_ids:
			# 	x, y, w, h, id = box_id
			# 	#cv2.putText(frame,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
			# 	#counter =id
			# 	#cv2.putText(frame,  str(id) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
			# 	counter_actual=id
				# global counter
				# if  counter_actual>counter:
				# 	Guardar(counter_actual)
				# 	counter = counter_actual
					
				#cv2.putText(frame,  str(counter) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
				#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
				#print(detections)					
				#cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
				#number_of_white_pix = np.sum(fgmask == 255) 
				#cv2.putText(frame, "centroid", ((int(m[0]) - 25, int(m[1]) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2))
			
	cv2.imshow('blank', blank) 
	cv2.imshow('result', result) 
	cv2.drawContours(frame, [area_pts], -1, (255,0,255), 2)
	
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 

	
	k = cv2.waitKey(80) & 0xff
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows()


