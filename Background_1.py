import cv2
import numpy as np 
import cv2 
import imutils
from tracker import *

import pyodbc

counter=0
detections = []
id=0
# def Guardar(param):
# 	user='MXAPDAPP'
# 	password='Sch0tt!'
# 	database='Gemtron'
# 	port='1433'
# 	TDS_Version='8.0'
# 	server='10.18.172.2'
# 	driver='SQL SERVER'
# 	con_string='UID=%s;PWD=%s;DATABASE=%s;PORT=%s;TDS=%s;SERVER=%s;driver=%s' % (user,password, database,port,TDS_Version,server,driver)
# 	cnxn=pyodbc.connect(con_string)
# 	cursor=cnxn.cursor()

# 	cursor.execute("""UPDATE Tbl_Conteos_Estados set PzMov=? where Con_Linea=331""", (param))

# 	cnxn.commit()
# 	cnxn.close()

def DetectorMovimiento():
	fr = 0
	#cap = cv2.VideoCapture(0) 
	#cap.set(cv2.CAP_PROP_FPS, 1)
	cap = cv2.VideoCapture('videos/vidrio51.mp4') 
	ret, frame = cap.read()

	#area_pts = cv2.selectROI("Frame", frame, fromCenter=False,  showCrosshair=True)
	# TopLeft = (area_pts[0],area_pts[1])
	# TopRight = ((area_pts[0]+ area_pts[2]),area_pts[1])
	# BotRight = ((area_pts[0]+ area_pts[2]), (area_pts[1]+ area_pts[3]))
	# BotLeft = (area_pts[0],(area_pts[1]+ area_pts[3]))
 
	#cap = cv2.VideoCapture('videos/vidrio51.mp4') 
	#object_detector = cv2.createBackgroundSubtractorMOG2()
	#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
	#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
	#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=False)
	object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
	#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
	#object_detector = cv2.createBackgroundSubtractorMOG2()
	#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
	def nothing(pos):
		pass

	#Mejor para lados verticales
	#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
	#fgbg = cv2.createBackgroundSubtractorMOG2() 

	#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
	tracker = EuclideanDistTracker() # type: ignore

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
	kernel = np.ones((1,1),np.uint8)
	#kernel = np.ones((2,2),np.uint8)
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

	while(True): 
		ret, frame = cap.read() 
		fr +=1
		cv2.putText(frame, "fr: " + str(fr), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

		#frame = imutils.resize (frame, width=720)
		#height, weight, _ = frame.shape
		#roi = frame[340:720, 500: 800]
		#roi = frame[0:500, 0: 600]
		#roi = frame[0:500, 100: 600]
		#frame = imutils.resize(frame, 200,200)
		#frame = imutils.resize(frame, 300,200)
		# min = cv2.getTrackbarPos('min','Thresholds')
		# max = cv2.getTrackbarPos('max','Thresholds')
		# area = cv2.getTrackbarPos('area','Thresholds')
		# lowc = cv2.getTrackbarPos('lowc','Thresholds')
		# maxc =  min = cv2.getTrackbarPos('min','Thresholds')
	 
	# Girar horizontalmente
		#frame = cv2.flip(frame, 1) 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# apply morphology open with square kernel to remove small white spots
		fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
		#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
				
		#area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])
		area_pts = np.array([[10, 10], [500,10], [500,500], [10,500]])
		#area_pts = np.array([[50,20], [500,20], [500,400], [50,400]])
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
		fgmask = cv2.dilate(fgmask, None, iterations=3)
		#erosion = cv2.erode(frame,kernel,iterations = 1)
	# threshold to binary
		thresh = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)[1]
		#edges = cv2.Canny(erosion, lowc, maxc)
	#Encontramos los contornos presentes en fgmask, para luego basándonos
		#en su área poder determina si existe movimiento
		#cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		#cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	
		#cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
		#contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# do connected components processing
		nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

		#get CC_STAT_AREA component as stats[label, COLUMN] 
		areas = stats[1:,cv2.CC_STAT_AREA]

		result = np.zeros((labels.shape), np.uint8)

		for i in range(0, nlabels - 1):
			#if areas[i] >= 100:   #keep
			if areas[i] >= 100:   #keep
				result[labels == i + 1] = 255
		#contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		#contours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# 		lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
# 		#cv2.putText(frame, "lines: " + str(len(lines)) , (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
# 		# print(len(lines))
# 		if lines is not None:
# 			cv2.putText(frame, "lines: " + str(len(lines)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
# 			for line in lines:
# 				x1,y1,x2,y2 = line[0]
# 				cv2.line(result,(x1,y1),(x2,y2),(255),1)
# 				cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
#    #(x, y, w, h) = cv2.boundingRect(convexHull)
# 			#(x, y, w, h) = cv2.boundingRect(cnt)
# 				cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 3)
				# rect = cv2.minAreaRect(cnt)
				# centerX = rect[0][0]
				# box = cv2.boxPoints(rect)
				# box = np.int_(box)
		contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
		#cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
		#f.write("Number of white pixels:"+ "\n")
  
		
	#convert to binary by thresholding
		#ret, binary_map = cv2.threshold(src,127,255,0)

		#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		#cv2.imshow("Binary", binary_map)
		#cv2.imshow("Result", result)
		for cnt in contours:
			#print(cv2.contourArea(cnt))
			#cv2.waitKey()

			if cv2.contourArea(cnt) > 100:
			#if cv2.contourArea(cnt) > 10000:
			#if w > 300:
				#convexHull = cv2.convexHull(cnt)
				#cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
	
				#(x, y, w, h) = convexHull
				#(x, y, w, h) = cv2.boundingRect(convexHull)
				(x, y, w, h) = cv2.boundingRect(cnt)
				# rect = cv2.minAreaRect(cnt)
				# centerX = rect[0][0]
				# box = cv2.boxPoints(rect)
				# box = np.int_(box)
				cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
				#cv2.drawContours(frame,[box],0,(0,255,0),2)
				detections.append([x, y, w, h])
				#center= box.
				#if 450 > w > 300:
				# calculate x,y coordinate of center
				m = cv2.moments(cnt)
				#m = cv2.moments(cnt)
				if m["m00"] != 0:
					cX = int(m["m10"] /  m["m00"]) 
					cY = int(m["m01"] / m["m00"])
					cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)    
						#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
					cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
						#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
						#cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)), (cX,cY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
						
				#print(len(detections))
				boxers_ids = tracker.update(detections)
				for box_id in boxers_ids:
					x, y, w, h, id = box_id
					#cv2.putText(frame,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
					#counter =id
					cv2.putText(frame,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
		#counter_actual=id
					#global counter
					#if  counter_actual>counter:
						#Guardar(counter_actual)
						#counter = counter_actual
						
					#cv2.putText(frame,  str(counter) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
					#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
					#print(detections)					
					#cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
					#number_of_white_pix = np.sum(fgmask == 255) 
					#cv2.putText(frame, "centroid", ((int(m[0]) - 25, int(m[1]) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2))
				
		cv2.imshow('result', result) 
		#cv2.imshow('edges', edges) 
		cv2.drawContours(frame, [area_pts], -1, (255,0,255), 2)
		#cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0,255,0), 2)
		#cv2.putText(frame, str(counter), (500,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		cv2.line(frame, (300,0), (300,500), (255, 0, 255), 1)
		#cv2.line(frame, (350,0), (350,500), (0, 255, 255), 1)
		#cv2.line(frame, (400,0), (400,500), (0, 255, 255), 1)
		cv2.imshow('thresh', thresh) 
		cv2.imshow('fgmask', fgmask) 
		cv2.imshow('frame',frame ) 

		k = cv2.waitKey(100) & 0xff
		if k == 27: 
			break
		
	cap.release() 
	cv2.destroyAllWindows()
 
DetectorMovimiento()
	#cv2.putText(frame, str(number_of_white_pix), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   
	
	# ret,binary_map = cv2.threshold(fgmask,min,max,cv2.THRESH_BINARY)
 
	# # do connected components processing
	# nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

	# #get CC_STAT_AREA component as stats[label, COLUMN] 
	# areas = stats[1:,cv2.CC_STAT_AREA]

	# result = np.zeros((labels.shape), np.uint8)

	# for i in range(0, nlabels - 1):
	# 	#if areas[i] >= 100:   #keep
	# 	if areas[i] >= area:   #keep
	# 		result[labels == i + 1] = 255
   
	
   
   #contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# calculate moments of binary image
	# calculate moments of binary image
	#M = cv2.moments(binary_map)
	# M = cv2.moments(fgmask)
	# if M["m00"] != 0:
	# 	# calculate x,y coordinate of center
	# 	cX = int(M["m10"] / M["m00"])
	# 	cY = int(M["m01"] / M["m00"])
	# 	cv2.putText(frame, "centroid", ((int(mc[i][0]) - 25, int(mc[i][1]) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2))
	# 	# calculate x,y coordinate of center
	 
		# display the image
		#cv2.imshow("Image", frame)
		#cv2.waitKey(0)
		
    # # Draw contours
	#contornos, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
   # Get the moments
	#mu = [None]*len(contours)
	#for i in range(len(contours)):
		#mu[i] = cv2.moments(contours[i])
 
		# put text and highlight the center
#		cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
  
    # Get the mass centers
	#mc = [None]*len(contours)
	#for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
		#mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
 
   
   
   
   
	
	
#     # apply morphology close with horizontal rectangle kernel to fill horizontal gap
# 	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
# 	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    #Areas big
# 	area_pts = np.array([[50,50], [715,50], [715,500], [50,500]])
	
# # show results
# 	# cv2.imshow("thresh", thresh)
# 	# cv2.imshow("morph1", morph1)
# 	# cv2.imshow("morph2", morph2)	
# 	cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# 	#a = cv2.getTrackbarPos('min','image')
# 	#b = cv2.getTrackbarPos('max','image')
# 	#f.write("Number of white pixels:"+ "\n")
 
	
	# apply LoG filter
	# LoG = cv2.Laplacian(img, cv2.CV_32F)
	# LoG = cv2.GaussianBlur(LoG, (5, 5), 0)

    # # apply DoG filter
	# DoG1 = cv2.GaussianBlur(img, (3, 3), 0) - cv2.GaussianBlur(img, (7, 7), 0)
	# DoG2 = cv2.GaussianBlur(img, (5, 5), 0) - cv2.GaussianBlur(img, (11, 11), 0)

    # # apply DoH filter
	# DoH = cv2.GaussianBlur(img, (5, 5), 0)
	# Dxx = cv2.Sobel(DoH, cv2.CV_64F, 2, 0)
	# Dyy = cv2.Sobel(DoH, cv2.CV_64F, 0, 2)
	# Dxy = cv2.Sobel(DoH, cv2.CV_64F, 1, 1)
	# DoH = (Dxx * Dyy) - (Dxy ** 2)

    # perform blob detection on the filtered images
	# params = cv2.SimpleBlobDetector_Params()
	# params.filterByArea = True
	# params.minArea = 10
	# params.filterByCircularity = False
	# params.filterByConvexity = False
	# params.filterByInertia = False

	# detector = cv2.SimpleBlobDetector_create(params)
	# keypoints_LoG = detector.detect(LoG)
	# keypoints_DoG1 = detector.detect(DoG1)
	# keypoints_DoG2 = detector.detect(DoG2)
	# keypoints_DoH = detector.detect(DoH)

    # # draw the detected blobs on the original image
	# img_with_keypoints_LoG = cv2.drawKeypoints(img, keypoints_LoG, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# img_with_keypoints_DoG1 = cv2.drawKeypoints(img, keypoints_DoG1, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# img_with_keypoints_DoG2 = cv2.drawKeypoints(img, keypoints_DoG2, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# img_with_keypoints_DoH = cv2.drawKeypoints(img, keypoints_DoH, np.array([]), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # # display the resulting images
	# cv2.imshow('LoG', img_with_keypoints_LoG)
	# cv2.imshow('DoG1', img_with_keypoints_DoG1)
	# cv2.imshow('DoG2', img_with_keypoints_DoG2)
	# cv2.imshow('DoH', img_with_keypoints_DoH)

 
 
 
 
