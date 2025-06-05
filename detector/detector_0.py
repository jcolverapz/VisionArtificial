import cv2
import numpy as np 
import math 
import imutils
from tracker import *

def nothing(pos):
	pass

cap = cv2.VideoCapture('videos/vidrio30.mp4') 
 
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.1, noiseSigma=0.001)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.01, noiseSigma=0.001)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.01, noiseSigma=0.01)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=5, backgroundRatio=0.7, noiseSigma=0.01)
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
#object_detector = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker() # type: ignore
 
 
while(True): 
	ret, frame = cap.read() 
	frame = imutils.resize (frame, width=720)

	# kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,21))  # 
	# temp1 = 255 - cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel_vertical)
	# #horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
	# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,1))
	# temp2 = 255 - cv2.morphologyEx(frame, cv2.MORPH_CLOSE, horizontal_kernel)
	# temp3 = cv2.add(temp1, temp2)
	# result = cv2.add(temp3, frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	area_pts = np.array([[200,5], [600,5], [600,400], [200,400]])
	 
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
 
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	
	fgmask = object_detector.apply(image_area)
# 	fgmask = fgbg.apply(frame) 
	
	#fgmask = cv2.dilate(fgmask, None, iterations=4)
	fgmask = cv2.dilate(fgmask, None, iterations=3)
	erosion = cv2.erode(fgmask,None,iterations = 2)

	thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY)[1] # threshold to binary
	 
	#ret, thresh_gray = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255, 255, cv2.THRESH_BINARY)
	contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# Erase small contours, and contours which small aspect ratio (close to a square)
	# for cnt in contours:
	# 	area = cv2.contourArea(cnt)

	# 	# Fill very small contours with zero (erase small contours).
		
				# cv2.drawContours(frame,approximatedShape,0,(0,255,0),1)
		# rect = cv2.minAreaRect(c)
		# (x, y), (w, h), angle = rect
		# aspect_ratio = max(w, h) / min(w, h)

		# # Assume zebra line must be long and narrow (long part must be at lease 1.5 times the narrow part).
		# if (aspect_ratio < 1.5):
		# 	#cv2.fillPoly(thresh_gray, pts=[c], color=0)
		# 	cv2.fillPoly(thresh_gray, pts=[c], color=(0,255,0))
		# 	continue

	thresh_gray = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))	# Use "close" morphological operation to close the gaps between contours
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_gray, None, None, None, 8, cv2.CV_32S)

	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)
	detections = []

	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= 10:   #keep
			result[labels == i + 1] = 255

	# Find contours in thresh_gray after closing the gaps
	contours, hier = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	cv2.putText(frame, "Contours: " + str(len(contours)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	for c in contours:
		area = cv2.contourArea(c)
		if area > 500:
			
			#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
			#lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=40,maxLineGap=80)
			lines = cv2.HoughLinesP(result,1,np.pi/180,100,minLineLength=40,maxLineGap=80)
			if lines is not None:
				for line in lines:
					x1,y1,x2,y2 = line[0]
					cv2.line(result,(x1,y1),(x2,y2),(255),1)
					cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)
	contours, hier = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		if cv2.contourArea(cnt) > 1500:
      
			#cv2.fillPoly(frame, pts=[cnt], color=(0,255,0))
			 
		
			print(cv2.contourArea(cnt))
			(x, y, w, h) = cv2.boundingRect(cnt)
			rect = cv2.minAreaRect(cnt)
			centerX = rect[0][0]
			box = cv2.boxPoints(rect)
			box = np.int_(box)
			#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
			cv2.drawContours(frame,[box],0,(0,222,0),2)
			cv2.fillPoly(frame, pts=[box],  color=(255, 0, 0))
			m = cv2.moments(cnt) # calculate x,y coordinate of center
			if m["m00"] != 0:
						cX = int(m["m10"] /  m["m00"]) 
						cY = int(m["m01"] / m["m00"])
						cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)
							#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
						cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			detections.append([x, y, w, h])
			boxers_ids = tracker.update(detections)
			for box_id in boxers_ids:
				x, y, w, h, id = box_id
				cv2.putText(frame,  str(id) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
				counter_actual=id	
	# 				(x, y, w, h) = cv2.boundingRect(cnt)
	# 				rect = cv2.minAreaRect(cnt)
	# 				centerX = rect[0][0]
	# 				box = cv2.boxPoints(rect)
	# 				box = np.int_(box)
     
     
     
	# 				#cv2.rectangle(frame, box, (x+w, y+h), (0,255,0), 3)
    #  # rect = cv2.minAreaRect(cnt)
	# 		# centerX = rect[0][0]
	# 		# box = cv2.boxPoints(rect)
	# 		# box = np.int_(box)
	# 		# #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	# 				cv2.drawContours(frame,box,0,(0,255,0),1)
	# 				#(x, y, w, h) = cv2.boundingRect(cnt)
					#rect = cv2.minAreaRect(cnt)
					#centerX = rect[0][0]
					#box = cv2.boxPoints(rect)
					#box = np.int_(box)
					#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	#cv2.waitKey()
				
	#cv2.imshow('result', result) 
	cv2.drawContours(frame, [area_pts], -1, (255,0,255), 2)

	cv2.imshow('thresh_gray', thresh_gray) 
	cv2.imshow('result', result) 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 


	k = cv2.waitKey(80) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.waitKey(0)
cv2.destroyAllWindows()
					#cv2.drawContours(frame, [cnt], 0, (0, 255, 255),2)
# borderThickness = 2
# 	borderColor = (0, 0, 0)
# 	grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
# 										cv2.BORDER_CONSTANT, None, borderColor)
# 	# Compute the skeleton:
# 	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
		# rect = cv2.minAreaRect(c)
		# box = cv2.boxPoints(rect)
		# # convert all coordinates floating point values to int
		# box = np.int_(box)
		# if cv2.contourArea(c) > 100:
		# 	cv2.drawContours(frame, [box], 0, (0, 255, 255),2)
		# else:
		# 	cv2.drawContours(frame, [box], 0, (0, 255, 0),1)
		
	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#cv2.imshow("Binary", binary_map)
	#cv2.imshow("Result", result)
	# 	#print(cv2.contourArea(cnt))
	# 	cv2.waitKey() 
			
	# 		#cv2.drawContours(frame,[box],0,(0,222,0),2)


#cv2.imshow('paper', image)