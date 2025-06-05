import cv2
import numpy as np 
import cv2 
import imutils
# from funciones.tracker import *
#from skimage.morphology import medial_axis
import os
from datetime import datetime
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
# Datos = 'objects'
# if not os.path.exists(Datos):
#     print('Carpeta creada: ',Datos)
#     os.makedirs(Datos)
counter=0
i=0
j=0

def nothing(pos):
    pass
 
#cap = cv2.VideoCapture(0) 
#cap.set(cv2.CAP_PROP_FPS, 1)
cap = cv2.VideoCapture('videos/vidrio51.mp4') 
#cap = cv2.VideoCapture(0) 
#object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=True)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#object_detector = cv2.bgsegm.createBackgroundSubtractorGMG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() 
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#kernel = np.ones((1,1),np.uint8)
kernel = np.ones((2,2),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1000))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

#Mejor para lados verticales

# tracker = EuclideanDistTracker() # type: ignore
cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

while(True): 
	ret, frame = cap.read() 
	frame = imutils.resize (frame, width=720)
 
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')

	# Girar horizontalmente
    #frame = cv2.flip(frame, 1) 
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# apply morphology open with square kernel to remove small white spots
	
 #desenfoque
	#gray = cv2.GaussianBlur(gray, (1,1), 0)

	edged = cv2.Canny(gray, lc, hc)
	#edged = cv2.dilate(edged, None, iterations=4)
	#edged = cv2.erode(edged, None, iterations=2)

	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	
	#area_pts_1 = np.array([[450,20], [600,20], [600,350], [450,350]])
	area_pts_1 = np.array([[100,5], [600,5], [600,410], [100,410]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	fgmask = cv2.dilate(fgmask, None, iterations=5)
 
     
	detections = []
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)
	result = np.zeros((labels.shape), np.uint8)
	 
	output = frame.copy()
	# Add borders to prevent skeleton artifacts:
	borderThickness = 1
	borderColor = (0, 0, 0)
	grayscaleImage = cv2.copyMakeBorder(result, borderThickness, borderThickness, borderThickness, borderThickness,
										cv2.BORDER_CONSTANT, None, borderColor)
	# Compute the skeleton:
	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
	#contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# # display skeleton
	#cv2.imshow("skeleton",skeleton) 
  # Create a black image
	i=0
	momentos =[]
	#cv2.drawContours(contours)
	c=0
	 
	contours = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	 
	#for i, cnt in enumerate(contours):
	eps=0.1
	for  cnt in contours:
	#for index in range(len(contours)):
		
		#cv2.waitKey()
		#cnt=contours[index]
		c=+1
		(x, y, w, h) = cv2.boundingRect(cnt)
		perimeter = cv2.arcLength(cnt, False)
		approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
		cv2.drawContours(frame, [approx], -1, (255, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
  
		cv2.putText(frame, "Contours: " + str(len(contours)) , (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		#cv2.putText(frame, "perimeter: " + str(round(perimeter,2)) , (x - 10, y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		#cv2.putText(frame, "perimeter: " + str(round(perimeter,2)) , (x - 10, y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
# Ancho
		cv2.putText(frame, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
# No. Contorno
		#cv2.putText(frame, "Index.:" + str(index), (x,y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
		#cv2.putText(frame, "Index:" + str(index), (100 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
  
		# extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
		# extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
		# #extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
		# #extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
		# cv2.circle(frame, extLeft, 4, (255, 0, 0), -1)
		# cv2.circle(frame, extRight, 4, (0, 0, 255), -1)
		# #cv2.circle(frame, extTop, 4, (0, 255, 255), -1)
		# #cv2.circle(frame, extBot, 4, (0, 255, 0), -1)
  
		#if 160 > w > 100:
      
		#cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
	#if cv2.contourArea(cnt) > 10000:
	#if w > 300:
		convexHull = cv2.convexHull(cnt)
		#cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
		#perimeter = cv2.arcLength(cnt, True)
		#(x, y, w, h) = convexHull
		#approximatedShape = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
		#v2.drawContours(output, [approximatedShape], -1, (0, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#cv2.drawContours(output, [convexHull], -1, (0, 255, 0), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#(x, y, w, h) = cv2.boundingRect(convexHull)
		(x, y, w, h) = cv2.boundingRect(cnt)
		rect = cv2.minAreaRect(cnt)
		centerX = rect[0][0]
		box = cv2.boxPoints(rect)
		box = np.int_(box)
			#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
			#cv2.drawContours(frame,box,0,(0,255,0),1)
			#detections.append([x, y, w, h])

			# line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
			# # line_color = [0, 0, 255]
			# # line_thickness = 2
			# # dot_color = [0, 255, 0]
			# # dot_size = 3
 
			# # 		y2 = int(y0 - 3000*(a))
			# # 		cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
			 
			# cv2.imshow('houghlines',imutils.resize(img, height=650))
			# for x1, y1, x2, y2 in line:
			# 	cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
			# 	cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
		
			# 	overlay = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
			# cv2.imshow("Overlay", overlay)
			 
		# calculate x,y coordinate of center
		m = cv2.moments(cnt)
		#cv2.waitKey()
		#left_line = np.polyfit(left_line_y, left_line_x, 2)
		# set up cross for tophat skeletonization
		if m["m00"] != 0:
			cX = int(m["m10"] /  m["m00"]) 
			cY = int(m["m01"] / m["m00"])
			cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)
			momentos.append((cX, cY))
				#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			#cv2.line(frame, ((cX, cY)), (endpt_x, endpt_y), 255, 2)  
				#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
				#cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)), (cX,cY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		#intersections = list()
			
		#boxers_ids = tracker.update(detections) #Tracker
		# for box_id in boxers_ids:
		# 	x, y, w, h, id = box_id
		# 	cv2.putText(frame,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
		# 	#counter =id
		# 	cv2.putText(frame,  str(id) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
		# 	#counter_actual=id
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
   
	
   
	#for i in momentos:
	print(len(momentos))
	
	if len(momentos) >1:
		for i in range(len(momentos)):
			if i==1:
				miline = cv2.line(frame, (momentos[i][0], momentos[i][1]), (momentos[i-1][0], momentos[i-1][1]), (0,0,255), 3)
				#detections.append([(momentos[i][0], (momentos[i][1], w, h])
				#(x, y, w, h) = miline
    
				deltaX = abs(momentos[i][0] - momentos[i-1][0])**2
				deltaY = abs(momentos[i][1] - momentos[i-1][1])**2
    #length = norm(p2 - p1) where p1(x1,y1) and p2(x2,y2)
    
        #var deltaY = pt2.Y - pt1.Y;
				#print((i),(momentos[i][0], momentos[i][1]), (momentos[i-1][0], momentos[i-1][1]))	
				#distancia= math.sqrt(deltaX + deltaY)
				#cv2.putText(frame,  "distancia: " + str(distancia) , (momentos[0][0], momentos[0][1]),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
				#print("distancia =", str(distancia))
				#cv2.waitKey()
				if j==0:
					#cv2.putText(frame,  "Unix Timestamp:" + str({time.time()}) , (330, 200),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
					now = datetime.now()
					formatted = now.strftime("%Y-%m-%d_%H_%M_%S")
					cv2.putText(frame,  "Timestamp: " + str(formatted) , (200, [area_pts_1][0][3][1]+30),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255,255), 2)
     
					objeto = frame[10:400,200:800]
					objeto = imutils.resize(objeto,width=350)
     
					#cv2.imwrite(Datos+'/objeto_{}.jpg'.format(counter),objeto) 
     #Unix Timestamp: {time.time()}'
					# #print("Imagen impresa: "+ str(i))
					# if 290 > distancia > 260:
         
					# 	cv2.imwrite(Datos+'/objeto_{}.jpg'.format(formatted),objeto) 
						# counter +=1
						# j+=1
	# else:
	# 	j=0
	#cv2.putText(frame,  , (330, 200),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
	cv2.putText(frame,  "Timestamp:  " + str(formatted) , (200, [area_pts_1][0][3][1]+30),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
	#areas = stats[1:,cv2.CC_STAT_AREA]
	#print()
	cv2.putText(frame,  str(counter) , (100, [area_pts_1][0][0][1]+60),cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)
				#cv2.line(frame, (300, 10), (400, 300), (0,0,255), 2)	
				#print(momentos[i][0])
    
  #lines[i][0]
       
			
		# Draw a diagonal blue line with thickness of 5 px
			#cv2.line(img,(centerX,centerY),(511,511),(255,0,0),5)    
 
	#cv2.imshow('result', result) 
	#cv2.drawContours(frame, [area_pts_1], -1, (255,0,255), 2)
	cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 


	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()


#Grabar captura de pantalla
    
     