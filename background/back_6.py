import cv2
import numpy as np 
import cv2 
import imutils
from funciones.tracker import *

def nothing(pos):
    pass

def tup(point): # tuplify
    return (point[0], point[1])


def overlap(source, target):# returns true if the two boxes overlap
    # unpack points
    tl1, br1 = source
    tl2, br2 = target

    # checks
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True

def getAllOverlaps(boxes, bounds, index):# returns all overlapping boxes
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps

#img = cv2.imread("images/gaps.png")
#orig = np.copy(img)
#blue, green, red = cv2.split(img)

def medianCanny(frame, thresh1, thresh2):
    median = np.median(img)
    img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
    return img

def closest_point(point, array):# Function to index and distance of the point closest to an array of points
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance
# blue_edges = medianCanny(blue, 0, 1)
# green_edges = medianCanny(green, 0, 1)
# red_edges = medianCanny(red, 0, 1)

#edges = blue_edges | green_edges | red_edges

# I'm using OpenCV 3.4. This returns (contours, hierarchy) in OpenCV 2 and 4
cap = cv2.VideoCapture('videos/vidrio20.mp4') 
#cap = cv2.VideoCapture(0) 
#object_detector = cv2.createBackgroundSubtractorMOG2()
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=True)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# borrowed shamelessly from: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points 

#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
tracker = EuclideanDistTracker() # type: ignore

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#kernel = np.ones((1,1),np.uint8)
kernel = np.ones((2,2),np.uint8)
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

	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# apply morphology open with square kernel to remove small white spots

	#gray = cv2.GaussianBlur(gray, (1,1), 0) #desenfoque

	edged = cv2.Canny(gray, lc, hc)
	edged = cv2.dilate(edged, None, iterations=4)
	#edged = cv2.erode(edged, None, iterations=2)
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	
	area_pts_1 = np.array([[100,20], [600,20], [600,350], [100,350]])
	#area_pts_1 = np.array([[100,5], [600,5], [600,410], [100,410]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	fgmask = cv2.dilate(fgmask, None, iterations=5)
 
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

	# #get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
		if areas[i] >= 300:   #keep
			result[labels == i + 1] = 255

	#cv2.imshow("result",result) 
	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#print(len(contours))

	output = frame.copy()
	# Add borders to prevent skeleton artifacts:
	borderThickness = 3
	borderColor = (0, 0, 0)
	grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
										cv2.BORDER_CONSTANT, None, borderColor)
	# Compute the skeleton:
	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
	
	cv2.imshow("skeleton",skeleton) # # display skeleton
  
	img = np.zeros((512,512,3), np.uint8) # Create a black image
	i=0
	momentos =[]
	#cv2.drawContours(contours)
	c=0
	
	#contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#contours = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#for i, cnt in enumerate(contours):
	eps=0.1
	#for  cnt in contours:
	contours,hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

	# go through the contours and save the box edges
	boxes = [] # each element is [[top-left], [bottom-right]]
	hierarchy = hierarchy[0]
	for component in zip(contours, hierarchy):
		#if cv2.contourArea(cnt) > 100:  
     
		currentContour = component[0]
		currentHierarchy = component[1]
		x,y,w,h = cv2.boundingRect(currentContour)
		if currentHierarchy[3] < 0:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
			boxes.append([[x,y], [x+w, y+h]])

	# filter out excessively large boxes
	filtered = []
	max_area = 19000
	for box in boxes:
		w = box[1][0] - box[0][0]
		h = box[1][1] - box[0][1]
		if w*h < max_area:
			filtered.append(box)
	boxes = filtered

	# go through the boxes and start merging
	merge_margin = 15

	# this is gonna take a long time
	finished = False
	highlight = [[0,0], [1,1]]
	points = [[[0,0]]]
	while not finished:
		# set end con
		finished = True

		print("Len Boxes: " + str(len(boxes))) # check progress

		# draw boxes # comment this section out to run faster
		#copy = np.copy(orig)
		copy = np.copy(frame)
		for box in boxes:
			cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 1)
		cv2.rectangle(copy, tup(highlight[0]), tup(highlight[1]), (0,0,255), 2)
		for point in points:
			point = point[0]
			cv2.circle(copy, tup(point), 4, (255,0,0), -1)
		cv2.imshow("Copy", copy)
		key = cv2.waitKey()
		if key == ord('q'):
			break

		# loop through boxes
		index = len(boxes) - 1
		while index >= 0:
			# grab current box
			curr = boxes[index]

			# add margin
			tl = curr[0][:]
			br = curr[1][:]
			tl[0] -= merge_margin
			tl[1] -= merge_margin
			br[0] += merge_margin
			br[1] += merge_margin

			# get matching boxes
			overlaps = getAllOverlaps(boxes, [tl, br], index)
			
			# check if empty
			if len(overlaps) > 0:
				# combine boxes
				# convert to a contour
				con = []
				overlaps.append(index)
				for ind in overlaps:
					tl, br = boxes[ind]
					con.append([tl])
					con.append([br])
				con = np.array(con)

				# get bounding rect
				x,y,w,h = cv2.boundingRect(con)

				# stop growing
				w -= 1
				h -= 1
				merged = [[x,y], [x+w, y+h]]

				# highlights
				highlight = merged[:]
				points = con

				# remove boxes from list
				overlaps.sort(reverse = True)
				for ind in overlaps:
					del boxes[ind]
				boxes.append(merged)

				# set flag
				finished = False
				break

			# increment
			index -= 1
#cv2.putText(frame,  str(id) , (100, [area_pts_1][0][0][1]+60),cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)
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

	#for index in range(len(contours)):
		#print(cv2.contourArea(cnt))
		#cv2.waitKey()
			#cnt=contours[index]
		#c=+1
		#(x, y, w, h) = cv2.boundingRect(cnt)
		#if cv2.contourArea(cnt) > 100:  
		# (x, y, w, h) = cv2.boundingRect(cnt)
		# perimeter = cv2.arcLength(cnt, False)
		# approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
		# #cv2.drawContours(frame, [approx], -1, (255, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#cv2.drawContours(frame, [approx], -1, (255, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)

		#cv2.putText(frame, "Contours: " + str(len(contours)) , (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		#cv2.putText(frame, "perimeter: #" + str(round(perimeter,2)) , (x - 10, y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		#cv2.putText(frame, "perimeter: " + str(round(perimeter,2)) , (x - 10, y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
# Ancho
		#cv2.putText(frame, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
	# No. Contorno
		#if 160 > w > 130:
			#rect = cv2.minAreaRect(cnt)
			#centerX = rect[0][0]
			#box = cv2.boxPoints(rect)
			#box = np.int_(box)
   
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
  
	
				#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
				#cv2.drawContours(frame,box,0,(0,255,0),1)
			#detections.append([x, y, w, h])
		#cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
	#if cv2.contourArea(cnt) > 10000:
	#if w > 300:
		#convexHull = cv2.convexHull(cnt)
		#cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
		#perimeter = cv2.arcLength(cnt, True)
		#(x, y, w, h) = convexHull
		#approximatedShape = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
		#v2.drawContours(output, [approximatedShape], -1, (0, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#cv2.drawContours(output, [convexHull], -1, (0, 255, 0), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#(x, y, w, h) = cv2.boundingRect(convexHull)
  
			#m = cv2.moments(cnt)  # calculate x,y coordinate of center
		#cv2.waitKey()
		#left_line = np.polyfit(left_line_y, left_line_x, 2)
		# # set up cross for tophat skeletonization
		# 	if m["m00"] != 0:
		# 		cX = int(m["m10"] /  m["m00"]) 
		# 		cY = int(m["m01"] / m["m00"])
		# 		cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)
		# 		momentos.append((cX, cY))
		# 			#cv2.putText(frame, "center: " + str((rect[0])) , (cX - 25, cY - 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		# 		cv2.putText(frame, "centroid: " + str((cX, cY)) , (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			#cv2.line(frame, ((cX, cY)), (endpt_x, endpt_y), 255, 2)  
				#cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (cX,cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
				#cv2.putText(frame, "area: " + str(cv2.contourArea(cnt)), (cX,cY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
		#intersections = list()
			
		# boxers_ids = tracker.update(detections) #Tracker
		# for box_id in boxers_ids:
		# 	x, y, w, h, id = box_id
			#counter =id
			#cv2.putText(frame,  str(id) , (cX, cY - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
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
			#for i in momentos:
		#print(len(momentos))
		
		# if len(momentos) >1:
		# 	for i in range(len(momentos)):
		# 		if i==1:
		# 			#miline
		# 			miline = cv2.line(frame, (momentos[i][0], momentos[i][1]), (momentos[i-1][0], momentos[i-1][1]), (0,0,255), 3)
		# 			#detections.append([(momentos[i][0], (momentos[i][1], w, h])
		# 			#(x, y, w, h) = cv2.minAreaRect(miline)
		# 			#rect = 
		
		# 			deltaX = abs(momentos[i][0] - momentos[i-1][0])**2
		# 			deltaY = abs(momentos[i][1] - momentos[i-1][1])**2
		# #length = norm(p2 - p1) where p1(x1,y1) and p2(x2,y2)
	
		#var deltaY = pt2.Y - pt1.Y;
				#print((i),(momentos[i][0], momentos[i][1]), (momentos[i-1][0], momentos[i-1][1]))	
					# distancia= math.sqrt(deltaX + deltaY)
					
					# if 260 > distancia > 200:
					# 	detections.append([momentos[i][0], momentos[i][1], 3, distancia])
			
			# m = cv2.moments(cnt)  # calculate x,y coordinate of center
			# #cv2.waitKey()
			# #left_line = np.polyfit(left_line_y, left_line_x, 2)
			# # set up cross for tophat skeletonization
			# if m["m00"] != 0:
			# 	cX = int(m["m10"] /  m["m00"]) 
			# 	cY = int(m["m01"] / m["m00"])
			# cv2.circle(frame, (int(cX), int(cY)), 3, (0, 0, 255), -1)
			# momentos.append((cX, cY))
					# centroX = deltaX/2
					# centroY = int(distancia/2)

					# cv2.putText(frame,  "distancia: " + str(int(distancia)) , (350, centroY),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
					# #print("distancia =", str(distancia))
					# #cv2.waitKey()
					# if j==0:
					# 	#cv2.putText(frame,  "Unix Timestamp:" + str({time.time()}) , (330, 200),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
					# 	now = datetime.now()
					# 	formatted = now.strftime("%Y-%m-%d_%H_%M_%S")
					# 	cv2.putText(frame,  "Timestamp: " + str(formatted) , (200, [area_pts_1][0][3][1]+30),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255,255), 2)
		
					# 	objeto = frame[10:400,200:800]
					# 	objeto = imutils.resize(objeto,width=350)
	
				#cv2.imwrite(Datos+'/objeto_{}.jpg'.format(counter),objeto) 
#Unix Timestamp: {time.time()}'
				#print("Imagen impresa: "+ str(i))
				#if 290 > distancia > 260:
	

		# boxers_ids = tracker.update(detections) #Tracker
		# for box_id in boxers_ids:
		# 	x, y, w, h, id = box_id
		# 	cv2.putText(frame,  str(id) , (x, y - 15),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
		# 	# cv2.imwrite(Datos+'/objeto_{}.jpg'.format(formatted),objeto) 
			#counter +=1
		# j+=1


# else:
	# 	j=0
	#cv2.putText(frame,  , (330, 200),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
	#cv2.putText(frame,  "Timestamp:  " + str(formatted) , (200, [area_pts_1][0][3][1]+30),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
	#areas = stats[1:,cv2.CC_STAT_AREA]
	#print()
	

	# k = cv2.waitKey(100) & 0xff
	# if k == 27: 
	# 	break

# cap.release() 
# cv2.destroyAllWindows()

 