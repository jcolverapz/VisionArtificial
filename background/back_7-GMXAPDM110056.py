import cv2
import numpy as np 
import cv2 
import imutils
#from tracker import *
#from skimage.morphology import medial_axis

def nothing(pos):
    pass

#cap = cv2.VideoCapture(0)
#object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=True)
cap = cv2.VideoCapture('videos/vidrio51.mp4')
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
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
#kernel = np.ones((5,5),np.uint8)9
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,101))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (101,3))
#kernel = np.ones((6,6),np.uint8)
    # morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  
#kernel = np.ones((1,1),np.uint8)
#kernel = np.ones((2,2),np.uint8)
#kernel = np.ones((3,3),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1000))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

while(True): 
	ret, frame = cap.read() 
	frame = imutils.resize (frame, width=720)
	num = cap.get(1)
	cv2.putText(frame, "frame: " + str((num)) , (5,20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')

  
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# apply morphology open with square kernel to remove small white spots
	#ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
 #desenfoque
	#gray = cv2.GaussianBlur(gray, (1,1), 0)

	edged = cv2.Canny(gray, lc, hc)
	#edged = cv2.dilate(edged, None, iterations=20)
	edged = cv2.dilate(edged, None, iterations=40)
	#edged = cv2.erode(edged, None, iterations=2)

	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	
	area_pts_1 = np.array([[150,5], [650,5], [650,450], [150,450]])
	#area_pts_1 = np.array([[100,5], [600,5], [600,410], [100,410]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	#fgmask = cv2.dilate(fgmask, None, iterations=5)
 
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
	cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
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
		if areas[i] >= 100:   #keep
			result[labels == i + 1] = 255

	cv2.imshow("result",result) 
 

	output = frame.copy()
	# Add borders to prevent skeleton artifacts:
	borderThickness = 4
	borderColor = (0, 0, 0)
	grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
										cv2.BORDER_CONSTANT, None, borderColor)
	#Compute the skeleton:
	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
	 
	eps=0.1
	#contours = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
	#contours = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.  CHAIN_APPROX_SIMPLE)[0]
	#contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	output2 = frame.copy()
	#for  cnt in contours:
	cv2.putText(frame, "Contours: " + str(len(contours)) , (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
	for index in range(len(contours)):
		#print(cv2.contourArea(cnt))
		#cv2.waitKey()
		cnt=contours[index]
		#c=+1
		(x, y, w, h) = cv2.boundingRect(cnt)
		#if cv2.contourArea(cnt) > 3000: 
		cv2.imshow("skeleton",skeleton) 
			
		cv2.imshow('fgmask', fgmask) 
		cv2.imshow('frame',frame ) 
		if w > 300: 
      
			print (str(cv2.contourArea(cnt)))
		#perimeter = cv2.arcLength(cnt, False)
		#approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
		#cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#cv2.drawContours(frame, cnt, -1, (255, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#(x, y, w, h) = cv2.boundingRect(cnt)
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
			rect = cv2.minAreaRect(cnt)
			centerX = int(rect[0][0])
			centerY = int(rect[0][1])
	
			box = cv2.boxPoints(rect)
			box = np.int_(box)
  
			cv2.circle(frame,(centerX, centerY),5,(0,0,255),-1)

			cv2.putText(frame,  str(w) + "," + str(h) , (centerX, centerY),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
		#if rect[2] > 70:
		#cv2.drawContours(frame,[box],0,(0,255,255),2)
			cv2.imshow('frame',frame ) 
			cv2.waitKey()
  
  
		# rect = cv2.minAreaRect(cnt)
		# centerX = rect[0][0]
		# box = cv2.boxPoints(rect)
		# box = np.int_(box)
		# perimeter = cv2.arcLength(cnt, False)
		# approx = cv2.approxPolyDP(cnt, eps * perimeter, True)
		# #if len(approx) == 4:
        #        # doc_cnts = approx
		# #cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		# cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		# convexHull = cv2.convexHull(cnt)   
		#(x, y, w, h) = convexHull
		#(x, y, w, h) = cv2.boundingRect(convexHull)
#Encontrar el contorno mas grande
#for i in range(len(contours)):



	minLineLength = 10
	maxLineGap = 10
	lines = cv2.HoughLinesP(skeleton,3,np.pi/180,100, minLineLength, maxLineGap)
	if lines is not None:
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(fgmask,(x1,y1),(x2,y2),(255,255,255),4)

	#cv2.imshow('hough',hough)
		#cv2.drawContours(frame, cnt, -1, (255, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
		#cv2.drawContours(frame, cnt, -1, (255, 0, 255), 3)#(centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(cnt)
  


	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()


#Grabar captura de pantalla
    
     