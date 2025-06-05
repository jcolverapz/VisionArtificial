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
cap = cv2.VideoCapture('videos/vidrio20.mp4')
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

while(True): 
	ret, frame = cap.read() 
  
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	edged = cv2.Canny(gray, 200, 255)
	edged = cv2.dilate(edged, None, iterations=40)
	
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	
	area_pts_1 = np.array([[150,5], [650,5], [650,450], [150,450]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)

    #
	cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
    #cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
	#f.write("Number of white pixels:"+ "\n")
	detections = []

	#convert to binary by thresholding
	 
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= 100:   #keep
			result[labels == i + 1] = 255

	cv2.imshow("result",result) 
	
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
			
		cv2.imshow('fgmask', fgmask) 
		cv2.imshow('frame',frame ) 
		
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
		rect = cv2.minAreaRect(cnt)
		centerX = int(rect[0][0])
		centerY = int(rect[0][1])

		box = cv2.boxPoints(rect)
		box = np.int_(box)

		cv2.circle(frame,(centerX, centerY),5,(0,0,255),-1)

	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()


     