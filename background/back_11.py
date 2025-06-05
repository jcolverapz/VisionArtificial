import cv2
import numpy as np 
import cv2 
import imutils
#from skimage import medial_axis
#from skimage.morphology import medial_axis
#import pyodbc
import os
from datetime import datetime
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
Datos = 'objects'

# if not os.path.exists(Datos):
#     print('Carpeta creada: ',Datos)
#     os.makedirs(Datos)
counter=0
i=0

def nothing(pos):
    pass
 
cap = cv2.VideoCapture('videos/vidrio51.mp4') 

#cap = cv2.VideoCapture(0) 
object_detector = cv2.createBackgroundSubtractorMOG2()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG(history=10)
#object_detector = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=0, detectShadows=True)
#object_detector = cv2.bgsegm.createBackgroundSubtractorMOG() #createBackgroundSubtractorMOG()
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# Function to index and distance of the point closest to an array of points
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

#Mejor para lados verticales
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
#fgbg = cv2.createBackgroundSubtractorMOG2() 
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40) 
 
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#kernel = np.ones((1,1),np.uint8)
kernel = np.ones((2,2),np.uint8)# apply morphology open with square kernel to remove small white spots
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

	 
    #frame = cv2.flip(frame, 1) # Girar horizontalmente
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#gray = cv2.GaussianBlur(gray, (1,1), 0) #desenfoque

	edged = cv2.Canny(gray, lc, hc)
	edged = cv2.dilate(edged, None, iterations=101)
	edged = cv2.erode(edged, None, iterations=101)

	fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	
	area_pts_1 = np.array([[100,5], [650,5], [650,400], [100,400]])
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
    #cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
	#f.write("Number of white pixels:"+ "\n")

	#convert to binary by thresholding
	#ret, binary_map = cv2.threshold(src,127,255,0)
	# do connected components processing
	#nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask, None, None, None, 8, cv2.CV_32S)

	#get CC_STAT_AREA component as stats[label, COLUMN] 
	#areas = stats[1:,cv2.CC_STAT_AREA]
	#result = np.zeros((labels.shape), np.uint8)

	# for i in range(0, nlabels - 1):
	# 	#if areas[i] >= 100:   #keep
	# 	if areas[i] >= 500:   #keep
	# 		result[labels == i + 1] = 255
	output = frame.copy()
	#result = fgmask.copy()
	# Add borders to prevent skeleton artifacts:
	borderThickness = 2
	borderColor = (0, 0, 0)
	grayscaleImage = cv2.copyMakeBorder(fgmask, borderThickness, borderThickness, borderThickness, borderThickness,
									cv2.BORDER_CONSTANT, None, borderColor)
	# Compute the skeleton:
	skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)
	# # display skeleton
	# Create a black image
	img = np.zeros((512,512,3), np.uint8)
	i=0
	eps=0.2
 
	lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, 100, minLineLength=30, maxLineGap=30)
	if lines is not None:

		# Draw lines on the image
		for line in lines:
			x1, y1, x2, y2 = line[0]
			#cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
			cv2.line(skeleton, (x1, y1), (x2, y2), (255), 3)
			#cv2.waitKey()
		# encuentra el controno mas grande
		maxsize = 0  
		best = 0  
		count = 0

# convert to grayscale		
	contours = cv2.findContours(skeleton , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	cv2.putText(frame, "contours: " + str(len(contours)) , (100, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
 
	# for cnt in contours:
	# 	#cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
	# 	if cv2.contourArea(cnt)>1000:
	# 		(x, y, w, h) = cv2.boundingRect(cnt)
	# 		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
	# 	##m = cv2.moments(cnt) # calculate x,y coordinate of center
		#cv2.waitKey()
  # Convert bw image back to colored so that red, green and blue contour lines are visible, draw contours
	modified_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	cv2.drawContours(modified_image, contours, -1, (255, 0, 0), 3)
	#image = process_image4(screen)
   
	cv2.drawContours(frame, [area_pts_1], -1, (0,255,0), 2)
	cv2.imshow("skeleton",skeleton) 
	#cv2.imshow("skeletonr",result) 
			 
	cv2.imshow('frame',frame ) 
	cv2.imshow('fgmask', fgmask) 
	#cv2.imshow("result",result) 
 
	#cv2.waitKey(0)
 


	k = cv2.waitKey(100) & 0xff
	if k == 27: 
		break

cap.release() 
cv2.destroyAllWindows()

 
def process_image4(original_image):  # Douglas-peucker approximation
    # Convert to black and white threshold map
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert bw image back to colored so that red, green and blue contour lines are visible, draw contours
    modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(modified_image, contours, -1, (255, 0, 0), 3)

    # Contour approximation
    try:  # Just to be sure it doesn't crash while testing!
        for cnt in contours:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # cv2.drawContours(modified_image, [approx], -1, (0, 0, 255), 3)
    except:
        pass
    return modified_image
