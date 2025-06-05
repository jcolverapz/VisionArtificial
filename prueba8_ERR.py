import cv2
import numpy as np
import imutils

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/vidrio0.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG2() 
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

def nothing(pos):
	pass
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel = cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5))
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(20,100))
#color = (0, 255, 0)
cv2.namedWindow('Thresholds')
cv2.createTrackbar('LS','Thresholds',0,255, nothing)
cv2.createTrackbar('LH','Thresholds',0,255, nothing)

i=0
#
# ls=0
#lh=255

while True:
	_, frame = cap.read()
	#kernel = np.ones((3,15), np.unit8)
	#lines= cv2.erode(img, kernel, iterations=1)
		#ret, frame = cap.read()
	#if ret == False: break
	
	#gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	#thresh = cv2.threshold(gray, args["threshold"], 255,
	#cv2.THRESH_BINARY)[1]
 
    # clone our original image (so we can draw on it) and then draw
    # a bounding box surrounding the connected component along with
    # a circle corresponding to the centroid
	#output = image.copy()
	# cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	# cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)    

    # construct a mask for the current connected component by
    # finding a pixels in the labels array that have the current
    # connected component ID
	#componentMask = (labels == i).astype("uint8") * 255
    # show our output image and connected component mask
	#cv2.imshow("Output", output)
	#cv2.imshow("Connected Component", componentMask)
 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#fgmask = fgbg.apply(frame)
	fgmask = fgbg.apply(gray)
	fgmask = cv2.dilate(fgmask, None, iterations=1)
    
	#gray = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
	#gray = cv2.cvtColor(img, cv2.IMREAD_UNCHANGED)
	#gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #COLOR_BGR2GRAY
	#edges = cv2.Canny(gray, 100, 200)
 
	#binary = cv2.threshold(gray, ls, lh, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	#binary = cv2.threshold(gray, ls, lh, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#ret,thresh = cv2.threshold(img, umbral, valorMax , tipo)	
	#binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	#img = cv2.imread('matizGris.png',0)
	#ret,thresh = cv2.threshold(img,ls,lh,cv2.THRESH_BINARY_INV)
	#ret,thresh = cv2.threshold(img,ls,lh,cv2.THRESH_TOZERO)
	#_, thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
	#ret, thresh = cv2.threshold(gray,0, lth, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#print('Umbral de th1:', ret)
 
 # Apply global (simple) thresholding on image
	#ret1,thresh1 = cv2.threshold(lines,127,255,cv2.THRESH_BINARY)

# Apply Otsu's thresholding on image
	#ret2,thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply Otsu's thresholding after Gaussian filtering
	#blur = cv2.GaussianBlur(img,(5,5),0)
	#ret3,thresh3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 # Setting parameter values 
	ls=cv2.getTrackbarPos('LS','Thresholds')
	lh=cv2.getTrackbarPos('LH','Thresholds')
  
	ret,thresh = cv2.threshold(fgmask,ls,lh,cv2.THRESH_BINARY)
    #cv2.bitwise_not(threshold, threshold)
# Applying the Canny Edge filter 
	#edges = cv2.Canny(frame, ls, lh) 
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
 
	
 
	#cv2.imshow("edges", edges)
	cv2.imshow("Imagen", thresh)
	contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	listx = []
	listy = []
    #cv2.drawContours(frame,countours,-1,(0,255,0),4)
    
	#for cnt in countours:
	for cntr in range(0, len(contours)):
		cntr = contours[i]
		size = cv2.contourArea(cntr)
		if size < 1000:
			M = cv2.moments(cntr)
			cX = int(M["m10"] / (M["m00"] + 1e-5))
			cY = int(M["m01"] / (M["m00"] + 1e-5))
			listx.append(cX)
			listy.append(cY)

		listxy = list(zip(listx,listy))
		listxy = np.array(listxy)

		for x1, y1 in listxy:    
			distance = 0
			secondx = []
			secondy = []
			dist_listappend = []
			sort = []   
			for x2, y2 in listxy:      
				if (x1, y1) == (x2, y2):
					pass     
				else:
					distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
					secondx.append(x2)
					secondy.append(y2)
					dist_listappend.append(distance)               
			secondxy = list(zip(dist_listappend,secondx,secondy))
			sort = sorted(secondxy, key=lambda second: second[0])
			sort = np.array(sort)
			cv2.line(thresh, (x1,y1), (int(sort[0,1]), int(sort[0,2])), (0,0,255), 2)

	cv2.imshow('img', thresh)
	#cv2.imwrite('connected.png', frame)
# # # 		#print(cv2.contourArea(cnt))
# 		if cv2.contourArea(cnt) > 10000:
# # # 		if cv2.contourArea(cnt) > 1000:
# 			(x, y, w, h) = cv2.boundingRect(cnt)
# # 			(x, y, w, h) = cv2.boundingRect(cnt)
# # # 			#band1 = True
# 			cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,255), 3)
# # # 			number_of_white_pix = np.sum(fgmask == 255)
 

	cv2.imshow("Contour",fgmask)
	cv2.imshow("Contour",frame)
 
	 
	k = cv2.waitKey(80) & 0xFF
	if k == 27:
		break
 
cap.release()
cv2.destroyAllWindows()