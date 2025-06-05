import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('videos/vidrio51.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history = 100,varThreshold = 16, detectShadows=False)
def nothing(pos):
	pass

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel = cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5))
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))

cv2.namedWindow('Thresholds')
cv2.createTrackbar('LS','Thresholds',127,255, nothing)
cv2.createTrackbar('LH','Thresholds',255,255, nothing)
cv2.createTrackbar('LA','Thresholds',0,1000, nothing)
 
while True:
	ret, frame = cap.read()
	if ret == False: break

	ls=cv2.getTrackbarPos('LS','Thresholds')
	lh=cv2.getTrackbarPos('LH','Thresholds')
	la=cv2.getTrackbarPos('LA','Thresholds')

	#src = cv2.imread('images/img.png', cv2.IMREAD_GRAYSCALE)
  
	fgmask = object_detector.apply(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,20))
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fgmask = cv2.dilate(fgmask, None, iterations=1)
 
	edges = cv2.Canny(gray, ls, lh)
	ret,mask =cv2.threshold(edges,100,255,cv2.THRESH_BINARY)
	image2 = cv2.medianBlur(fgmask, 3)  # this	
	cv2.imshow("Mask", mask)

 
 	# #Areas big
	area_pts = np.array([[50,10], [715,10], [715,500], [50,500]])
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
 
	image_area = cv2.bitwise_and(gray, gray, mask=imAux)
 	
	fgmask = object_detector.apply(image_area)
	
 
# convert to binary by thresholding
	#ret, binary_map = cv2.threshold(fgmask,127,255,0)
	ret,binary_map = cv2.threshold(fgmask,ls,lh,cv2.THRESH_BINARY)
	#ret, binary_map = cv2.threshold(fgmask,ls,lh,0)
  
# Applying the Canny Edge filter 
	#edges = cv2.Canny(frame,250,200)
 
# do connected components processing
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
		#if areas[i] >= 100:   #keep
		if areas[i] >= la:   #keep
			result[labels == i + 1] = 255
   
   #Contornos
	#cnts = cv2.findContours(result, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#c = max(cnts, key=cv2.contourArea)
	#contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	LENGTH = len(contours)

	#contours = sorted(result, key=lambda x: cv2.contourArea(x), reverse=True)	
	#cv2.putText(frame, "contours: " + str (LENGTH),(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	for cnt in contours:
	#for i in range(0, nlabels - 1):
     
	#if cv2.contourArea(cnt) > 100:  #convexHull = cv2.convexHull(contour)
		cv2.drawContours(frame, cnt,-1,(0,255,0),4)
	#cv2.drawContours(frame,cnt,-1,(0,255,0),4)
 
	cv2.imshow("frame", frame)
	cv2.imshow("Binary", binary_map)
	cv2.imshow("Result", result)
 
	#drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
   
	#for i in range(len(contours)):
		#cv2.drawContours(drawing, contours, i, (0,255,0), 3)
		#cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
		#cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		#cv2.putText(frame, "centroid", ((int(mc[i][0]) - 25, int(mc[i][1]) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2))
    	#cv2.imshow("drawing", drawing)
 	#cv2.waitKey()
	#cv2.imwrite('Filterd_result.png, result') 
	 
	k = cv2.waitKey(80) & 0xFF
	if k == 27:
		break
 
cap.release()
cv2.destroyAllWindows()
	#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(100,5,0.7,0)
#fgbg = cv2.createBackgroundSubtractorMOG2() 
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

#kernel = np.ones((vertical,horizontal), np.uint8)  # note this is a horizontal kernel#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(20,100))
#kernel = np.ones((1,10), np.uint8)  # note this is a horizontal kernel#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(20,100))
#kernel = np.ones((1,15), np.unit8)
	#lines= cv2.erode(img, kernel, iterations=1)
	 
	
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
 
	# cv2.imshow("thresh",thresh)
	# cv2.imshow("fgmask",fgmask)
	# cv2.imshow("edges",edges)
	# cv2.imshow("frame",frame)
	# #fgmask = fgbg.apply(frame)
    
	#gray = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
	#gray = cv2.cvtColor(img, cv2.IMREAD_UNCHANGED)
	#gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #COLOR_BGR2GRAY
 
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
	 
# 	countours,hierarchy=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 	#hull
# 	for cnt in countours:
# 		#hull = cv2.convexHull(countours[0])
# 		hull = cv2.convexHull(cnt)
     
# # # # 		#print(cv2.contourArea(cnt))
# # 		if cv2.contourArea(cnt) > 10000: 	
# 		#cv2.drawContours(fgmask, [hull], -1, (255,255,255), 5)
# 		cv2.drawContours(fgmask, [hull], -1, (0,255,0), 5)
#Test

	# for i in range(len(countours)):
	# 	min_dist = max(fgmask.shape[0], fgmask.shape[1])
	# 	cl = []
		
	# 	ci = countours[i]
	# 	ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
	# 	ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
	# 	ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
	# 	ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
	# 	ci_list = [ci_bottom, ci_left, ci_right, ci_top]
		
	# 	for j in range(i + 1, len(countours)):
	# 		cj = countours[j]
	# 		cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
	# 		cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
	# 		cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
	# 		cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
	# 		cj_list = [cj_bottom, cj_left, cj_right, cj_top]
			
	# 		for pt1 in ci_list:
	# 			for pt2 in cj_list:
	# 				dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))     #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
	# 				if dist < min_dist:
	# 					min_dist = dist             
	# 					cl = []
	# 					cl.append([pt1, pt2, min_dist])
	# 	if len(cl) > 0:
	# 		cv2.line(fgmask, cl[0][0], cl[0][1], (255, 255, 255), thickness = 1)
	#lines = cv2.HoughLinesP(edge_image, 1, np.pi/180, 60, minLineLength=10, maxLineGap=250)
	#lines = cv2.HoughLinesP(edge_image, 1, np.pi/180, 60, minLineLength=10, maxLineGap=250)
	#Going through every line we found and drawing it in an image based on starting and ending point
	#for i in range(2):
	#print(lines[i])
	#[[  2   1 589   1]]
	#[[336   4 542 297]]
 
	#cv2.imshow("edges", edges)
	#cv2.imshow("Imagen", thresh)
 
#	cv2.drawContours(frame,countours,-1,(0,255,0),4)
# 	for cnt in countours:
# # # 		#print(cv2.contourArea(cnt))
# 		if cv2.contourArea(cnt) > 10000:
# # # 		if cv2.contourArea(cnt) > 1000:
# 			(x, y, w, h) = cv2.boundingRect(cnt)
# # 			(x, y, w, h) = cv2.boundingRect(cnt)
# # # 			#band1 = True
# 			cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,255), 3)
# # # 			number_of_white_pix = np.sum(fgmask == 255)
 

	#cv2.imshow("edge_image",edges)
# 	cv2.imshow("THRESH_OTSU1", thresh1)
	#cv2.imshow("THRESH_OTSU2", output)
	#cv2.imshow("THRESH_OTSU2", thresh2)
	#cv2.imshow("THRESH_OTSU3", thresh3)
 
	# Adaptative threshold
	#th1  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	#th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2 )
	#threshold = cv2.adaptiveThreshold(img,maxValue,adaptiveMethod,thresholdType,blockSize,C)
	# th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,10)
	# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,10)
 
	# cv2.imshow("ADAPTIVE_THRESH_MEAN_C", th1)
	# cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th2)
 
    # getting mask with connectComponents
	# ret, labels = cv2.connectedComponents(binary)
	# #ret, labels = cv2.connectedComponents(gray)
	# for label in range(1,ret):
	# 	mask = np.array(labels, dtype=np.uint8)
	# 	mask[labels == label] = 255
	# 	cv2.imshow('component',mask)
	#cv2.waitKey(0)
    
	#cv2.imshow("Imagen", img)
	#cv2.imshow("THRESH", thresh)
     
 	# #Areas big
	# #area_pts = np.array([[50,10], [715,10], [715,500], [50,500]])
	 
 	# # Con ayuda de una imagen auxiliar, determinamos el área
	# # sobre la cual actuará el detector de movimiento
	# #imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	 
	# #imAux = cv2.drawContours(imAux, [area_pts], -1,(255), -1)
  
	# fgmask = fgbg.apply(frame)
  
	  