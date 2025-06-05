# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2 
import imutils

#cap = cv2.VideoCapture('/home/sourabh/Downloads/people-walking.mp4') 
cap = cv2.VideoCapture('videos/vidrio51.mp4') 
#cap = cv2.VideoCapture('videos/vidrio0.mp4') 
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG2() 
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2() 

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#kernel = np.ones((5,5),np.uint8)

while(1): 
	#img, frame = cap.read() 
	_, frame = cap.read() 
	#frame = imutils.resize (frame, width=720)
	#img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Girar horizontalmente
    #frame = cv2.flip(frame, 1) 
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
 #cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
	 
	#area_pts2 = np.array([[500,150], [550,150], [550,200], [500,200]])
	# imAux2 = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	# imAux2 = cv2.drawContours(imAux2, [area_pts2], -1, (255), -1)
 
	# image_area2 = cv2.bitwise_and(gray, gray, mask=imAux2)
 	
	# fgmask2 = fgbg.apply(image_area2)
	
    # # threshold to binary
	# thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)[1]

    # # apply morphology open with square kernel to remove small white spots
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19,19))
	# morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# 	fgmask = fgbg.apply(frame) 
# 	#kernel = np.ones((5,5),np.uint8)
# 	#kernel = np.ones((2,2),np.uint8)
# 	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,10))
# 	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101,1))
	
	
# 	# morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel)
#  # Encontramos los contornos presentes en fgmask, para luego basándonos
# 	# en su área poder determina si existe movimiento
# 	#cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# 	#a = cv2.getTrackbarPos('min','image')
# 	#b = cv2.getTrackbarPos('max','image')
# 	#f.write("Number of white pixels:"+ "\n")
    
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
 
# 	for cnt in cnts:
# 		#print(cv2.contourArea(cnt))
# 		#if cv2.contourArea(cnt) > 10000:
# 		if cv2.contourArea(cnt) > 1000:
# 			(x, y, w, h) = cv2.boundingRect(cnt)
# 			#band1 = True
# 			cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
# 			number_of_white_pix = np.sum(fgmask == 255) 
	
    #fgmask2 = cv2.dilate(fgmask2, None, iterations=1)
	# apply LoG filter
	LoG = cv2.Laplacian(gray, cv2.CV_32F)
	LoG = cv2.GaussianBlur(LoG, (5, 5), 0)

    # apply DoG filter
	DoG1 = cv2.GaussianBlur(gray, (3, 3), 0) - cv2.GaussianBlur(gray, (7, 7), 0)
	DoG2 = cv2.GaussianBlur(gray, (5, 5), 0) - cv2.GaussianBlur(gray, (11, 11), 0)

    # apply DoH filter
	DoH = cv2.GaussianBlur(gray, (5, 5), 0)
	Dxx = cv2.Sobel(DoH, cv2.CV_64F, 2, 0)
	Dyy = cv2.Sobel(DoH, cv2.CV_64F, 0, 2)
	Dxy = cv2.Sobel(DoH, cv2.CV_64F, 1, 1)
	DoH = (Dxx * Dyy) - (Dxy ** 2)

    # perform blob detection on the filtered images
	params = cv2.SimpleBlobDetector_Params()
	params.filterByArea = True
	params.minArea = 10
	params.filterByCircularity = False
	params.filterByConvexity = False
	params.filterByInertia = False

	detector = cv2.SimpleBlobDetector_create(params)
	keypoints_LoG = detector.detect(LoG)
	keypoints_DoG1 = detector.detect(DoG1)
	keypoints_DoG2 = detector.detect(DoG2)
	keypoints_DoH = detector.detect(DoH)

    # draw the detected blobs on the original image
	img_with_keypoints_LoG = cv2.drawKeypoints(gray, keypoints_LoG, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img_with_keypoints_DoG1 = cv2.drawKeypoints(gray, keypoints_DoG1, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img_with_keypoints_DoG2 = cv2.drawKeypoints(gray, keypoints_DoG2, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img_with_keypoints_DoH = cv2.drawKeypoints(gray, keypoints_DoH, np.array([]), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # display the resulting images
	cv2.imshow('LoG', img_with_keypoints_LoG)
	cv2.imshow('DoG1', img_with_keypoints_DoG1)
	cv2.imshow('DoG2', img_with_keypoints_DoG2)
	cv2.imshow('DoH', img_with_keypoints_DoH)

  
	k = cv2.waitKey(120) & 0xff
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows() 
