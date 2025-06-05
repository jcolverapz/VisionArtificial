# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2 
import imutils

#cap = cv2.VideoCapture('/home/sourabh/Downloads/people-walking.mp4') 
cap = cv2.VideoCapture('videos/vidrio23.mp4') 
#cap = cv2.VideoCapture('videos/vidrio0.mp4') 
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorMOG2() 
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
def nothing(pos):
	pass
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#kernel = np.ones((5,5),np.uint8)
cv2.namedWindow('Thresholds')
cv2.createTrackbar('value','Thresholds',0,255,nothing)

while(1): 
	#img, frame = cap.read() 
	_, frame = cap.read() 
	#frame = imutils.resize (frame, width=720)
	#img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
	fgmask = fgbg.apply(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	value = cv2.getTrackbarPos('value','Thresholds')
	retval, threshold = cv2.threshold(frame, value, 255, cv2.THRESH_BINARY_INV)
 
 
# Change thresholds
	params = cv2.SimpleBlobDetector_Params()
	params.minThreshold = 0
	params.maxThreshold = 255


# Filter by Area.
	params.filterByArea = True
	params.minArea = 100

# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.1

# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.87

# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.01

# Create a detector with the parameters
# OLD: detector = cv2.SimpleBlobDetector(params)
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
	keypoints = detector.detect(gray)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

	#im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.imshow("threshold", threshold)
	#cv2.waitKey(0)
  
	k = cv2.waitKey(120) & 0xff
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows() 
