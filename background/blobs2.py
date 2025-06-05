import cv2
import numpy as np;

# Read image
#img = cv2.imread("images/blobs2.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("images/screw.jpg", cv2.IMREAD_GRAYSCALE)
retval, threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 255;

blur = cv2.GaussianBlur(img,(5,5),0)

#params.filterByCircularity = True
params.filterByCircularity = False
params.minCircularity = 0.2

params.filterByArea = True;
params.minArea = 10;

# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3 :
#     detector = cv2.SimpleBlobDetector(params)
# else :
detector = cv2.SimpleBlobDetector_create(params) #version 4

# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector()

# Detect blobs.
keypoints = detector.detect(threshold)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("threshold", threshold)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey()