import cv2
import numpy as np

img = cv2.imread('images/blobs2.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set up the detector and configure its params.
params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 0
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 10
params.maxArea = 300000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 1
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypointsb = detector.detect(img)

# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(img, keypointsb, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imwrite('test3.png',im_with_keypoints)
cv2.waitKey()