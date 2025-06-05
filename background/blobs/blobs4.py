import cv2
import sys
import numpy as np

img = cv2.imread('images/piso.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('images/blobs2.jpg', cv2.IMREAD_GRAYSCALE)
#mg = cv2.imread('images/blobs3.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('images/blobs2.png', cv2.IMREAD_GRAYSCALE)

# set up blob detector params
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByInertia = True
detector_params.minInertiaRatio = 0.001

detector_params.filterByArea = True
detector_params.maxArea = 10000000
detector_params.minArea = 1000

detector_params.filterByCircularity = True
detector_params.minCircularity = 0.0001
detector_params.filterByConvexity = True
detector_params.minConvexity = 0.01

detector = cv2.SimpleBlobDetector_create(detector_params)

# Detect blobs.
keypoints = detector.detect(img)

# print properties of identified blobs
for p in keypoints:
    print(p.pt) # locations of blobs
    # circularity???
    # inertia???
    # area???
    # convexity???
    # etc...
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)