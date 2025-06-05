import cv2
import sys
import numpy as np

#img = cv2.imread('images/piso.jpg', cv2.IMREAD_GRAYSCALE)

im_gray = cv2.imread('images/blobs.jpg',cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

thresh = 50
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

#detect blobs based on features
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 1000

# Filter by Color (black=0)
params.filterByColor = False  # Set true for cast_iron as we'll be detecting black regions
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0.3
params.maxInertiaRatio = 0.9

# Distance Between Blobs
#params.minDistBetweenBlobs = 0

#thresholded to value 70 detecting blobs:

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im_bw)
print("Number of blobs detected are : ", len(keypoints))
#detect blobs: missing the detection based on features
#im_with_keypoints = cv2.drawKeypoints(im_bw, keypoints, np.array([]), (0, 0, 255),
                                    #  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im_gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("im_gray", im_gray)
cv2.imshow("im_bw", im_bw)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)