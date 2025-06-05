# Standard imports
import cv2
import numpy as np;

# Read image
#im = cv2.imread("./images/screw_simple.jpg", cv2.IMREAD_GRAYSCALE)
#im = cv2.imread("images/pieza1.jpg", cv2.IMREAD_GRAYSCALE)
im = cv2.imread("images/screw.jpg", cv2.IMREAD_GRAYSCALE)
#im = cv2.imread("images/aerea2.jpg", cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (1440, 880))

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10 #10
params.maxThreshold = 200 #200

# Filter by Area.
params.filterByArea = True # True
params.minArea = 500 #1500

# Filter by Circularity
params.filterByCircularity = True #True
params.minCircularity = 0.1 #0.1

# Filter by Convexity
params.filterByConvexity = True #True
params.minConvexity = 0.0 #0.87

# Filter by Inertia
params.filterByInertia = True #True
params.minInertiaRatio = 0.0 #0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    
    detector = cv2.SimpleBlobDetector_create(params) #Error

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
total_count = 0
for i in keypoints:
    total_count = total_count + 1


im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

print(total_count)