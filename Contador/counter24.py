import cv2
import numpy as np

# Read image
#img = cv2.imread('images/slika.jpg')
img = cv2.imread('images/aerea1.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show grayscale image
cv2.imshow('gray image', gray)
cv2.waitKey(0)

#BIG PROBLEM: IM FINDING VALUE OF `40` IN THE LINE BELOW MANUALLY

# Inverse binary threshold image with threshold at 40,
_, threshold_one = cv2.threshold(gray, 125 , 255, cv2.THRESH_BINARY_INV)

# Show thresholded image
cv2.imshow('threshold image', threshold_one)
cv2.waitKey(0)

# Find contours
contours, h = cv2.findContours(threshold_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print('Number of trees found:', len(contours))  #GIVES WRONG RESULT

# Iterate all found contours
for cnt in contours:

    # Draw contour in original/final image
    cv2.drawContours(img, [cnt], 0, (0, 0, 255), 1)

# Show final image
cv2.imshow('result image', img)
cv2.waitKey(0)