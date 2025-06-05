import cv2
import numpy as np
from matplotlib import pyplot as plt
#image = cv2.imread('images/drops.jpg')
image = cv2.imread('images/area4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY
plt.imshow(gray, cmap='gray')
blur = cv2.GaussianBlur(gray, (11, 11), 0)
plt.imshow(blur, cmap='gray')
canny = cv2.Canny(blur, 30, 40, 3)
plt.imshow(canny, cmap='gray')
dilated = cv2.dilate(canny, (1, 1), iterations=0) 
plt.imshow(dilated, cmap='gray')
(contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
cv2.putText(rgb, str(len(contours)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
for index in range(len(contours)):
	cnt = contours[index]
	#cnt = contours(index)
	#(x, y, w, h) = cv2.boundingRect(cnt)

plt.imshow(rgb)
#print("No of circles: ", len(cnt))

# Show keypoints
cv2.imshow("circles", image)
cv2.imshow("rgb", rgb)
cv2.waitKey()