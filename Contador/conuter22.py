import numpy as np
import cv2
import random

#img = cv2.imread('images/circles.jpg')
img = cv2.imread('images/area5.jpg')
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
bilateral = cv2.bilateralFilter(gray,5,50,50)

minDist = 45
param1 = 45
param2 = 80
minRadius = 5
maxRadius = 150
minR = 0

for maxR in range(100,200,10):

    circles = cv2.HoughCircles(bilateral, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        # If there are some detections, convert radius and x,y(center) coordinates to integer
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            c= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            cv2.circle(img, (x, y), r, c, 3)                                      # draw circumference
            cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), c, 5)              # draw center

    #cv2.imwrite('circle.jpg', img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()