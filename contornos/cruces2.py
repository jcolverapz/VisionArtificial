import cv2
import numpy as np

frame=cv2.imread("images/dots.jpg")
dots=np.zeros_like(frame)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([112, 176, 174])
higher_hsv = np.array([179,210,215])
mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

cnts, h = cv2.findContours( mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
mnts  = [cv2.moments(cnt) for cnt in cnts]

centroids = [( int(round(m['m10']/m['m00'])),int(round(m['m01']/m['m00'])) ) for m in mnts]


for c in centroids:
    cv2.circle(dots,c,5,(0,255,0))
    print (c)

cv2.imshow('red_dots', dots)

cv2.waitKey(0)
cv2.destroyAllWindows()