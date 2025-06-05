import cv2
import numpy as np

#img = cv2.imread("assets/shapes.png")
img = cv2.imread("vidrio1.png")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Shape", img)
cv2.imshow("gray", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
