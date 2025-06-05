import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the imput image")
#args = vars(ap.parse_args())

#img = cv2.imread('images/sudoku.png')
img = cv2.imread('images/aerea1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#cv2.imwrite('edges_found.jpg',edges)
#lines = cv2.HoughLines(edges, 1, np.pi/180, 400)
lines = cv2.HoughLines(gray, 1, np.pi/180, 200)

for line in lines:
    for r,theta in line:
        a = np.cos(theta)
        b=np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)

   # cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

#cv2.imshow('edges_found', gray)
cv2.imshow('gray', gray)
cv2.imshow('img', img)
cv2.waitKey()