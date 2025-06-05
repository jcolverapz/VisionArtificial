import cv2
import numpy as np
import pdb


#img = cv2.imread('images/numeros.jpg')
#img = cv2.imread('images/gaps.png')
img = cv2.imread('images/vidrio2.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 140, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,0,255), 2)

edges = cv2.Canny(gray,50,255,apertureSize = 3)
minLineLength = 5
maxLineGap = 100
lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=100,minLineLength=minLineLength,maxLineGap=maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

#cv2.imwrite('probHough.jpg',img)

cv2.imshow('image', img)

cv2.waitKey()