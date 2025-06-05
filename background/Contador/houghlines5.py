#!/usr/bin/env python3.5
#Opencv 4.0.1
#Date: 1st April, 2019

import cv2

img =  cv2.imread('images/cartera.jpg')
im2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(im2,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for i in range(0,len(contours)):
    cnt = contours[i]
    cv2.drawContours(img, [cnt],-1,(0,255,0)) 
    cv2.imshow('Features', img)
    cv2.waitKey()