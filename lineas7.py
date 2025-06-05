import cv2
import numpy as np
img = cv2.imread('images/gaps.png')
#blur_img = cv2.GaussianBlur(img, (5, 5), 0)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
height, width = gray_img.shape
horizontal = gray_img.copy()
horizontal_size = int(width/30)
horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                             (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontal, horizontal_struct, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontal, horizontal_struct, (-1, -1))
cv2.imshow('image', horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()