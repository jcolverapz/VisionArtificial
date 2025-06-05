import cv2
import imutils
import numpy as np

#img = cv2.imread('images/lines.png', 0)
img = cv2.imread('images/area4.jpg')
cv2.imshow('original Image', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh1', thresh)

contours, hierarchy =    cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Number of lines:', len(contours))

img = imutils.resize (img, width=1024)
#img = resize_aspect_ratio(img, 800) # resize image for display + lets not use such a big image for now
area_pts = np.array([[5,5], [10,5], [10,10], [5,10]])
imAux = np.zeros(shape=(img.shape[:2]), dtype=np.uint8)
imAux = cv2.drawContours(imAux, [area_pts], -1, (255,0,255), -1)
image_area = cv2.bitwise_and(img, img, mask=imAux)
cv2.drawContours(img, [area_pts], -1, (255,0,255), 2)


cv2.waitKey(0)
cv2.destroyAllWindows()