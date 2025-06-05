import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# load image
img = cv2.imread("images/aerea1.jpg")
# resize for ease of use
#img_ori = cv2.resize(img_large, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
# create grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create mask for image size
mask = np.zeros((img.shape[:2]), dtype=np.uint8)
# do a morphologic close to merge dotted line
kernel = np.ones((8, 8))
res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# detect edges for houghlines
edges = cv2.Canny(res, 50, 50)
# detect lines
lines = cv2.HoughLines(edges, 1, np.pi/180, 250)
# draw detected lines
if lines is not None:
	for line in lines:
		rho, theta = line[0]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*a)
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*a)
		cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
		cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 1)

#cv2.imwrite('houghlines3.jpg',img)

#lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
cv2.imshow('img',img)
cv2.imshow('mask',mask)
cv2.waitKey()
cv2.destroyAllWindows()
#cv22.imwrite('.\\detected\\{}'.format("p14_"+samplename),lines_edges)

#lines = cv2.HoughLines(edges, 1, np.pi/180, 200)