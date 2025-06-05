# Python program to illustrate 
# corner detection with 
# Shi-Tomasi Detection Method 
	
# organizing imports 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
#%matplotlib inline 

# path to input image specified and 
# image is loaded with imread command 
#img = cv2.imread('images/chess.png') 
#img = cv2.imread('images/aerea1.jpg') 
#img = cv2.imread('images/area4.jpg') 
img = cv2.imread('images/central.jpg') 

#cv2.line(img, (400, 100), (400, 1000), (0,0,0), 2)	
# area_pts = np.array([[20,100], [100,100], [100,200], [20,200]])
# imAux = np.zeros(shape=(img.shape[:2]), dtype=np.uint8)
# imAux = cv2.drawContours(imAux, [area_pts], -1, (0,0,0), 2)
# image_area = cv2.bitwise_and(img, img, mask=imAux)
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.line(img, (400, 100), (400, 1000), (0,0,0), 2)	
cv2.line(img, (150, 5), (150, 1000), (0,0,0), 2)	
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

#area_pts = np.array([[280,100], [700,100], [700,1000], [280,1000]])
area_pts = np.array([[5,10], [200,10], [200,300], [5,300]])
imAux = np.zeros(shape=(img.shape[:2]), dtype=np.uint8)
imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
image_area = cv2.bitwise_and(gray, gray, mask=imAux)

#area_pts = np.array([[280,20], [700,20], [700,800], [280,800]])

# convert image to grayscale 

# Shi-Tomasi corner detection function 
# We are detecting only 100 best corners here 
# You can change the number to get desired result. 
#corners = cv2.goodFeaturesToTrack(gray_img, 1000, 0.01, 10) 
corners = cv2.goodFeaturesToTrack(image_area,5000, 0.1, 10) 

# convert corners values to integer 
# So that we will be able to draw circles on them 
corners = np.int_(corners) 

# draw red color circles on all corners 
for i in corners: 
	x, y = i.ravel() 
	#cv2.circle(img, (x, y), 2, (0, 0, 255), -1) 
#cv2.line(img, (400, 100), (400, 1000), (0,0,0), 2)	
	cv2.line(img, (x, y), (x+20, y), (0, 0, 255), 2) 

#fgmask = object_detector.apply(image_area)
#fgmask = cv2.dilate(fgmask, None, iterations=3)

# resulting image 
#cv2.imshow(img) 
#cv2.imshow('image_area', image_area)

cv2.drawContours(img, [area_pts], -1, (0,255,0), 2)
cv2.imshow('img', img)
cv2.imshow('gray', gray)

cv2.waitKey()



# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
# convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Use Shi-Tomasi to detect object corners / edges from initial frame
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)
# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (100, 3))
# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



# # De-allocate any associated memory usage 
# if cv2.waitKey(0) & 0xff == 27: 
# 	cv2.destroyAllWindows() 
