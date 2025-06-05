import numpy as np
import matplotlib.pyplot as pyp
import cv2

segmented = cv2.imread('photo')
houghCircles = cv2.HoughCircles(segmented, cv2.HOUGH_GRADIENT, 1, 80, param1=450, param2=10, minRadius=30, maxRadius=200)
houghArray = np.uint16(houghCircles)[0,:]
def nothing(pos):
    pass

for circle in houghArray:
    cv2.circle(segmented, (circle[0], circle[1]), circle[2], (0, 250, 0), 3)
    


cap = cv2.VideoCapture('videos/vidrio50.mp4')
 
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()  
 
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel = np.ones((2,2),np.uint8)# apply morphology open with square kernel to remove small white spots

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)
 

while(True): 
	ret, frame = cap.read() 