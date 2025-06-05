import cv2
import numpy as np
import cv2
import imutils
#from tracker import *
#from skimage.morphology import medial_axis
#import pyodbc

counter=0

def nothing(pos):
    pass

def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 1)
cap = cv2.VideoCapture('videos/vidrio51.mp4')
#object_detector = cv2.createBackgroundSubtractorMOG2()
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
  
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)
 
def contour_intersect(cnt_ref,cnt_query, edges_only = True):
    
    intersecting_pts = []
    
    ## Loop through all points in the contour
    for pt in cnt_query:
        x,y = pt[0]

        ## find point that intersect the ref contour
        ## edges_only flag check if the intersection to detect is only at the edges of the contour
        
        if edges_only and (cv2.pointPolygonTest(cnt_ref,(x,y),True) == 0):
            intersecting_pts.append(pt[0])
        elif not(edges_only) and (cv2.pointPolygonTest(cnt_ref,(x,y),True) >= 0):
            intersecting_pts.append(pt[0])
            
    if len(intersecting_pts) > 0:
        return True
    else:
        return False
    
while(True):
	ret, frame = cap.read()
	frame = imutils.resize (frame, width=720)
	lc = cv2.getTrackbarPos('lc','Thresholds')
	hc = cv2.getTrackbarPos('hc','Thresholds')
	 
	size = np.size(frame)
	#fgmask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	 
	edged = cv2.Canny(gray, lc, hc)
	edged = cv2.dilate(edged, None, iterations=4)
	edged = cv2.erode(edged, None, iterations=1)
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

	area_pts_1 = np.array([[50,20], [500,20], [500,400], [50,400]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	fgmask = cv2.dilate(fgmask, None, iterations=3)

	# find contours.
	contours = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] 
	# Suppose this has the contours of just the car and the obstacle.

	# create an image filled with zeros, single-channel, same size as img.
	blank = np.zeros(fgmask.shape[0:2] )
	cv2.line(blank, (300, 0), (300, 300), (0, 255, 255), 3)

	# copy each of the contours (assuming there's just two) to its own image. 
	# Just fill with a '1'.
	img1 = cv2.drawContours( blank.copy(), contours, 0, 1 )
	img2 = cv2.drawContours( blank.copy(), contours, 1, 1 )

	# now AND the two together
	intersection = np.logical_and( img1, img2 )

	# OR we could just add img1 to img2 and pick all points that sum to 2 (1+1=2):
	intersection2 = (img1+img2)==2
	 
	cv2.drawContours(frame, [intersection2], -1, (255,0,255), 2)

	cv2.imshow('intersection2', intersection2)
	cv2.imshow('frame',frame )


	k = cv2.waitKey(100) & 0xff
	if k == 27:
		break

    
cap.release()
cv2.destroyAllWindows()
