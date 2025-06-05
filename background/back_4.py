import cv2
import numpy as np
import cv2
import imutils
#from tracker import *
#from skimage.morphology import medial_axis

counter=0
 
def nothing(pos):
    pass
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance
 
cap = cv2.VideoCapture('videos/vidrio51.mp4')
#object_detector = cv2.createBackgroundSubtractorMOG2()
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
 
# Function to index and distance of the point closest to an array of points
# borrowed shamelessly from: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points

#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold= 40)
#tracker = EuclideanDistTracker() # type: ignore
 
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

	#area_pts_1 = np.array([[50,20], [500,20], [500,400], [50,400]])
	area_pts_1 = np.array([[100,200], [400,200], [400,300], [100,300]])

	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts_1], -1, (255), -1)
	image_area = cv2.bitwise_and(frame, frame, mask=imAux)
	fgmask = object_detector.apply(image_area)
	fgmask = cv2.dilate(fgmask, None, iterations=3)

	# Find Contours
	contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#image2 = np.zeros((300, 300), dtype=np.uint8)
	cv2.drawContours(frame, contours, -1, (0, 255, 0), 2) 

# Read image img

    # Binarize
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_#,thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
	#_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# # Draw Contours
	blank_mask = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
	# contours_idx = blank_mask[...,1] == 255
	# # Define lines coordinates
	line1 = [300, 10, 300, 300]
	 
	# # Draw Lines over Contours
	cv2.line(blank_mask, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0), thickness=1)
	# lines_idx = blank_mask[...,0] == 255
	# overlap = np.where(contours_idx * lines_idx)
	# #(array([ 90, 110, 110, 140, 140], dtype=int64), array([ 80,  40, 141,  27, 156], dtype=int64))
	# # # Show Image
	# # fig = plt.figure()
	# ax  = fig.add_subplot(111)
	# ax.imshow(blank_mask)
	#contours = np.vstack([contours[5], contours[6]])
	# #cv2.imshow('overlap',overlap )

	# Detect corners using the contours
	corners = cv2.goodFeaturesToTrack(image=gray,maxCorners=1000,qualityLevel=0.01,minDistance=50) # Determines strong corners on an image
	cv2.drawContours(blank_mask, [frame], -1, 255, 2)
	cv2.imshow('blank_mask',blank_mask )
	# Draw the corners on the original image
	for corner in corners:
		x,y = corner.ravel()
		cv2.circle(frame,(x,y),10,(0,0,255),-1)
    
	# Find horizonal lines
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,5))
	horizontal = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

	# Find vertical lines
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,150))
	vertical = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

	# Find joints
	joints = cv2.bitwise_and(blank_mask, vertical)
	cv2.imshow('joints',joints )
 
	k = cv2.waitKey(100) & 0xff
	if k == 27:
		break

    
cap.release()
cv2.destroyAllWindows()
