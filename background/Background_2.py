# Python code for Background subtraction using OpenCV 
import cv2
import numpy as np 
import cv2 
import imutils
counter=0

cap = cv2.VideoCapture('videos/vidrio20.mp4') 
object_detector = cv2.createBackgroundSubtractorMOG2(history = 10,varThreshold = 16, detectShadows=False)
 
def nothing(pos):
	pass
#create a dictionary of all trackers in OpenCV that can be used for tracking
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}
 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
 
# Create MultiTracker object
trackers = cv2.legacy.MultiTracker_create()

while True: 
	ret, frame = cap.read() 
	frame = imutils.resize (frame, width=720)
	#height, weight, _ = frame.shape

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
# apply morphology open with square kernel to remove small white spots
	fgmask = cv2.morphologyEx(gray, cv2.MORPH_OPEN,  kernel)
	#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
 
	area_pts = np.array([[10,10], [750,10], [750,500], [10,500]])
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
	image_area = cv2.bitwise_and(gray, gray, mask=imAux)
 		#fgmask = fgbg.apply(image_area)
	#fgmask = fgbg.apply(image_area)
	mask = object_detector.apply(frame)
# 	fgmask = fgbg.apply(frame) 
		#fgmask = cv2.dilate(fgmask, None, iterations=4)
	fgmask = cv2.dilate(mask, None, iterations=1)
	 
     
	(success, boxes) = trackers.update(frame)
 
# loop over the bounding boxes and draw then on the frame
	if success == False:
		bound_boxes = trackers.getObjects()
		idx = np.where(bound_boxes.sum(axis= 1) != 0)[0]
		bound_boxes = bound_boxes[idx]
		trackers = cv2.legacy.MultiTracker_create()
		for bound_box in bound_boxes:
			trackers.add(tracker,frame,bound_box)

	for i,box in enumerate(boxes):
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(frame,'TRACKING',(x+10,y-3),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)
 
	cv2.imshow('fgmask', fgmask) 
	cv2.imshow('frame',frame ) 
	k = cv2.waitKey(100) 
	

	if k == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
		roi = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS['kcf']()
		trackers.add(tracker, frame, roi)	
  
	cap.release() 
	cv2.destroyAllWindows() 
 