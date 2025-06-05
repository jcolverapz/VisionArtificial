import cv2
#from tracker import *
from  funciones.tracker  import  *
# Create tracker object
#tracker = EuclideanDistTracker()
tracker = EuclideanDistTracker() # type: ignore
cap = cv2.VideoCapture("videos/vidrio0.mp4")
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=0, detectShadows=False)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # Extract Region of interest
    #roi = frame[10: 650,200: 400]
    roi = frame[10: 650,100: 700]
    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=5)
    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 10000:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.waitKey() 
            #cv2.putText(frame, "w: " + str(w) + ", h:" + str(h), (x+10),(y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.waitKey() 
            cv2.putText(frame, "w: " + str(w), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #if 500 > w > 280:
            detections.append([x, y, w, h])
            
        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(100)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()