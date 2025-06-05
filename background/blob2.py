# Standard imports
import cv2
import numpy as np;

# Read image
#im = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/vidrio50.mp4')

while(True):
        
        ret, frame=cap.read()


        lower = (130,150,80)  #130,150,80
        upper = (250,250,120) #250,250,120
        mask = cv2.inRange(frame, lower, upper)
        contours, upper = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blob = max(contours, key=lambda el: cv2.contourArea(el))
        M = cv2.moments(blob)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        canvas = cap.copy()
        cv2.circle(canvas, center, 2, (0,0,255), -1)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()