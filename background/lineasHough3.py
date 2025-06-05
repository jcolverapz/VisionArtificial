import numpy as np
import cv2
import math

#image = cv2.imread('images/lineas5.jpg')
#image = cv2.imread('images/aerea1.jpg')
image = cv2.imread('images/area4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 250)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=100, maxLineGap=50)

hough = np.zeros(image.shape, np.uint8)

for line in lines:
    for i in range(0, len(lines)):
        x1, y1, x2, y2 = line[0]
        
        l = lines[i][0]

        #here l contains x1,y1,x2,y2  of your line
        #so you can compute the orientation of the line 
        p1 = np.array([l[0],l[1]])
        p2 = np.array([l[2],l[3]])

        p0 = np.subtract( p1,p1 ) #not used
        p3 = np.subtract( p2,p1 ) #translate p2 by p1

        angle_radiants = math.atan2(p3[1],p3[0])
        angle_degree = angle_radiants * 180 / math.pi
            
        #if 0 < angle_degree < 15 or 0 > angle_degree > -15 :
        if angle_degree ==0 or  angle_degree == 180:
            cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(angle_degree)
            cv2.waitKey()
#cv2.imwrite('hough.jpg', hough)


cv2.imshow('hough',hough)
cv2.waitKey()
cv2.destoryAllWindows(0)