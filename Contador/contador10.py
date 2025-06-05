import cv2
import numpy as np
import math

#path='images/aerea1.jpg'
path='images/area4.jpg'
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dst = cv2.Canny(gray, 50, 255, None, 3)
#linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
linesP = cv2.HoughLinesP(gray, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]

        #here l contains x1,y1,x2,y2  of your line
        #so you can compute the orientation of the line 
        p1 = np.array([l[0],l[1]])
        p2 = np.array([l[2],l[3]])

        p0 = np.subtract( p1,p1 ) #not used
        p3 = np.subtract( p2,p1 ) #translate p2 by p1

        angle_radiants = math.atan2(p3[1],p3[0])
        angle_degree = angle_radiants * 180 / math.pi

        print("line degree", angle_degree)
        #if angle_degree ==0 or  angle_degree == 180:

        if 0 < angle_degree < 35 or 0 > angle_degree > -35 :
        #if 0 < angle_degree < 45 :
           # cv2.line(image,  (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
            cv2.line(image,  (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)


cv2.imshow("gray", gray)
cv2.imshow("dst", dst)
cv2.imshow("Source", image)

print("Press any key to close")
cv2.waitKey(0)
cv2.destroyAllWindows()