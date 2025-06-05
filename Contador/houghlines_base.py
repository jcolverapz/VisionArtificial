import cv2 
import numpy as np
 
img = cv2.imread('images/sudoku.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
 
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
 
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
 
#cv2.imwrite('houghlines3.jpg',img)

#lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
cv2.imshow('houghlines3',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv22.imwrite('.\\detected\\{}'.format("p14_"+samplename),lines_edges)
