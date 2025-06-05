import cv2
import numpy as np
import math

# read image
#img = cv2.imread('images/lineas4.png')
img = cv2.imread('images/area4.jpg')

# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray,165,255,cv2.THRESH_BINARY)[1]

# apply close to connect the white areas
kernel = np.ones((15,1), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = np.ones((17,3), np.uint8)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

# apply canny edge detection
edges = cv2.Canny(img, 175, 200)

# get hough lines
result = img.copy()
lines= cv2.HoughLines(edges, 1, math.pi/180.0, 165, np.array([]), 0, 0)
a,b,c = lines.shape
for i in range(a):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0, y0 = a*rho, b*rho
    pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
    pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
    cv2.line(result, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)


# save resulting images
# cv2.imwrite('fabric_equalized_thresh.jpg',thresh)
# cv2.imwrite('fabric_equalized_morph.jpg',morph)
# cv2.imwrite('fabric_equalized_edges.jpg',edges)
# cv2.imwrite('fabric_equalized_lines.jpg',result)

# show thresh and result    
cv2.imshow("thresh", thresh)
cv2.imshow("morph", morph)
cv2.imshow("edges", edges)
cv2.imshow("result", result)
cv2.waitKey()
cv2.destroyAllWindows()