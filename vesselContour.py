import cv2
import numpy as np

#img = cv2.imread('vessel.png')
#img = cv2.imread('images/vessel.png')
img = cv2.imread('images/gaps.png')
#img = cv2.imread('images/test.jpg')
h, w, ch = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,100,200)
contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
points = []

cv2.imshow('edges', edges)

for cnt in contours:
    for i in cnt[:,0]:
        x = int(i[0])
        y = int(i[1])
        if x == 0:
            points.append((x,y))
        elif y == 0:
            points.append((x,y))
        elif w-1<= x <= w+1:
            points.append((x,y))
        elif h-1<= y <= h+1:
            points.append((x,y))

if len(points) == 4:
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    dist1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    dist2 = np.sqrt((x3-x1)**2 + (y3-y1)**2)    
    dist3 = np.sqrt((x4-x1)**2 + (y4-y1)**2)

    if dist2 < dist1 and dist2 < dist3:
        cv2.line(edges, (x3,y3), (x1,y1), 255, 1)
        cv2.line(edges, (x2,y2), (x4,y4), 255, 1)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, 0, (0,255,0), 2)

cv2.imshow('img', img)
cv2.imshow('edges+lines', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()