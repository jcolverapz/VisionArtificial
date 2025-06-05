import numpy as np
import cv2
import scipy.ndimage as ndi

#img = cv2.imread('images/tenis.jpg')
img = cv2.imread('images/aerea1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smooth = ndi.filters.median_filter(gray, size=2)
edges = smooth > 180
lines = cv2.HoughLines(edges.astype(np.uint8), 120, np.pi/180, 120)

for rho,theta in lines[0]:
    print(rho, theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Show the result
cv2.imshow("Line Detection", img)
cv2.waitKey()