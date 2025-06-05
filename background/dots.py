import cv2
import numpy as np
import matplotlib.pyplot as plt
#27-feb-2025

#frame=cv2.imread("images/dots.jpg")
frame=cv2.imread("images/aerea1.jpg")
dots=np.zeros_like(frame)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
lower_hsv = np.array([52, 0, 55])
higher_hsv = np.array([104, 255, 255])
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower = np.array([52, 0, 55])
# upper = np.array([104, 255, 255])

mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

cnts, h = cv2.findContours( mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
mnts  = [cv2.moments(cnt) for cnt in cnts]

#centroids = [( int(round(m['m10']/m['m00'])),int(round(m['m01']/m['m00'])) ) for m in mnts]

print(len(cnts))
#for c in centroids:
for c in cnts:
    #imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    cv2.drawContours(frame, c, -1, (0,255,0), 1)
#     cv2.circle(dots,c,5,(0,255,0))
#     print (c)

cv2.imshow('frame', frame)
cv2.imshow('hsv', hsv)
cv2.imshow('mask', mask)
cv2.imshow('red_dots', dots)

#plt.figure(figsize=(8, 8))
#plt.title(title)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()
#plot the intercepting points
#plt.imshow('img', hsv)
#plt.plot(x[1], y[1], 'rs', label='second intercept')
#plt.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
plt.show()
#plot.show()


#cv2.waitKey(0)
cv2.destroyAllWindows()