import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('images/examen.png', 0)

template = cv2.imread('/path/circle.jpg', 0)
h, w = template.shape

res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

#cv2.imwrite('res.png',img)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()