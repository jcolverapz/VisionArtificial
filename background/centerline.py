import cv2
import numpy as np

img = cv2.imread('images/centerline2.png')
mask = np.zeros((img.shape[:2]), np.uint8)
h2, w2 = img.shape[:2]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)

_, thresh = cv2.threshold(equ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    print(h, w)
    if h < 30 and w > 270:
        cv2.drawContours(mask, [cnt], 0, (255,255,255), -1)

res = cv2.bitwise_and(img, img, mask=mask)
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(thresh,(5,5),0)
contours, hierarchy = cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)

M = cv2.moments(cnt)
cy = int(M['m01']/M['m00'])

mask = np.zeros((img.shape[:2]), np.uint8)
cv2.drawContours(mask, [cnt], 0, (255,255,255), -1)

up = []
down = []

for i in cnt:
    x = i[0][0]
    y = i[0][1]
    if x == 0:
        pass
    elif x == w2:
        pass
    else:
        if y > cy:
            down.append(tuple([x,y]))
        elif y < cy:
            up.append(tuple([x,y]))
        else:
            pass


up.sort(key = lambda x: x[0])
down.sort(key = lambda x: x[0])

up_1 = []
down_1 = []

for i in range(0, len(up)-1):
    if up[i][0] != up[i+1][0]:
        up_1.append(up[i])
    else:
        pass

for i in range(0, len(down)-1):
    if down[i][0] != down[i+1][0]:
        down_1.append(down[i])
    else:
        pass

lines = zip(up_1, down_1)

for i in lines:
    x1 = i[0][0]
    y1 = i[0][1]
    x2 = i[1][0]
    y2 = i[1][1]
    middle = np.sqrt(((x2-x1)**2)+((y2-y1)**2))
    cv2.circle(img, (x1, y1+int(middle/2)), 1, (0,0,255), -1)    

cv2.imshow('img', img)