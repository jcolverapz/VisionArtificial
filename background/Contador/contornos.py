import cv2

# encuentra el controno mas grande

maxsize = 0  
best = 0  
count = 0
img = cv2.imread('images/gaps.png')


# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for cnt in contours:  
    if cv2.contourArea(cnt) > maxsize:  
        maxsize = cv2.contourArea(cnt)  
        best = count  
    count += 1  

cv2.drawContours(img, contours[best], -1, (0,0,255), 2) 
#cv2.drawContours(img, contours, best, (0,0,255), 2)
cv2.imshow('Image', img)
cv2.waitKey(0)