import cv2
import random


def nothing(pos):
    pass
#img = cv2.imread("images/bricks.jpg")
img = cv2.imread("images/area4.jpg")

# To hsv
hsv =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Get the Saturation out
S=hsv[:,:,1]

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',100,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)

#cv2.imwrite('image1.jpg',inv_img)
while (True):
    lc = cv2.getTrackbarPos('lc','Thresholds')
    hc = cv2.getTrackbarPos('hc','Thresholds')
    # Threshold it
    #(ret,T)=cv2.threshold(S,42,255,cv2.THRESH_BINARY)
    (ret,T)=cv2.threshold(S,lc,hc,cv2.THRESH_BINARY)

    # Show intermediate result
    cv2.imshow('win',T)
    cv2.waitKey(0)

    # Find contours
    contours,h = cv2.findContours(T, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #img2 = img.copy()


    for c in contours:
        area = cv2.contourArea(c)
        # Only if the area is not miniscule (arbitrary)
        if area > 0:
            (x, y, w, h) = cv2.boundingRect(c)

            # Uncomment if you want to draw the conours
            cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

            # Get random color for each brick
            tpl = tuple([random.randint(0, 255) for _ in range(3)])
            cv2.rectangle(img, (x, y), (x + w, y + h), tpl, -1)

    cv2.imshow("bricks", img)
    cv2.waitKey(0)