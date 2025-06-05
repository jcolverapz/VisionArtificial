import cv2

#image = cv2.imread('images/slika.jpg')
#image = cv2.imread('images/aerea1.jpg')
image = cv2.imread('images/central.jpg')
#image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (5,5), 0)
#thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=2)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
trees = 0
for c in cnts:
    area = cv2.contourArea(c)
    # if area > 50:
    #     x,y,w,h = cv2.boundingRect(c)
    cv2.drawContours(image, [c], -1, (36,255,12), 1)
        #trees += 1

print('Trees:', trees)
cv2.imshow('image', image)
#cv2.imshow('blur', blur)
cv2.imshow('thresh', thresh)
cv2.waitKey()