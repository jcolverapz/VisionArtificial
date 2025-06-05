import cv2

#img = cv2.imread('images/test3.png')
img = cv2.imread('images/vidrio2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)

_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

#findContours
contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

canvas = img.copy()

## draw approx contours

for cnt in contours:
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, arclen*0.005, True)
    #drawContours
    cv2.drawContours(canvas, [approx], -1, (0,0,255), 1, cv2.LINE_AA)

#cv2.imwrite("result.png", canvas)

cv2.imshow("approx", canvas)
cv2.waitKey(0)

cv2.destroyAllWindows()