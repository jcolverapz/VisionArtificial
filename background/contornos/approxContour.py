import cv2

#img = cv2.imread('images/test3.png')
img = cv2.imread('images/vidrio2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)

_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# cv2.imshow("binary", binary)
# cv2.waitKey(0)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.009 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    cv2.drawContours(img, [approx], -1, (0, 255, 255), 1)

cv2.imshow("approx", img)
cv2.waitKey(0)

cv2.destroyAllWindows()