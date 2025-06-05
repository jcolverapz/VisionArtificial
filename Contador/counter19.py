import cv2
#img = cv2.imread("images/lines.png")
img = cv2.imread("images/area4.jpg")
h,w = img.shape[0:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(img, 50, 200, None, 3)


def dist(x, y, x1, y1):
    return ((x-x1)**2+(y-y1)**2)**(0.5)


def slope(x, y, x1, y1):
    if y1 != y:
        return ((x1-x)/(y1-y))
    else:
        return 0


fld = cv2.ximgproc.createFastLineDetector()
lines = fld.detect(edges)
no_of_hlines = 0
#result_img = fld.drawSegments(img, lines)
for line in lines:
    x0 = int(round(line[0][0]))
    y0 = int(round(line[0][1]))
    x1 = int(round(line[0][2]))
    y1 = int(round(line[0][3]))
    d = dist(x0, y0, x1, y1)
    if d>150: #You can adjust the distance for precision
        m = (slope(x0, y0, x1, y1))
        if m ==0: #slope for horizontal lines and adjust slope for vertical lines
            no_of_hlines+=1
            cv2.line(img, (x0, y0), (x1, y1), (255, 0, 255), 1, cv2.LINE_AA)
print(no_of_hlines)
cv2.imshow("lines",img)
cv2.waitKey(0)
cv2.destroyAllWindows()