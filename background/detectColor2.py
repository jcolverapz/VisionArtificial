import cv2

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print("HSV values at ({}, {}): {}".format(x, y, hsv[y, x]))

img = cv2.imread('images/aerea1.jpg')
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_mouse)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()