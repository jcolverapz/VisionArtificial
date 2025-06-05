import cv2
import numpy as np 
import imutils
import math

wndname = "Line Drawing Demo"
width, height = 500, 500
image = np.zeros((height, width, 3), dtype = np.uint8)
cv2.line(image, [100,100], [200,200], (255,255,255), 1)
aX=100
aY=100
bX=200
bY=200
length=5
cv2.line(image,(aX,aY), (bX,bY), (250,0,0),2)
vX = bX-aX
vY = bY-aY
mag = math.sqrt(vX + vY)
vX = vX / mag
vY = vY / mag
temp = vX
vX = 0-vY
vY = temp
cX = bX + vX * length
cY = bY + vY * length
dX = bX - vX * length
dY = bY - vY * length

cv2.line(image,(bX,bY), (int(dX),int(dY)), (0,0,255),2)
cv2.imshow(wndname, image)
cv2.waitKey()
cv2.destroyAllWindows()