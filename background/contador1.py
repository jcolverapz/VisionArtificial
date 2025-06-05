import cv2
import numpy as np 
import imutils

img = cv2.imread('images/area4.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshhold, threshhold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(threshhold_img, 150, 200, 3, 5)


# lines = cv2.HoughLinesP(gray.copy(),1, np.pi/180, 100, minLineLength=150, maxLineGap=25)
# mid_xs = []


#lines = cv2.HoughLinesP(edges,1,np.pi/180,500, minLineLength = 600, maxLineGap = 75)[0].tolist()

# for x1,y1,x2,y2 in lines:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
    
    # Vertical lines
lines = cv2.HoughLinesP(threshhold_img, 1, np.pi, threshold=100, minLineLength=100, maxLineGap=1)

# Horizontal lines
lines2 = cv2.HoughLinesP(threshhold_img, 1, np.pi / 2, threshold=500, minLineLength=500, maxLineGap=1)
if lines is not None:
		for x1,y1,x2,y2 in lines:
			cv2.line(img,(x1,y1),(x2,y2),(0,255,0),4)

		#cv2.imshow("fgmask",fgmask)
#cv2.imshow("edges",edges)
		#cv2.imshow("frame",frame)
#lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
cv2.imshow('img',img)
#cv2.imshow('mask',mask)
cv2.waitKey()
cv2.destroyAllWindows()
#cv22.imwrite('.