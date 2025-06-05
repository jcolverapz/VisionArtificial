#HoughLine
import cv2
import numpy as np
samplename = "sam09.jpg"

#First, get the gray image and process GaussianBlur.
img = cv2.imread('images/area5.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
#blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

#Second, process edge detection use Canny.
low_threshold = 50
high_threshold = 150
#edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
# cv2.imshow('photo2',edges)
# cv2.waitKey(0)
#Then, use HoughLinesP to get the lines. You can adjust the parameters for better performance.

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(gray, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
#print(lines)
#print(len(lines))
for line in lines:
	for x1,y1,x2,y2 in line:
		if lines[1][1] - lines[1][3] == 0:
		# Horizontal line
			cv2.line(line_image,(x1,y1),(x2,y2 ),(255,0,0),1)

#Finally, draw the lines on your srcImage.
# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
cv2.imshow('photo',lines_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('.\\detected\\{}'.format("p14_"+samplename),lines_edges)
