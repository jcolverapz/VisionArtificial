"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2
import numpy as np
def main(argv):
    
	#default_file = 'images/gaps.png'
	#default_file = 'images/area5.jpg'
	default_file = 'images/sudoku.png'
	filename = argv[0] if len(argv) > 0 else default_file
	# Loads an image
	src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
	# Check if image is loaded fine
	if src is None:
		print ('Error opening image!')
		print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
		return -1


	dst = cv2.Canny(src, 50, 200, None, 3)

	# Copy edges to the images that will display the results in BGR
	cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
	cdstP = np.copy(cdst)

	lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 200*(-b)), int(y0 + 200*(a)))
			pt2 = (int(x0 - 200*(-b)), int(y0 - 200*(a)))
			if rho<0.9:
				cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    
    
	linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
	if linesP is not None:
		for i in range(0, len(linesP)):
			l = linesP[i][0]
			cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
cv2.imshow("Source", src)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv2.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])