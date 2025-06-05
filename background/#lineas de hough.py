#lineas de hough
lines = cv2.HoughLines(result, 1, np.pi / 180, 150, None, 0, 0)
    
			if lines is not None:
				for i in range(0, len(lines)):
					rho = lines[i][0][0]
					theta = lines[i][0][1]
					a = math.cos(theta)
					b = math.sin(theta)
					x0 = a * rho
					y0 = b * rho
					pt1 = (int(x0 + 100*(-b)), int(y0 + 100*(a)))
					pt2 = (int(x0 - 100*(-b)), int(y0 - 100*(a)))
					cv2.line(frame, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)