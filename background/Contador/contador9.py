import numpy as np
import cv2
import matplotlib.pyplot as plt 
import imutils

def resize_aspect_ratio(img, height = None):
    original_height, original_width = img.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(height * aspect_ratio)
    return cv2.resize(img, (new_width, height))

if __name__ == "__main__":
    
    # Parameters set looking at the histogram below, this is brittle
	min_screw_area = 30
	max_screw_area = 200
	overlap_coef = 1.05 # Blobs of screw looses area because of overlaping, this correct that a little

	#img = cv2.imread("images/screw.jpg")
	#img = cv2.imread("images/pieza1.jpg")
	img = cv2.imread("images/aerea1.jpg")
	#img = cv2.imread("images/aerea1.jpg")
	#img = cv2.imread("images/lateral1.jpg")
	img = imutils.resize (img, width=1024)
	#img = resize_aspect_ratio(img, 800) # resize image for display + lets not use such a big image for now
	area_pts = np.array([[20,100], [100,100], [100,200], [20,200]])
	imAux = np.zeros(shape=(img.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255,0,255), 2)
	image_area = cv2.bitwise_and(img, img, mask=imAux)
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
	#_, thresh = cv2.threshold(imgray, 127, 255, 0)
	_, thresh = cv2.threshold(imgray, 100, 255, 0)
	thresh = cv2.bitwise_not(thresh) # inverse the image so that objects of interest are white
 
	#fgmask = object_detector.apply(image_area)
	# Find the contours and remove small ones (noise)
	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = list(filter(lambda cnt: cv2.contourArea(cnt) > min_screw_area, contours)) 

	# Compute contours area and assuming most of the screws are separated compute typical size of a screw
	areas = [cv2.contourArea(cnt) for cnt in contours] 
	median_area_screw = np.median(areas)

	# Number of screws = individual screws + multiple_screw_areas / typical size of a screw
	multiple_screw_areas = list(filter(lambda area: area > max_screw_area, areas))
	num_screws = sum([np.round(area*overlap_coef/median_area_screw ) for area in multiple_screw_areas]) 
	num_screws += len(areas) - len(multiple_screw_areas)

	#print(f"Estimated number of screws: {num_screws}")

	cv2.putText(img, str(len(contours)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

	# Display some info for experimentation
	for index in range(len(contours)):
		cnt =  contours[index]
		(x, y, w, h) = cv2.boundingRect(cnt)
		cv2.putText(img, str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
		if cv2.contourArea(cnt) > max_screw_area:
			cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1) 
		else:
			cv2.drawContours(img, [cnt], -1, (0, 80, 255), 1)
   
   
	cv2.imshow('thresh', thresh) 
	cv2.imshow('Contour', img) 
	plt.hist(areas, bins=100)
	plt.show()
	cv2.waitKey(0) 
	cv2.destroyAllWindows()