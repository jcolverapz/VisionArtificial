from datetime import datetime
from urllib.request import urlopen
import numpy as np
import cv2  # OpenCV

cap = cv2.VideoCapture('videos/vidrio51.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

def find_spots_adaptive(frame):
	image_blur = cv2.GaussianBlur(frame, (13, 13), 0)
	binary_img = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									cv2.THRESH_BINARY_INV, 301, 30)
	binary_img = cv2.dilate(binary_img, None, iterations=1)
	binary_img = cv2.erode(binary_img, None, iterations=1)

	_, _, boxes, centroid = cv2.connectedComponentsWithStats(binary_img)

	boxes = boxes[1:]
	filtered_boxes = []
	filtered_centroid = []
	for (x, y, w, h, pixels), cent in zip(boxes, centroid):
		if pixels < 10000 and h > 10 and w > 10:
			filtered_boxes.append((x, y, w, h))
			filtered_centroid.append(cent)

	pixel_size_km = 363.313
	im_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
	oversize = 2
	for x, y, w, h in filtered_boxes:
		cv2.rectangle(im_rgb, (x - oversize, y - oversize), (x + w + oversize, y + h + oversize), (255, 0, 0), 1)

	for (x, y, w, h), (x_cent, y_cent) in zip(filtered_boxes, filtered_centroid):
		r = max(w / 2, h / 2)
		cv2.putText(im_rgb, f'{int(pixel_size_km * r)}km', (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
					cv2.LINE_AA)
	cv2.putText(im_rgb, f'{len(filtered_boxes)}', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 12, (255, 0, 0), 10, cv2.LINE_AA)
	cv2.imshow("binary_img", binary_img)
	return im_rgb

while True:
	_, frame = cap.read()
	find_spots_adaptive(frame)
 
	cv2.imshow("Biggest component", frame)
	cv2.waitKey()
# date = datetime(2022, 7, 17)
# #url = f'https://spaceweather.com/images{date.year}/{date.strftime("%d%b%y").lower()}/hmi4096_blank.jpg'
# #resp = urlopen(url)
# #image = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# spots_im = find_spots_adaptive(image)