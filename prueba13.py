import cv2
import numpy as np
#from matplotlib import pyplot as plt
fondo = cv2.imread('images/fondo.png')
video = cv2.VideoCapture('videos/vidrio23.mp4')
def nothing(pos):
	pass
cv2.namedWindow('Thresholds')
cv2.createTrackbar('LS','Thresholds',160,255, nothing)
cv2.createTrackbar('LH','Thresholds',255,255, nothing)

i = 0
while True:
	ret, frame = video.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((3, 3), np.float32) / 9
	dst = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
	cv2.imshow('Filtered', dst)
 # if i == 20:
	# 	bgGray = gray
		#bgGray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)
	# if i > 20:
	# 	dif = cv2.absdiff(gray, bgGray)
		#_, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
	#ls=cv2.getTrackbarPos('LS','Thresholds')
	#lh=cv2.getTrackbarPos('LH','Thresholds')
	#_, th = cv2.threshold(dif, ls, lh, cv2.THRESH_BINARY)
		# Para OpenCV 3
		#_, cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# Para OpenCV 4
	
 
	# create a list of first 5 frames
	#img = [cap.read()[1] for i in range(5)]
	# convert all to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
	# convert all to float64
	#gray = [np.float64(i) for i in gray]
	gray = np.float64(frame)  
	# create a noise of variance 25
	noise = np.random.randn(*gray[1].shape)*10
	# Add this noise to images
	noisy = [i+noise for i in gray]
	# Convert back to uint8
	noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]
	# Denoise 3rd frame considering all the 5 frames
	dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
	# plt.subplot(131),plt.imshow(gray[2],'gray')
	# plt.subplot(132),plt.imshow(noisy[2],'gray')
	# plt.subplot(133),plt.imshow(dst,'gray')
	# plt.show()
  
	cv2.imshow('gray',gray)
	#cv2.imshow('fgmask',fgmask)
	cv2.imshow('Frame',frame)

	i = i+1
	if cv2.waitKey(80) & 0xFF == ord ('q'):
		break
video.release()