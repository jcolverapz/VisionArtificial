
	# Set up the detector with default parameters.
	detector = cv2.SimpleBlobDetector()
	
	# Detect blobs.
	keypoints = detector.detect(fgmask)
	
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	Show keypoints
	cv2.imshow("Keypoints", im_with_keypoints)
 
  
	
	cv2.imshow("frame", frame)
	cv2.imshow('fgmask', fgmask)
	 