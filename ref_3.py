# HARRIS CORNER DETECTION
			corners = cv2.goodFeaturesToTrack(output, maxCorners=50,
							qualityLevel=0.01, minDistance=50,
							useHarrisDetector=True, k=0.1)
			corners = np.int_(corners)
			# Create a black image
			img = np.zeros((512,512,3), np.uint8)
   
   for c in corners:
				x, y = c.ravel()
				img = cv2.circle(frame, center=(x, y), radius=5, 
								color=(0, 250, 0), thickness=1)