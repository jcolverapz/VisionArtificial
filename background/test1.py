 # Params for corner detection
    feature_params = dict(maxCorners=20,  # We want only one feature
                        qualityLevel=0.2,  # Quality threshold 
                        minDistance=7,  # Max distance between corners, not important in this case because we only use 1 corner
                        blockSize=7)

    first_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Harris Corner detection
    points = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)


     # Create corners and draw on image
        corners = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        drawn = cv.drawChessboardCorners(img, (width, height), corners, ret)
        
        
        # generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
# convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)