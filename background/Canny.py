import cv2 

#img = cv2.imread("images/aerea1.jpg") # Read image 
img = cv2.imread("images/area4.jpg") # Read image 

# Defining all the parameters 
t_lower = 100 # Lower Threshold 
t_upper = 200 # Upper threshold 
aperture_size = 5 # Aperture size 
L2Gradient = True # Boolean 

# Applying the Canny Edge filter 
# with Aperture Size and L2Gradient 
edge = cv2.Canny(img, t_lower, t_upper, 
				apertureSize = aperture_size, 
				L2gradient = L2Gradient ) 

thresh = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY)[1] # threshold to binary
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    # if cv2.contourArea(cnt) > 500:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
    #     cv2.waitKey()

#cv2.drawContours(img, [contours], -1, (0,255,0), 1)
    cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
	# cv2.putText(frame, texto_estado , (10, 
#cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
#cv2.rectangle(frame,(10,10),(frame.shape[1],40),(0,0,0),-1)
#image_area = cv2.bitwise_and(gray, gray, mask=imAux)
#image_area = cv2.bitwise_and(fgmask, fgmask, mask=imAux)

#fgmask = fgbg.apply(image_area)
#fgmask = fgbg.apply(image_area)
#mask = object_detector.apply(frame)
cv2.imshow('original', img) 
cv2.imshow('edge', edge) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
