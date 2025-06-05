import cv2
import numpy as np
#from matplotlib import pyplot
import matplotlib.pyplot as plt 
# Load image, bilaterial blur, and Otsu's threshold
#image = cv2.imread('images/gaps.png')
#image = cv2.imread('images/aerea1.jpg')
image = cv2.imread('images/area4.jpg')
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #escala de grises
blur = cv2.bilateralFilter(gray,9,75,75)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  #imagen binaria

blank_mask = np.zeros(image.shape, np.uint8)
linea_mask = np.zeros(image.shape, np.uint8)
cv2.line(blank_mask, (150, 0), (150, 700), (255,255,255), 1)	

#image_area = cv2.bitwise_and(image, blank_mask)

# Perform morpholgical operations
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
##opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find distorted rectangle contour and draw onto a mask
#cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# rect = cv2.minAreaRect(cnts[0])
# box = cv2.boxPoints(rect)
# box = np.int_(box)
#cv2.drawContours(image,[box],0,(36,255,12),4)
#cv2.fillPoly(mask, [box], (255,255,255))

# Find corners
# Detect corners using the contours
corners = cv2.goodFeaturesToTrack(image=thresh, maxCorners=1000,qualityLevel=0.1, minDistance=10) # Determines strong corners on an image
#cv2.drawContours(image, [frame], -1, 255, 2)
#cv2.imshow('blank_mask',blank_mask )
# Draw the corners on the original image
for index in range(len(corners)):
    corner = corners[index]
    x,y = corner.ravel()
    cv2.circle(image,(int(x),int(y)),3,(0,0,255),-1)
    #cv2.line(lineas, (int(x), int(y)), (int(x)+50, int(y)), (255,255,255), 1)	
    #cv2.putText(image, str(index), (int(x),int(y)),1, 1,(0,0,255), 2)
#mask = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(close,4,.8,100)
# offset = 25
# for corner in corners:
#     x,y = corner.ravel()
#     cv2.circle(image,(int(x),int(y)),5,(36,255,12),-1)
#     x, y = int(x), int(y)
#     cv2.rectangle(image, (x - offset, y - offset), (x + offset, y + offset), (36,255,12), 3)
#     print("({}, {})".format(x,y))
#image_area = cv2.bitwise_and(lineas, blank_mask)
intersections = [] 
#contours1, _ = cv2.findContours(image_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# for cnt in contours1: 
#    #area = cv2.contourArea(cnt1) 
#    #Draw it 
#    cv2.drawContours(image,[cnt],0,(0,0,255),1)
contours_idx = blank_mask[...,1] == 255
lines_idx = blank_mask[...,0] == 255

#contours_idx = np.all(blank_mask == (0, 255, 0), axis=-1)
#lines_idx = np.all(blank_mask == (255, 0, 0), axis=-1)

overlap = np.where(contours_idx * lines_idx)
## threshold
th, threshed = cv2.threshold(blank_mask, 0, 255,cv2.THRESH_BINARY)
#img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
#n_white_pix = np.sum(threshed == 255)
#white_pixels = np.array(np.where(threshed == 255))
#bin = cv2.erode(bin, None)
  
dots=np.zeros_like(threshed)
#graydots= cv2.cvtColor(dots, cv2.COLOR_BGR2HSV)
#graydots = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #escala de grises
# lower_hsv = np.array([112, 176, 174])
# higher_hsv = np.array([179,210,215])
#mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
#mask = cv2.inRange(dots,)
contours, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#mnts  = [cv2.moments(cnt) for cnt in cnts]
for cnt in contours:
    #centroids = [( int(round(m['m10']/m['m00'])),int(round(m['m01']/m['m00'])) ) for m in mnts]
    m = cv2.moments(cnt)
    if m["m00"] != 0:
        cX = int(m["m10"] /  m["m00"]) 
        cY = int(m["m01"] / m["m00"])
        cv2.circle(image, (int(cX), int(cY)), 3, (0, 0, 255), -1)

# for c in centroids:
#     cv2.circle(dots,c,5,(0,255,0))
#     print (c)

cv2.imshow('red_dots', dots)

cv2.waitKey(0)
cv2.destroyAllWindows()
# contours, _ = cv2.findContours(image_area, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# rc = cv2.minAreaRect(contours[0])
# box = cv2.boxPoints(rc)
# for p in box:
#     pt = (p[0],p[1])
#     #print pt
#     cv2.circle(image, pt ,5,(200,0,0),2)
    
    #first_white_pixel = white_pixels[:,0]
#last_white_pixel = white_pixels[:,-1]
# for p in threshed: 
# #     #image[p]==[0,0,255]
# #     print(p[0])
#     #cv2.circle(image, ([p][0], p[1]), 2,(0,255,0),-1)
# #image[100,100]=[0,0,255]


# #cv2.circle(image, (x, y), radius, (B,G,R), thickness)
#     cv2.circle(image, (p[0], p[1]), 5, (0,0,255), 2)


# whitepx=[]
# for p in threshed: 
#     if p<>255:
#         whitepx.append()
# #cv2.countNonZero(binary_img)
#print('Number of white pixels:', n_white_pix)
## fi
#print(overlap)
# ## findcontours
# cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(cnts))
# #for cnt in cnts:
# #     x,y,w,h = cv2.boundingRect(cnt)	
    #if p==255:
        #cv2.circle(image, [p], 2,(0,255,0),-1)
   # print(p)
#     #cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
    #if s1<cv2.contourArea(cnt) <s2:
    #xcnts.append(cnt)
    #cv2.drawContours(image, p, -1, (0, 255, 0), 1)
#image[p]=[0,0,255]
#print("Dots number: {}".format(len(xcnts)))
   #hull = cv2.convexHull(cnt1)
   #cv2.drawContours(frame,hull,3,(255,0,0),2)   
#cv2.imshow('image', image)


cv2.imshow('threshed', threshed)
#cv2.imshow('lineas', lineas)
#cv2.imshow('image_area', image_area)
cv2.imshow('thresh', thresh)
cv2.imshow('blank_mask', blank_mask)
#cv2.imshow('close', close)
#cv2.imshow('mask', mask)
plt.imshow(image)
plt.show()

cv2.waitKey()

# # Draw Contours
# blank_mask = np.zeros((thresh.shape[0],thresh.shape[1],3), np.uint8)
# cv2.drawContours(blank_mask, contours, -1, (0, 255, 0), 1)
# contours_idx = blank_mask[...,1] == 255

# # Define lines coordinates
# line1 = [x1, y1, x2, y2]
# line2 = [x1, y1, x2, y2]
# line3 = [x1, y1, x2, y2]

# # Draw Lines over Contours
# cv2.line(blank_mask, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 0), thickness=1)
# cv2.line(blank_mask, (line2[0], line2[1]), (line2[2], line2[3]), (255, 0, 0), thickness=1)
# cv2.line(blank_mask, (line3[0], line3[1]), (line3[2], line3[3]), (255, 0, 0), thickness=1)
# lines_idx = blank_mask[...,0] == 255
# overlap = np.where(contours_idx * lines_idx)
# (array([ 90, 110, 110, 140, 140], dtype=int64), array([ 80,  40, 141,  27, 156], dtype=int64))
# list(zip(*overlap))
# # Change these
# contours_idx = blank_mask[...,1] == 255
# lines_idx = blank_mask[...,0] == 255

# # To this
# contours_idx = np.all(blank_mask == (0, 255, 0), axis=-1)
# lines_idx = np.all(blank_mask == (255, 0, 0), axis=-1)