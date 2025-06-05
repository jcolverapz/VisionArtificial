import cv2
import numpy as np

# read input image
img = cv2.imread('images/blobs_connected.jpg')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold to binary
thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

# find contours
#label_img = img.copy()
contour_img = img.copy()
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
index = 1
isolated_count = 0
cluster_count = 0
for cntr in contours:
    area = cv2.contourArea(cntr)
    convex_hull = cv2.convexHull(cntr)
    convex_hull_area = cv2.contourArea(convex_hull)
    ratio = area / convex_hull_area
    #print(index, area, convex_hull_area, ratio)
    #x,y,w,h = cv2.boundingRect(cntr)
    #cv2.putText(label_img, str(index), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    if ratio < 0.91:
        # cluster contours in red
        cv2.drawContours(contour_img, [cntr], 0, (0,0,255), 2)
        cluster_count = cluster_count + 1
    else:
        # isolated contours in green
        cv2.drawContours(contour_img, [cntr], 0, (0,255,0), 2)
        isolated_count = isolated_count + 1
    index = index + 1
    
print('number_clusters:',cluster_count)
print('number_isolated:',isolated_count)

# save result
cv2.imwrite("blobs_connected_result.jpg", contour_img)

# show images
cv2.imshow("thresh", thresh)
#cv2.imshow("label_img", label_img)
cv2.imshow("contour_img", contour_img)
cv2.waitKey()