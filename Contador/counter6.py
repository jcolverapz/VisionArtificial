import cv2
import numpy as np

#img = cv2.imread('images/area4.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('images/screw.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('images/blobs.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img,(700,700))

img = cv2.subtract(255, img)

detector = cv2.SimpleBlobDetector_create()

 # Detect the blobs in the image
keypoints = detector.detect(img)
print(len(keypoints))

imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

blobs = 0
# for c in cnts:
#    #cv2.waitKey()
#     area = cv2.contourArea(c)
#     cv2.drawContours(img, [c], -1, (0,255,0), -1)
#    # (x, y, w, h) = cv2.boundingRect(c)
    #cv2.putText(img, str(blobs), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    
cv2.imshow("img", img)
cv2.imshow("Keypoints", imgKeyPoints)
cv2.waitKey()

cv2.destroyAllWindows()