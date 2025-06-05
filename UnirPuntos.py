# import the necessary packages
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
import imutils
#from imutils import contours
import numpy as np

# Method to find the mid point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

 
cap = cv2.VideoCapture('videos/vidrio51.mp4') 
object_detector = cv2.bgsegm.createBackgroundSubtractorMOG()
 
def nothing(pos):
    pass

cv2.namedWindow('Thresholds')
cv2.createTrackbar('lc','Thresholds',255,255, nothing)
cv2.createTrackbar('hc','Thresholds',255,255, nothing)


while(True): 
    ret, frame = cap.read() 
    frame = imutils.resize (frame, width=720)
    # load the image, convert it to grayscale, and blur it slightly
    # area = cv2.getTrackbarPos('area','Thresholds')
    lc = cv2.getTrackbarPos('lc','Thresholds')
    hc = cv2.getTrackbarPos('hc','Thresholds')
    #image = cv2.imread("images/test2.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, lc, hc)
    edged = cv2.dilate(edged, None, iterations=4)
    edged = cv2.erode(edged, None, iterations=2)

    # find contours in the edge map
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    #print(len(cnts))
    # loop over the contours individually
    orig = frame.copy() 
    for c in cnts:
        # This is to ignore that small hair countour which is not big enough
        #if cv2.contourArea(c) < 5000:
        if cv2.contourArea(c) > 1000:
            cv2.waitKey()
            #if cv2.contourArea(c) > 10:
            #continue

        # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            # draw the contours on the image
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 3)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # draw and write the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.putText(orig, "({},{})".format(tltrX, tltrY), (int(tltrX - 50), int(tltrY - 10) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.putText(orig, "({},{})".format(blbrX, blbrY), (int(blbrX - 50), int(blbrY - 10) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.putText(orig, "({},{})".format(tlblX, tlblY), (int(tlblX - 50), int(tlblY - 10) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv2.putText(orig, "({},{})".format(trbrX, trbrY), (int(trbrX     - 50), int(trbrY - 10) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # loop over the original points
            for (xA, yA) in list(box):
                # draw circles corresponding to the current points and
                cv2.circle(orig, (int(xA), int(yA)), 5, (0,0,255), -1)
                cv2.putText(orig, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.imshow("gray", gray) 
    cv2.imshow("edged", edged) 
    cv2.imshow("Image", orig) 
    k = cv2.waitKey(100) & 0xff
    if k == 27: 
        break
cap.release() 
cv2.destroyAllWindows()# show the output image, resize it as per your requirements
