"""
Find the intersection points of lines.
"""

import numpy as np
import cv2
from collections import defaultdict
import sys
import math
def GetFieldLayer(src_img):
    hsv = cv2.cvtColor(src_img, code=cv2.COLOR_BGR2HSV)
    # green range
    lower_green = np.array([35, 10, 60])
    upper_green = np.array([65, 255, 255])
    # layer masks
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    player_mask = cv2.bitwise_not(field_mask)
    # player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # extract layers from original image
    field_layer = cv2.bitwise_and(src_img, src_img, mask=field_mask)
    return field_layer

img2 = cv2.imread("images/futbol.jpg")
img = GetFieldLayer(img2)
edges = cv2.Canny(img, 50, 200)
lines = cv2.HoughLines(edges, 0.5, math.radians(1.7), 150, None, 0, 1)
#cv2.imshow("Field Layer", edges)
#cv2.imshow("Fieer", img)
#cv2.imshow("Layer", field)
#cv2.imshow("Layeer", field2)



def segment_by_angle_kmeans(lines, k=2, **kwargs):


    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

    # Run k-means
    if sys.version_info[0] == 2:
        # python 2.x
        ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
    else: 
        # python 3.x, syntax has changed.
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))

    return segmented


def intersection(line1, line2):
    """
    Find the intersection of two lines 
    specified in Hesse normal form.

    Returns closest integer pixel locations.

    See here:
    https://stackoverflow.com/a/383527/5087436
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]


def segmented_intersections(lines):
    """
    Find the intersection between groups of lines.
    """

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections


def drawLines(img, lines, color=(0,0,255)):
    """
    Draw lines on an image
    """
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), color, 2)




# Detect lines
rho = 3
theta = np.pi/50
thresh = 400
lines = cv2.HoughLines(edges, rho, theta, thresh)

print("Found lines: %d" % (len(lines)))

# Draw all Hough lines in red
img_with_all_lines = np.copy(img2)
drawLines(img_with_all_lines, lines)
#cv2.imshow("Hough lines", img_with_all_lines)
#cv2.waitKey()

# Cluster line angles into 2 groups (vertical and horizontal)
segmented = segment_by_angle_kmeans(lines, 2)

# Find the intersections of each vertical line with each horizontal line
intersections = segmented_intersections(segmented)

img_with_segmented_lines = np.copy(img2)

# Draw vertical lines in green
vertical_lines = segmented[1]
img_with_vertical_lines = np.copy(img2)
drawLines(img_with_segmented_lines, vertical_lines, (255,255,0))

# Draw horizontal lines in yellow
horizontal_lines = segmented[0]
img_with_horizontal_lines = np.copy(img2)
drawLines(img_with_segmented_lines, horizontal_lines, (0,255,255))

# Draw intersection points in magenta
#print(intersections)

intersections.pop(5)
intersections.pop(4)
#print(intersections)

mn = intersections[3]
mk = intersections[2]
intersections = intersections[:2] 
#print(intersections)

intersections.append(mn)
intersections.append(mk)

print(intersections)
for point in intersections:
    pt = (point[0][0], point[0][1])
    length = 5
  
    cv2.circle(img_with_segmented_lines, pt, 5 , (255, 0, 255), -1)
    #cv2.line(img_with_segmented_lines, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 2) # vertical line
    #cv2.line(img_with_segmented_lines, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 2)


cv2.imshow("Segmented lines", img_with_segmented_lines)


cv2.waitKey()