import numpy as np
import cv2

# Load the picture
img_path = 'path'
img = cv2.imread('images/cruz.png')

# Extract contours
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Apply Canny edge detection
edges = cv2.Canny(thresh, 100, 200)

# Find contour points and assign them to pixel rows
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rows, cols = img.shape[:2]
pixel_counts = [0] * cols
for cnt in contours:
    for point in cnt:
        row, col = point[0]
        pixel_counts[col] += 1

# Find pixel row with the most contour points
max_count = max(pixel_counts)
max_col = pixel_counts.index(max_count) - 50

# Draw line
cv2.line(img, (0, max_col), (cols-1, max_col), (0, 0, 0), 2)

# Threshold the image a second time to create a binary image better suited for contour
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Apply Canny edge detection a second time
edges = cv2.Canny(thresh, 100, 200)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area (which is assumed to be the bubble)
cnt = max(contours, key=cv2.contourArea)

# Fit a polynom to the largest contour
epsilon = 0.0001 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# Draw the polynom fit in yellow
cv2.polylines(img, [approx], True, (0, 255, 255), 2)

# Convert line points to numpy array
line_points = np.array([(0, max_col), (cols-1, max_col)])
print(line_points)
# Find intersection of line and largest contour
if len(cnt) > 0:
    intersection_points = []
    for i in range(len(cnt)):
        x, y = cnt[i][0]
        x = int(x)
        y = int(y)
        dist = cv2.pointPolygonTest(line_points, (x, y), True)

        if dist == 0:
            intersection_points.append((x, y))
        

else:
    print("Error: No contour points found")
    exit()

print("longitud: " + str(len(intersection_points)))
# Draw intersection points in green
for point in intersection_points:
    cv2.circle(img, point, 4, (0, 255, 0), thickness=-1)
    

#print(contourIntersect(original_image, contour_list[0], contour_list[1]))
cv2.imshow("img", img)
cv2.imshow("thresh", thresh)
cv2.waitKey()
