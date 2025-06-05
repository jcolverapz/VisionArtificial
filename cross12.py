import cv2
import numpy as np


fileName = "images/cruz.png" # Your "cross" image

# Reading an image in default mode:
Image = cv2.imread(fileName)

# Prepare a deep copy of the input for results:
inputImageCopy = Image.copy()

# Grayscale conversion:
grayscaleImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

# Add borders to prevent skeleton artifacts:
borderThickness = 1
borderColor = (0, 0, 0)
grayscaleImage = cv2.copyMakeBorder(grayscaleImage, borderThickness, borderThickness, borderThickness, borderThickness,
                                    cv2.BORDER_CONSTANT, None, borderColor)
# Compute the skeleton:
skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)

frame = np.zeros((240,320,3), np.uint8)


# A Python3 program to find if 2 given line segments intersect or not 

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def contourIntersect(original_image, contour1, contour2):
    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), [contours[0]], 0, 1)
    image2 = cv2.drawContours(blank.copy(), [contours[1]], 1, 1)
    
    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)
    
    # Check if there was a '1' in the intersection
    return intersection.any()
# def contour_intersect(cnt_ref,cnt_query):

#     ## Contour is a list of points
#     ## Connect each point to the following point to get a line
#     ## If any of the lines intersect, then break

#     for ref_idx in range(len(cnt_ref)-1):
#     ## Create reference line_ref with point AB
#         A = cnt_ref[ref_idx][0]
#         B = cnt_ref[ref_idx+1][0] 
    
#         for query_idx in range(len(cnt_query)-1):
#             ## Create query line_query with point CD
#             C = cnt_query[query_idx][0]
#             D = cnt_query[query_idx+1][0]
        
#             ## Check if line intersect
#             if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
#                 ## If true, break loop earlier
#                 return True

#     return False

class Point: 
	def __init__(self, x, y): 
		self.x = x 
		self.y = y 
p1 = Point(10,10) 
q1 = Point(100, 10)

p2 = Point(50, 1) 
q2 = Point(50, 100)  


cv2.circle(Image, (10, 10), 2, (255, 0, 255), -1)
#cv2.line(Image, (p1, q1), (p2, q2), (0,255,0), 2)
#cv2.line(Image,(Point(p1, q1)),Point(p2,q2),(255),1)
cv2.line(Image,(10,10),(100,10),(255),1)
#cv2.line(Image,(10,10),(50,10),(255),1)

# if doIntersect(p1, q1, p2, q2): 
# 	print("Yes") 
# else: 
# 	print("No") 
 
 

cv2.imshow('frame',Image ) 
cv2.imshow('frame',Image ) 
cv2.waitKey()


# This code is contributed by Ansh Riyal 
