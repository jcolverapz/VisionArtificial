import cv2
from numpy import *
Image = cv2.imread('images/cruz.png')


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
# #p1 = array( [0.0, 0.0] )
# p1 = array( [10, 100] )
# #p2 = array( [1.0, 0.0] )
# p2 = array( [100, 100] )

# #p3 = array( [4.0, -5.0] )
# p3 = array( [75, 50] )
# p4 = array( [75, 150] )

# Given these endpoints
#line 1
A = [10, 100]
B = [100, 100]

#line 2
C = [50, 50]
D = [50, 150]

# Compute this:
#point_of_intersection = [X, Y]

print (line_intersection((A, B), (C, D)))


cv2.line(Image,(A[0], A[1]),(B[0], B[1]), (255,0,0), 2)
cv2.line(Image,(C[0], C[1]),(D[0], D[1]), (255,0,0), 2)
#cv2.line(Image,(p3[0], p3[1]),(p4[0], p4[1]), (255,0,0), 2)


#print (seg_intersect( p1,p2, p3,p4))

#for index in len(seg_intersect( p1,p2, p3,p4)):
for p in line_intersection((A, B), (C, D)):
    #cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=-1)
    #cv2.circle(Image, Point(p(1)), 2, (255, 0, 255), 1)
    #cv2.circle(Image, (int(p[0][0]),int(p[0][1])), 2, (255, 0, 255), -1)
    print((p(1)))