#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
import cv2
from numpy import *

Image = cv2.imread('images/cruz.png')

class Point: 
	def __init__(self, x, y): 
		self.x = x 
		self.y = y 

def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

#p1 = array( [0.0, 0.0] )
p1 = array( [10, 100] )
#p2 = array( [1.0, 0.0] )
p2 = array( [100, 100] )

#p3 = array( [4.0, -5.0] )
p3 = array( [75, 50] )
p4 = array( [75, 150] )

#frame = numpy.zeros((240,320,3), numpy.uint8)

cv2.line(Image,(p1[0], p1[1]),(p2[0], p2[1]), (255,0,0), 2)
cv2.line(Image,(p3[0], p3[1]),(p4[0], p4[1]), (255,0,0), 2)


print (seg_intersect( p1,p2, p3,p4))

#for index in len(seg_intersect( p1,p2, p3,p4)):
for p in seg_intersect( p1,p2, p3,p4):
    #cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=-1)
    #cv2.circle(Image, Point(p(1)), 2, (255, 0, 255), 1)
    #cv2.circle(Image, (int(p[0][0]),int(p[0][1])), 2, (255, 0, 255), -1)
    print(int(p(1)))
# p1 = array( [2.0, 2.0] )
# p2 = array( [4.0, 3.0] )

# p3 = array( [6.0, 0.0] )
# p4 = array( [6.0, 3.0] )

# print (seg_intersect( p1,p2, p3,p4))

cv2.imshow('img', Image)
cv2.waitKey()