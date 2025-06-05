import cv2
import numpy as np

gray = cv2.imread('images/test9.png')
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

vlines = []
hlines = []
final_lines = []

a,b,c = lines.shape
for i in range(a):
    if abs(lines[i][0][1] - lines[i][0][3]) < b / 100:
        # print("horizental")
        hlines.append(lines[i][0].tolist())
    else:
        # print("vertical")
        vlines.append(lines[i][0].tolist())

def ccw(p1,p2,p3):
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

# Return true if line segments l1 and l2 intersect
def intersect(l1, l2):
    return ccw((l1[0], l1[1]), (l2[0], l2[1]),(l2[2], l2[3])) != ccw((l1[2], l1[3]),(l2[0], l2[1]), (l2[2], l2[3])) and ccw((l1[0], l1[1]), (l1[2], l1[3]), (l2[0], l2[1])) != ccw((l1[0], l1[1]), (l1[2], l1[3]), (l2[2], l2[3]))

# check intersect of each vertical lines with horizenal lines
for vl in vlines:
    for hl in hlines:
        if intersect(vl, hl):
            final_lines.append(vl)
            break

print(final_lines)
for line in final_lines:
    cv2.line(gray, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('vertical line',gray)
    cv2.waitKey()