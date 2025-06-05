import cv2

# get image
img = cv2.imread('images/aerea1.jpg', cv2.IMREAD_GRAYSCALE)

# threshold to binary
ret, imgbin = cv2.threshold(img,5,255,cv2.THRESH_BINARY)

# morph 
dilateKernelSize = 80; erodeKernelSize = 65;
imgbin = cv2.dilate(imgbin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [dilateKernelSize,dilateKernelSize]))
imgbin = cv2.erode(imgbin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [erodeKernelSize,erodeKernelSize]))

# extract contours
contours, _ = cv2.findContours(imgbin,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print("Found ",len(contours),"contours")

# fit lines for large contours
lines = []; threshArea = 110;
for cnt in contours:
    if(cv2.contourArea(cnt)>threshArea):
        lines += [cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)] # [vx,vy,x,y]


# show results
imgresult = cv2.cvtColor(imgbin,cv2.COLOR_GRAY2RGB)
cv2.drawContours(imgresult, contours, -1, (255,125,0), 3)

VX_ = 0; VY_ = 1; X_ = 2; Y_ = 3;
rows,cols = imgbin.shape[:2]
p1 = [0,0]; p2 = [cols-1,0];
for l in lines:
    p1[1]  = int(((    0-l[X_])*l[VY_]/l[VX_]) + l[Y_])
    p2[1] = int(((cols-l[X_])*l[VY_]/l[VX_]) + l[Y_])
    cv2.line(imgresult,p1,p2,(0,255,255),2)

# save image    
#print(cv2.imwrite("<YourPathHere>", imgresult))

# HighGUI
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.namedWindow("imgresult", cv2.WINDOW_NORMAL)
cv2.imshow("imgresult", imgresult)
cv2.waitKey(0)
cv2.destroyAllWindows()