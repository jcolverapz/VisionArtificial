import cv2
import numpy as np

img = cv2.imread('images/dave.png')

winname="raw image"
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.moveWindow(winname, 100,100)


img = cv2.GaussianBlur(img,(5,5),0)

winname="blurred"
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.moveWindow(winname, 100,150)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

winname="gray"
cv2.namedWindow(winname)
cv2.imshow(winname, gray)
cv2.moveWindow(winname, 100,200)

close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

winname="res2"
cv2.namedWindow(winname)
cv2.imshow(winname, res2)
cv2.moveWindow(winname, 100,250)

 #find elements
thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for cnt in contour:
    area = cv2.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt

cv2.drawContours(mask,[best_cnt],0,255,-1)
cv2.drawContours(mask,[best_cnt],0,0,2)

res = cv2.bitwise_and(res,mask)

winname="puzzle only"
cv2.namedWindow(winname)
cv2.imshow(winname, res)
cv2.moveWindow(winname, 100,300)

# vertical lines
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

dx = cv2.Sobel(res,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

winname="vertical lines"
cv2.namedWindow(winname)
#cv2.imshow(winname, img_d)
cv2.imshow(winname, thresh)
cv2.moveWindow(winname, 100,350)

# find horizontal lines
kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(res,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()

winname="horizontal lines"
cv2.namedWindow(winname)
cv2.imshow(winname, close)
cv2.moveWindow(winname, 100,400)


# intersection of these two gives dots
res = cv2.bitwise_and(closex,closey)

winname="intersections"
cv2.namedWindow(winname)
cv2.imshow(winname, res)
cv2.moveWindow(winname, 100,450)

# text blue
textcolor=(0,255,0)
# points green
pointcolor=(255,0,0)

# find centroids and sort
contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(img,(x,y),4,(0,255,0),-1)
    centroids.append((x,y))

# sorting
centroids = np.array(centroids,dtype = np.float32)
c = centroids.reshape((100,2))
c2 = c[np.argsort(c[:,1])]

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
bm = b.reshape((10,10,2))

# make copy
labeled_in_order=res2.copy()

for index, pt in enumerate(b):
    cv2.putText(labeled_in_order,str(index),tuple(pt),cv2.FONT_HERSHEY_DUPLEX, 0.75, textcolor)
    cv2.circle(labeled_in_order, tuple(pt), 5, pointcolor)

winname="labeled in order"
cv2.namedWindow(winname)
cv2.imshow(winname, labeled_in_order)
cv2.moveWindow(winname, 100,500)

# create final

output = np.zeros((450,450,3),np.uint8)
for i,j in enumerate(b):
    ri = int(i/10) # row index
    ci = i%10 # column index
    if ci != 9 and ri!=9:
        src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
        dst = np.array( [ [ci*50,ri*50],[(ci+1)*50-1,ri*50],[ci*50,(ri+1)*50-1],[(ci+1)*50-1,(ri+1)*50-1] ], np.float32)
        retval = cv2.getPerspectiveTransform(src,dst)
        warp = cv2.warpPerspective(res2,retval,(450,450))
        output[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1].copy()

winname="final"
cv2.namedWindow(winname)
cv2.imshow(winname, output)
cv2.moveWindow(winname, 600,100)

cv2.waitKey(0)
cv2.destroyAllWindows()