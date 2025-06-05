import cv2
import numpy as np
from matplotlib import pyplot as plt
# Functions

def resizewithAspectRatio(img,width=None,height=None):
    return cv2.resize(img,(width,height),cv2.INTER_LINEAR)

#img=resizewithAspectRatio(cv2.imread("images/tela1.jpg"),640,640)
img=resizewithAspectRatio(cv2.imread("images/area4.jpg"),640,640)
gray_img=resizewithAspectRatio(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),640,640) 

empty_img=np.zeros((640,640),np.uint8)+255
kernel = np.ones((1,1),np.uint8)
kernel_size = (3,3)
#Apply Filter

gray_img=cv2.medianBlur(gray_img,3)
gray_img = cv2.bilateralFilter(gray_img,9,75,75)


threshold = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,83,3)


Reverse_img=np.where(threshold==255, 0, (np.where(threshold==0, 255, threshold)))


closing = cv2.morphologyEx(Reverse_img, cv2.MORPH_CLOSE, kernel,iterations=2)

edges = cv2.Canny(closing,50,150,apertureSize = 3)
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 1, 100)
    
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(empty_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 4, cv2.LINE_AA)

cv2.imshow("img",empty_img)

contours,hierarchy=cv2.findContours(empty_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

titles=['Org_Img','threshold','Reverse_img','closing','empty_img']
images=[img,threshold,Reverse_img,closing,empty_img]
for i in range(5):
    plt.subplot(3,3,i+1),plt.imshow(images[i] , 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()