import cv2
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib in line
from pathlib import Path
import math
import sys

img_unbearbeitet = cv2.imread('Bild2NIO.jpg',)
plt.imshow(img_unbearbeitet)

## Bild in ein Graustudenbild umwandeln

def rgb2gray(img_unbearbeitet):
    img_g=np.dot(img_unbearbeitet, [0.299, 0.578, 0.114])
            
    return img_g.astype(np.uint8)

img_g = rgb2gray(img_unbearbeitet)

plt.imshow(img_g)

## Gausfilter an Bild  

kernel = np.ones((5,5),np.float32)/25
img_gaus = cv2.filter2D(img_g,-1,kernel)
plt.imshow(img_gaus)
## Histogrammebnung

img_H = cv2.equalizeHist(img_gaus)
plt.imshow(img_H)
(t,imgbin) = cv2.threshold(img_gaus, 0, 
255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)    
crop_img1 =imgbin[150:500, 80:300]
plt.imshow(crop_img1)
## Konturen finden bei Treshold-Bild
cnts, _ = cv2.findContours(crop_img1, cv2.RETR_TREE,
cv2.CHAIN_APPROX_SIMPLE)
len(cnts)
out1= cv2.drawContours(cv2.merge( (crop_img1, crop_img1,  
crop_img1)) , cnts, -1, (0,0,255), 1)

## Canny-Operation mit Bild aus Histogrammebnung

img_canny1 = cv2.Canny(out1,250,150)

plt.subplot(121),plt.imshow(out1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_canny1,cmap = 'gray')
plt.title('Canny-Bild'), plt.xticks([]), plt.yticks([])
plt.show()