import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read image
img = cv2.imread('images/area4.jpg', 0)

# #------------------------
# # Morphology
# #========================
# # Closing
# #------------------------
#closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))

# #------------------------
# # Statistics
# #========================
dens = np.sum(img, axis=0)
mean = np.mean(dens)

#------------------------
# Thresholding
#========================
thresh = 255 * np.ones_like(img)
k = 0.9
for idx, val in enumerate(dens):
    if val< k*mean:
        thresh[:,idx] = 0

thresh = 255 - thresh
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
count = len(contours)


#------------------------
# plotting the results
#========================
plt.figure(num='{} Lines'.format(count))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(223)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded')
plt.axis('off')

plt.subplot(224)
plt.imshow((thresh/255)*img, cmap='gray')
plt.title('Result')
plt.axis('off')

plt.subplot(222)
plt.hist(dens)
plt.axvline(dens.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('dens hist')

plt.show()
