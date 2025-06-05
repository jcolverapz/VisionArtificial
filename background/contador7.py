import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread('images/area4.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Apply median blur
img = cv2.medianBlur(img, 5)
cv2.imshow('Image', img)

# Initialize lists to store counts
cntsUpper = []
cntsLower = []

# Loop through columns
for i in range(img.shape[1]):
    cntUpper = 0
    cntLower = 0
    for j in range(img.shape[0]):
        pixel_value = img[j, i]
        if 15 < pixel_value < 250:
            cntUpper += 1
        elif pixel_value < 15:
            cntLower += 1

    if cntUpper != 0:
        cntsUpper.append(cntUpper)
    if cntLower != 0:
        cntsLower.append(cntLower)

# Calculate the average for lower counts
if cntsLower:
    averageLower = sum(cntsLower) / len(cntsLower)
    print(f"Average Lower: {averageLower}")

# Calculate the average for upper counts
if cntsUpper:
    averageUpper = sum(cntsUpper) / len(cntsUpper)
    print(f"Average Upper: {averageUpper}")

cv2.waitKey()
cv2.destroyAllWindows()