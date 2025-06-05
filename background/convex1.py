import cv2
import numpy as np
import matplotlib.pyplot as plt
#Read an Image

# Read the image
image1 = cv2.imread('images/original.png') 

# Display the image
plt.figure(figsize=[10,10])
plt.imshow(image1[:,:,::-1]);plt.title("Original Image");plt.axis("off");

# Make a copy of the original image so it is not overwritten
image1_copy = image1.copy()

# Convert to grayscale.
image_gray = cv2.cvtColor(image1_copy,cv2.COLOR_BGR2GRAY)

# Find all contours in the image.
contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Draw the selected contour
cv2.drawContours(image1_copy, contours, -1, (0,255,0), 3);

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(image1_copy[:,:,::-1]);plt.axis("off");plt.title('Contours Drawn');

image1_copy = image1.copy()

# Convert to grayscale
gray_image = cv2.cvtColor(image1_copy,cv2.COLOR_BGR2GRAY)

# Find all contours in the image.
contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Retreive the biggest contour
biggest_contour = max(contours, key = cv2.contourArea)

# Draw the biggest contour
cv2.drawContours(image1_copy, biggest_contour, -1, (0,255,0), 4);

# Display the results
plt.figure(figsize=[10,10])
plt.imshow(image1_copy[:,:,::-1]);plt.axis("off");

image1_copy = image1.copy()

# Convert to grayscale.
imageGray = cv2.cvtColor(image1_copy,cv2.COLOR_BGR2GRAY)

# Find all contours in the image.
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Sort the contours in decreasing order
sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)

# Draw largest 3 contours
for i, cont in enumerate(sorted_contours[:3],1):

    # Draw the contour
    cv2.drawContours(image1_copy, cont, -1, (0,255,0), 3)
    
    # Display the position of contour in sorted list
    cv2.putText(image1_copy, str(i), (cont[0,0,0], cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0),4)


# Display the result
plt.figure(figsize=[10,10])
plt.imshow(image1_copy[:,:,::-1]); 
plt.axis("off");
cv2.waitKey()