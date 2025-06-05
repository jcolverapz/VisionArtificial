import cv2
import os
import numpy as np

#path = os.path("D:\")
img = cv2.imread('images/area4.jpg')

#cv2.imshow('image',img)
#cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(img)

'''height, width, channel = img.shape[:3]
size = img.size

print(height, width, channel, size)'''


if img is not None:
    lines = cv2.HoughLines(img, 1, np.pi / 180, 100)


    blank_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for x in range(0, len(lines)):    
        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(blank_image, (x1, y1), (x2, y2), (255), 1)
            #print(x1, y1, x2, y2)

        print("\n")

    cv2.imshow("img", img)
    cv2.imshow("out", blank_image)
    cv2.waitKey()
else:
    print("empty img. Cannot read file")