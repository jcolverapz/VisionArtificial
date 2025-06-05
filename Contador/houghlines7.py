import numpy as np
import sys
import cv2

function findlineshv(I)

#% Read Image
img = cv2.imread(I);

# % Convert to black and white because
# % edge function only works with BW imgs
bwImage = cv2.rgb2gray(img);

#% figure(1),imshow(bwImage);

#% find edges using edge function
b=cv2.edge(bwImage,'sobel');

# % show edges
# % figure(1),imshow(b);


# % compute the Hough transform of the edges found
# % by the edge function
[hou,theta,rho] = cv2.hough(b);

#% define peaks, x and y
peaks = cv2.houghpeaks(hou,5,'threshold',ceil(0.3*max(hou(:))));

x = theta(peaks(:,2));
y = rho(peaks(:,1));


lines = cv2.houghlines(bwImage,theta,rho,peaks,'FillGap',5,'MinLength',7);

figure, imshow(bwImage), hold on

for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',3,'Color','red');
end