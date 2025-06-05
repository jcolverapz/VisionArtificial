import sys
import cv2
import numpy as np

# Load and resize image
im = cv2.imread(sys.path[0]+'/im.png')
H, W = im.shape[:2]
S = 4
im = cv2.resize(im, (W*S, H*S))

# Convert to binary
msk = im.copy()
msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
msk = cv2.threshold(msk, 200, 255, cv2.THRESH_BINARY)[1]

# Glue char blobs together
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 5))
msk = cv2.medianBlur(msk, 3)
msk = cv2.erode(msk, kernel1)
msk = cv2.erode(msk, kernel2)

# Skeletonization-like operation in OpenCV
thinned = cv2.ximgproc.thinning(~msk)

# Make final chars
msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
thinned = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
thicked = cv2.erode(~thinned, np.ones((9, 15)))
thicked = cv2.medianBlur(thicked, 11)

# Save the output
top = np.hstack((im, ~msk))
btm = np.hstack((thinned, thicked))
cv2.imwrite(sys.path[0]+'/im_out.png', np.vstack((top, btm)))