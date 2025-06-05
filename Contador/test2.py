import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def detect_ridges(img: np.ndarray, sigma: int = 3) -> np.ndarray:
    img = cv2.equalizeHist(img.astype(np.uint8))

    elements = hessian_matrix(img, sigma, use_gaussian_derivatives=False)
    eigvals = hessian_matrix_eigvals(elements)

    cv2.normalize(eigvals[0], eigvals[0], 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return eigvals[0]

original_image = cv2.imread("zkN7m.png", cv2.IMREAD_GRAYSCALE)
ridges = detect_ridges(original_image)

thresholded = cv2.adaptiveThreshold((255-ridges).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

plt.imshow(thresholded, cmap="bone")