import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('images/cnetral.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edged = cv2.Canny(gray, 30, 100)
#Un método de agrupación para agrupar sus Me gusta :contoursKMeans
#from sklearn.cluster import KMeans

# Get the list of points
points = np.argwhere(edged)

# Create 4 clusters of points
#kmeans = KMeans(n_clusters=4).fit(points)
#Ahora, encuentra la pendiente y los coeficientes de intersección de con (o ):mby = mx + bLinearRegressionnp.polyfit
#from sklearn.linear_model import LinearRegression

# Find coefficients
coeffs = []
for i in range(4):
    idx = np.where(i == kmeans.labels_)[0]
    x = points[idx, 0].reshape(-1, 1)
    y = points[idx, 1]

    reg = LinearRegression().fit(x, y)
    m, b = reg.coef_[0], reg.intercept_  
    coeffs.append((m, b))

    plt.scatter(x, y, s=0.1)
    plt.axline(xy1=(0, b), slope=m, color='k', zorder=-1, lw=0.5)

plt.xlim(0, image.shape[0])
plt.ylim(0, image.shape[1])
plt.show()