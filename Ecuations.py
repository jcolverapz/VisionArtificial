from numpy import ones,vstack
from numpy.linalg import lstsq
import math

points = [(30, 220),(1385, 1050)]
x_coords, y_coords = zip(*points)
A = vstack([x_coords,ones(len(x_coords))]).T
m, c = lstsq(A, y_coords)[0]
print("Line Solution is y = {m}x + {c}".format(m=m,c=c))