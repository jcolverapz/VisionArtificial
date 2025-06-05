import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

def intercepting_line() :
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

#fake data

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10.0 * (Z2 - Z1)

#plot
cs = plt.contour(X,Y,Z)
whichContour = 2 # change this to find the right contour lines

#get the vertices to calculate an intercept with a line
p = cs.collections[whichContour].get_paths()[0]
#see: http://matplotlib.org/api/path_api.html#module-matplotlib.path
v = p.vertices
xx = v[:, 0]
yy = v[:, 1]

#this shows the innermost ring now
plt.plot(xx, yy, 'r--', label='inner ring')

#fake line
x = np.arange(-2, 3.0, 0.1)
y=lambda x,m:(m*x)
y=y(x,0.9)
lineMesh = np.meshgrid(x,y)
plt.plot(x,y,'r' ,label='line')

#get the intercepts, two in this case 
x, y = find_intersections(v, lineMesh[1])
print x
print y
#plot the intercepting points
plt.plot(x[0], y[0], 'bo', label='first intercept')
#plt.plot(x[1], y[1], 'rs', label='second intercept')
plt.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
plt.show()

#now we need to calculate the intercept of the vertices and whatever line
#this is pseudo code but works in case of two intercepting contour vertices

def find_intersections(A, B):
# min, max and all for arrays
amin = lambda x1, x2: np.where(x1<x2, x1, x2)
amax = lambda x1, x2: np.where(x1>x2, x1, x2)
aall = lambda abools: np.dstack(abools).all(axis=2)
slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))

x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])
m1, m2 = np.meshgrid(slope(A), slope(B))
m1inv, m2inv = 1/m1, 1/m2

yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
xi = (yi - y21)*m2inv + x21

xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
          amin(x21, x22) < xi, xi <= amax(x21, x22) )
yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
          amin(y21, y22) < yi, yi <= amax(y21, y22) )

return xi[aall(xconds)], yi[aall(yconds)]