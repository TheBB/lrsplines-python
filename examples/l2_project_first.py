import lrspline as lr
import numpy as np
from scipy import linalg
from numpy import pi

### SETUP INITIAL MESH PARAMETERS
p = np.array([3,3]) # quadratic functions (this is order, polynomial degree+1)
n = p + 3           # number of basis functions, start with 4x4 elements

### Target approximation function
def f(u,v):
    return np.sin(2*pi*u) * np.cos(2*pi*v)


### GENERATE THE MESH and initiate variables
surf    = lr.LRSplineSurface(n[0], n[1], p[0], p[1])

u,wu = np.polynomial.legendre.leggauss(p[0]+3)
v,wv = np.polynomial.legendre.leggauss(p[1]+3)

n = len(surf.basis)  # number of basis functions
M = np.zeros((n,n))  # mass matrix
b = np.zeros((n,1))  # right-hand-side

for el in surf.elements: # for all elements
    xi  = (u+1)/2 * np.diff(el.span('u')) + el.start('u')
    eta = (v+1)/2 * np.diff(el.span('v')) + el.start('v')
    for i in range(len(u)): # for all gauss points
        for j in range(len(v)):
            for bi in el.support(): # for all functions with support
                for bj in el.support():
                    # assemble mass matrix
                    M[bi.id, bj.id] += bi(xi[i],eta[j]) * bj(xi[i],eta[j]) * wu[i] * wv[j]
                # assemble right-hand side
                b[bi.id] += bi(xi[i],eta[j]) * f(xi[i],eta[j]) * wu[i] * wv[j]

u = np.linalg.solve(M,b)
print(u)
