import lrspline as lr
import numpy as np
from scipy import linalg
from splipy import BSplineBasis
from numpy import pi

### SETUP INITIAL MESH PARAMETERS
p = np.array([3,3]) # quadratic functions (this is order, polynomial degree+1)
n = p + 3           # number of basis functions, start with 4x4 elements

### Target approximation function
def f(u,v):
    return np.sin(2*pi*u) * np.cos(2*pi*v)


### GENERATE THE MESH and initiate variables
surf    = lr.LRSplineSurface(n[0], n[1], p[0], p[1])

# establish bezier basis for quick evaluation
bezier1 = BSplineBasis(order=p[0], knots=[-1]*p[0]+[1]*p[0])
bezier2 = BSplineBasis(order=p[1], knots=[-1]*p[1]+[1]*p[1])
u,wu = np.polynomial.legendre.leggauss(p[0]+3)
v,wv = np.polynomial.legendre.leggauss(p[1]+3)
Iwu = np.diag(wu)
Iwv = np.diag(wv)

n = len(surf.basis)  # number of basis functions
M = np.zeros((n,n))  # mass matrix
b = np.zeros((n,1))  # right-hand-side

for el in surf.elements:
    # compute bezier basis functions 1D
    Nu = bezier1(u)
    Nv = bezier2(v)

    # assemble 1D functions to 2D
    Nb = np.kron(Nu,Nv)
    W  = np.kron(Iwu,Iwv)

    # fetch extraction operator and map bezier functions to LR functions
    C = el.bezier_extraction()
    N  = C @ Nb.T # index (i,j) is function(i) at point(j)

    # get evaluation points (in parameter space), and sample target function
    xi  = (u+1)/2 * np.diff(el.span('u')) + el.start('u')
    eta = (v+1)/2 * np.diff(el.span('v')) + el.start('v')
    xi,eta = np.meshgrid(xi,eta)
    F  = f(xi.flatten(), eta.flatten())
    F  = np.expand_dims(F,1)

    idx = np.array([bf.id for bf in el.support()]) # index of support functions
    b[idx] += (N @ W @ F) # Integrate over this element (sum over gauss points)
    i,j = np.meshgrid(idx,idx)
    M[i,j] += N @ W @ N.T

u = np.linalg.solve(M,b)
print(u)
