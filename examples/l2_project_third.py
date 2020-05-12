import lrspline as lr
import numpy as np
from scipy import linalg
from splipy import BSplineBasis
from numpy import pi

### SETUP INITIAL MESH PARAMETERS
p = np.array([3,3]) # quadratic functions (this is order, polynomial degree+1)
n = p + 3           # number of basis functions, start with 4x4 elements
n_refinements = 6   # number of edge refinements

### Target approximation function
def f(u,v):
    eps = 1e-4
    d1 = np.power(u-.2,2) + np.power(v-.2,2)
    d2 = np.power(u-.9,2) + np.power(v-.5,2)
    d3 = np.power(u-.1,2) + np.power(v-.8,2)
    return 1/(d1+eps) + 1/(d2+eps) + 1/(d3+eps)


### GENERATE THE MESH and initiate variables
surf    = lr.LRSplineSurface(n[0], n[1], p[0], p[1])

bezier1 = BSplineBasis(order=p[0], knots=[-1]*p[0]+[1]*p[0])
bezier2 = BSplineBasis(order=p[1], knots=[-1]*p[1]+[1]*p[1])
u,wu = np.polynomial.legendre.leggauss(p[0]+3)
v,wv = np.polynomial.legendre.leggauss(p[1]+3)
Iwu = np.diag(wu)
Iwv = np.diag(wv)

for i_ref in range(n_refinements):
    print('Assembling mass matrix')
    n = len(surf.basis)
    M = np.zeros((n,n))  # mass matrix
    b = np.zeros((n,1))  # right-hand-side

    for el in surf.elements:
        xi  = (u+1)/2 * np.diff(el.span('u')) + el.start('u')
        eta = (v+1)/2 * np.diff(el.span('v')) + el.start('v')

        C = el.bezier_extraction()
        i = np.array([bf.id for bf in el.support()])
        Nu = bezier1(u)
        Nv = bezier2(v)
        Nb = np.kron(Nu,Nv)
        W  = np.kron(Iwu,Iwv)
        N  = C @ Nb.T
        xi,eta = np.meshgrid(xi,eta)
        F  = f(xi.flatten(), eta.flatten())
        F  = np.expand_dims(F,1)

        b[i] += (N @ W @ F)
        i,j = np.meshgrid(i,i)
        M[i,j] += N @ W @ N.T

    print('Solving system')
    sol = np.linalg.solve(M,b)

    # compute errors
    print('Computing errors')
    element_error = np.zeros(len(surf.elements))
    for el in surf.elements:
        C = el.bezier_extraction()
        i = np.array([bf.id for bf in el.support()])
        Nu = bezier1(u)
        Nv = bezier2(v)
        Nb = np.kron(Nu,Nv)
        W  = np.kron(Iwu,Iwv)
        N  = C @ Nb.T
        xi  = (u+1)/2 * np.diff(el.span('u')) + el.start('u')
        eta = (v+1)/2 * np.diff(el.span('v')) + el.start('v')
        xi,eta = np.meshgrid(xi,eta)
        F  = np.expand_dims(f(xi.flatten(), eta.flatten()),1)

        error = F - N.T @ sol[i]
        element_error[el.id] = np.sum(np.power(error,2))
    function_error = [np.sum(element_error[[el.id for el in bf.support()]]) for bf in surf.basis]

    print('Refining mesh')
    idx = np.argsort(function_error)
    n = len(surf.basis) // 5
    refine_functions = [bf for bf in surf.basis if bf.id in idx[-n:]]
    surf.refine(refine_functions)
    print(f'New mesh:')
    print(f'  # basis    = {len(surf.basis)}')
    print(f'  # elements = {len(surf.elements)}')


with open('mesh.eps','wb') as myfile:
    surf.write_postscript(myfile)
