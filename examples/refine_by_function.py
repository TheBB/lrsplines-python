from splipy.io import G2
from splipy import curve_factory
from splipy import surface_factory
import lrspline as lr
import sys
import numpy as np
import inspect
import matplotlib.pyplot as plt

def set_max_meshsize(myspline, amount):

    # build up the list of arguments (in case not all of (x,y,z,r,u,v,w) are specified)
    arg_names = inspect.signature(amount).parameters
    argc = len(arg_names)

    use_physical_meshsize = False
    if 'x' in arg_names or 'y' in arg_names or 'z' in arg_names or 'r' in arg_names:
        use_physical_meshsize = True

    # use the convencience function with all arugments present
    def h_target(x,y,z,r,u,v,w):
        kwarg = {'x':x, 'y':y, 'z':z, 'r':r, 'u':u, 'v':v, 'w':w}
        argv = {}
        for name in arg_names:
            argv[name] = kwarg[name]
        return amount(**argv)

    did_refinement = True
    while did_refinement:
        did_refinement = False
        functions_to_refine = set()
        for el in myspline.elements:
            print(el)
            u0 = np.array(el.start())
            u1 = np.array(el.end())
            u = (u0+u1)/2
            x = myspline(*u, iel=el.id)
            r = np.linalg.norm(u)
            if len(x) == 2: x = np.append(x,0)
            if len(u) == 2: u = np.append(u,0)
            # look up the target mesh size for this element as measured in either
            # geometric coordinates (x,y,z) or parametric (u,v,w)
            if use_physical_meshsize:
                if myspline.pardim == 2:
                    # pick the largest diagonal
                    h0 = np.linalg.norm(myspline(*u1) - myspline(*u0))
                    h1 = np.linalg.norm(myspline(u0[0], u1[1]) - myspline(u1[0], u0[1]))
                    h = max(h0,h1)
                else:
                    # pick the largest diagonal
                    h0 = np.linalg.norm(myspline(u0[0], u0[1], u0[2]) - myspline(u1[0], u1[1], u1[2]))
                    h1 = np.linalg.norm(myspline(u1[0], u0[1], u0[2]) - myspline(u0[0], u1[1], u1[2]))
                    h2 = np.linalg.norm(myspline(u0[0], u1[1], u0[2]) - myspline(u1[0], u0[1], u1[2]))
                    h3 = np.linalg.norm(myspline(u0[0], u0[1], u1[2]) - myspline(u1[0], u1[1], u0[2]))
                    h = np.max([h0,h1,h2,h3])
            else:
                h = np.max(u1-u0)

            print(f'Evaluation point: x={x},  u={u})')
            print(f'Actual size: {h}')
            print(f'Target size: {h_target(*x,r,*u)}')
            if h > h_target(*x, r, *u):
                did_refinement = True
                for function in el.support():
                    functions_to_refine.add(function.id)

        # make a list of the basisfunction objects tagged for refinement
        functions_to_refine = [f for f in myspline.basis if f.id in functions_to_refine]

        if did_refinement:
            print('Trying to refine the following functions')
            print([f.id for f in functions_to_refine])
            myspline.refine(functions_to_refine)



### USER DEFINED REFINEMENT FUNCTION
def ref(x,y):
    return np.abs(y-x)+0.025


# read model from file
with G2(sys.argv[1]) as myfile:
   mymodel = myfile.read()


### make an example with controlpoint values
# crv1 = curve_factory.circle_segment(theta=np.pi/2, r=1).rebuild(p=3,n=6)
# crv2 = curve_factory.polygon([2,0], [2,2], [0,2])
# srf = surface_factory.edge_curves(crv1, crv2)
# srf.raise_order(0,1)
# srf.refine(0,3)
#
# mymodel = [srf]
# # write model to file
# with G2('plate_with_hole.g2') as myfile:
#    myfile.write(srf)


# convert all patches to LRSpline representations
lr_model = []
for spline in mymodel:
    cp = spline.controlpoints
    cp = np.reshape(cp, (np.prod(cp.shape[:-1]),cp.shape[-1]), order='F')
    print(cp)
    if spline.pardim == 2:
        lr_model.append(lr.LRSplineSurface(spline.shape[0], spline.shape[1], spline.order(0), spline.order(1), spline.knots(0, True), spline.knots(1, True), cp))
    elif spline.pardim == 3:
        lr_model.append(lr.LRSplineVolume(spline.shape[0], spline.shape[1], spline.shape[2], spline.order(0), spline.order(1), spline.order(2), spline.knots(0, True), spline.knots(1, True), spline.knots(2, True), cp))
    print([b.controlpoint for b in lr_model[-1].basis])

# refine accordint to specifications
for i, lrs in enumerate(lr_model):
    print(lrs)
    set_max_meshsize(lrs, ref)

    # for SURFACE cases
    if lrs.pardim == 2: 
        ### dump results from refinement to eps file
        # filename = f'patch_{i}.eps'
        # print(f'Writing mesh to {filename}')
        # with open(filename, 'wb') as myfile:
        #    lrs.write_postscript(myfile)

        # create a matplotlib figure of the mesh results
        for m in lrs.meshlines:
            t = np.linspace(*m.span(m.variable_direction), 40)
            if m.constant_direction == 0:
                x = np.array([lrs(m.value,t0) for t0 in t])
            else:
                x = np.array([lrs(t0,m.value) for t0 in t])
            plt.plot(x[:,0],x[:,1], 'b-')

    filename = f'patch_{i}.lr'
    print(f'Writing result to {filename}')
    with open(filename, 'wb') as myfile:
        lrs.write(myfile)

plt.axis('equal')
plt.show()

