from splipy.io import G2
import lrspline as lr
import sys
import numpy as np
import inspect

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
        functions_to_refine = []
        for el in myspline.elements:
            print(el)
            u0 = np.array(el.start())
            u1 = np.array(el.end())
            u = (u0+u1)/2
            x = myspline(*u)
            r = np.linalg.norm(u)
            if len(x) == 2: x = np.append(x,0)
            if len(u) == 2: u = np.append(u,0)
            # look up the target mesh size for this element as measured in either
            # geometric coordinates (x,y,z) or parametric (u,v,w)
            if use_physical_meshsize:
                h = np.linalg.norm(myspline(*u1) - myspline(*u0))
            else:
                h = np.max(u1-u0)

            print(f'Evaluation point: x={x},  u={u})')
            print(f'Actual size: {h}')
            print(f'Target size: {h_target(*x,r,*u)}')
            if h > h_target(*x, r, *u):
                did_refinement = True
                for function in el.support():
                    # since one function might have support on mulitple elements
                    # this list might contain repetitions
                    functions_to_refine.append(function)

            # make the unique list of functions to refine
            functions_to_refine = [f for f in myspline.basis if f in functions_to_refine]

        if did_refinement:
            print('Trying to refine the following functions')
            print([f.id for f in functions_to_refine])
            myspline.refine(functions_to_refine)



### USER DEFINED REFINEMENT FUNCTION
def ref(x):
    return x + 0.100


# read model from file
with G2(sys.argv[1]) as myfile:
    mymodel = myfile.read()

# convert all patches to LRSpline representations
lr_model = []
for spline in mymodel:
    if spline.pardim == 2:
        lr_model.append(lr.LRSplineSurface(spline.shape[0], spline.shape[1], spline.order(0), spline.order(1), spline.knots(0, True), spline.knots(1, True)))
    elif spline.pardim == 3:
        lr_model.append(lr.LRSplineVolume(spline.shape[0], spline.shape[1], spline.shape[2], spline.order(0), spline.order(1), spline.order(2), spline.knots(0, True), spline.knots(1, True), spline.knots(2, True)))

# refine accordint to specifications
for i, lrs in enumerate(lr_model):
    print(lrs)
    set_max_meshsize(lrs, ref)

    if lrs.pardim == 2: # dump results from surface refinement
        filename = f'patch_{i}.eps'
        print(f'Writing mesh to {filename}')
        with open(filename, 'wb') as myfile:
            lrs.write_postscript(myfile)

    filename = f'patch_{i}.lr'
    print(f'Writing result to {filename}')
    with open(filename, 'wb') as myfile:
        lrs.write(myfile)

