from functools import partial
import operator as op
import io
import numpy as np
from itertools import combinations_with_replacement, repeat, chain

from . import raw


__version__ = '1.6.0'


def _check_direction(direction, pardim):
    if direction in {0, 'u', 'U'} and 0 < pardim:
        return 0
    elif direction in {1, 'v', 'V'} and 1 < pardim:
        return 1
    elif direction in {2, 'w', 'W'} and 2 < pardim:
        return 2
    raise ValueError('Invalid direction')


def _check_edge(edge):
    side = raw.parameterEdge.NONE
    for arg in edge:
        side |= {
            'west': raw.parameterEdge.WEST,
            'east': raw.parameterEdge.EAST,
            'south': raw.parameterEdge.SOUTH,
            'north': raw.parameterEdge.NORTH,
            'top': raw.parameterEdge.TOP,
            'bottom': raw.parameterEdge.BOTTOM,
        }[arg]
    return side


def _constructor(stream):
    peek = stream.readline()
    if not peek:
        raise raw.EOFError('')
    if isinstance(peek, bytes):
        peek = peek.decode('utf-8')
    if peek.startswith('# LRSPLINE SURFACE'):
        return LRSplineSurface
    if peek.startswith('# LRSPLINE VOLUME'):
        return LRSplineVolume
    raise ValueError("Unknown LRSpline object type: '{}'".format(peek))


def _derivative_index(d):
    """Calculate the derivative index of 'd' (a 2-tuple or 3-tuple) using
    LRSplines' derivative numbering scheme.  Return nderivs and index.
    """
    nderivs = sum(d)
    if len(d) == 2:
        index = sum(dd + 1 for dd in range(nderivs)) + d[1]
        return nderivs, index
    index = nderivs * (nderivs + 1) * (nderivs + 2) // 6
    tgt = tuple(chain.from_iterable(repeat(i,r) for i,r in enumerate(d)))
    index += next(i for i,t in enumerate(combinations_with_replacement(range(len(d)), nderivs)) if t == tgt)
    return nderivs, index


def _derivative_helper(pts, derivs, func):
    """Helper for calculating derivatives.

    Pts must be a tuple of points, or a tuple of point arrays.

    Derivs must be a tuple indicating which derivative to take, or a
    list of such tuples.

    Func must be a callable accepting two or three floats (the point
    to evaluate) and an 'nderivs' argument, returning a 1-dimensional
    array.

    - If one point, one deriv: return a 1-dimensional array (coord)
    - If multiple points, one deriv: return a 2-dimensional array (point x coord)
    - If one point, multiple derivs: return a 2-dimensional array (deriv x coord)
    - If multiple points, multiple derivs: return 3-dimensional (deriv x point x coord)
    """

    singlept = not isinstance(pts[0], np.ndarray)
    singlederiv = isinstance(derivs[0], int)

    if singlederiv:
        nderiv, index = _derivative_index(derivs)
    else:
        nderiv, indexes = 0, []
        for deriv in derivs:
            n, i = _derivative_index(deriv)
            nderiv = max(n, nderiv)
            indexes.append(i)

    if singlept:
        data = func(*pts, nderiv)
        if singlederiv:
            return data[index]
        else:
            return np.array([data[i] for i in indexes])
    else:
        data = [func(*pt, nderiv) for pt in zip(*pts)]
        if singlederiv:
            return np.array([d[index] for d in data])
        else:
            return np.array([[d[i] for d in data] for i in indexes])


class SimpleWrapper:

    def __init__(self, lr, w):
        self.w = w
        self.lr = lr


class BasisFunction(SimpleWrapper):

    def __str__(self):
        return f'BasisFunction#{self.id}'

    @property
    def id(self):
        return self.w.getId()

    @property
    def controlpoint(self):
        return self.w.getControlPoint()

    @property
    def nvariate(self):
        return self.w.nVariate()

    def evaluate(self, u, v):
        retval = np.array([self.w.evaluate(up, vp, True, True) for up, vp in zip(u.flat, v.flat)])
        return retval.reshape(u.shape)

    def derivative(self, *pts, d=(1,1)):
        if self.nvariate == 2:
            wrapper = lambda u,v,n: self.w.evaluate(u, v, n, True, True)
        else:
            wrapper = lambda u,v,w,n: self.w.evaluate(u, v, w, n, True, True, True )
        return _derivative_helper(pts, d, wrapper)

    def __getitem__(self, idx):
        return self.w.getknots(idx)

    __call__ = evaluate

    def refine(self):
        self.lr.w.refineBasisFunction(self.id)
        self.lr.w.generateIDs()


class Element(SimpleWrapper):

    def __str__(self):
        return f'Element#{self.id}'

    @property
    def id(self):
        return self.w.getId()

    @property
    def pardim(self):
        return self.w.getDim()

    def start(self, direction=None):
        if direction is None:
            return tuple(self.w.getParmin(d) for d in range(self.pardim))
        return self.w.getParmin(_check_direction(direction, self.pardim))

    def end(self, direction=None):
        if direction is None:
            return tuple(self.w.getParmax(d) for d in range(self.pardim))
        return self.w.getParmax(_check_direction(direction, self.pardim))

    def span(self, direction=None):
        if direction is None:
            return tuple((self.w.getParmin(d), self.w.getParmax(d)) for d in range(self.pardim))
        direction = _check_direction(direction, self.pardim)
        return (self.w.getParmin(direction), self.w.getParmax(direction))

    def support(self):
        for w in self.w.supportIter():
            yield BasisFunction(self.lr, w)

    def refine(self):
        self.lr.w.refineElement(self.id)
        self.lr.w.generateIDs()


class MeshInterface(SimpleWrapper):

    def __str__(self):
        cls = self.__class__.__name__
        const = '{} = {}'.format('uvw'[self.constant_direction], self.value)
        variables = [
            '{} < {} < {}'.format(self.start(i), 'uvw'[i], self.end(i))
            for i in self.variable_directions
        ]
        return cls + '(' + const + '; ' + ', '.join(variables) + ')'

    def start(self, direction=None):
        if direction is None:
            return tuple(k[0] for k in self.span())
        return self.span(direction)[0]

    def end(self, direction=None):
        if direction is None:
            return tuple(k[1] for k in self.span())
        return self.span(direction)[1]


class MeshLine(MeshInterface):

    @property
    def variable_direction(self):
        return 0 if self.w.is_spanning_u() else 1

    @property
    def variable_directions(self):
        return (0,) if self.w.is_spanning_u() else (1,)

    @property
    def constant_direction(self):
        return 1 if self.w.is_spanning_u() else 0

    @property
    def value(self):
        return self.w.const_par_

    @property
    def multiplicity(self):
        return self.w.multiplicity_

    def span(self, direction=None):
        if direction is None:
            var = (self.w.start_, self.w.stop_)
            const = (self.w.const_par_, self.w.const_par_)
            return (var, const) if self.w.is_spanning_u() else (const, var)
        direction = _check_direction(direction, self.lr.pardim)
        if direction == self.variable_direction:
            return (self.w.start_, self.w.stop_)
        return (self.w.const_par_, self.w.const_par_)


class MeshRect(MeshInterface):

    @property
    def variable_directions(self):
        return {
            0: (1, 2),
            1: (0, 2),
            2: (0, 1),
        }[self.w.constCirection()]

    @property
    def constant_direction(self):
        return self.w.constDirection()

    @property
    def value(self):
        return self.w.constParameter()

    @property
    def multiplicity(self):
        return self.w.multiplicity_

    def span(self, direction=None):
        start, end = self.w.start_, self.w.end_
        if direction is None:
            return tuple(zip(start, end))
        direction = _check_direction(direction, self.lr.pardim)
        return start[direction], end[direction]



class ListLikeView:

    def __init__(self, obj, lenf, itemf, iterf, wrapf):
        self.obj = obj
        self.lenf = op.methodcaller(lenf)
        self.itemf = itemf
        self.iterf = op.methodcaller(iterf)
        self.wrapf = wrapf

    def __len__(self):
        return self.lenf(self.obj)

    def __getitem__(self, idx):
        return self.wrapf(getattr(self.obj, self.itemf)(idx))

    def __iter__(self):
        for w in self.iterf(self.obj):
            yield self.wrapf(w)


class ElementView(ListLikeView):

    def __init__(self, lr):
        super().__init__(lr.w, 'nElements', 'getElement', 'elementIter', partial(Element, lr))
        self.lr = lr

    def edge(self, *edge):
        for w in self.lr.w.getEdgeElementsIter(_check_edge(edge)):
            yield Element(self.lr, w)


class BasisView(ListLikeView):

    def __init__(self, lr):
        super().__init__(lr.w, 'nBasisFunctions', 'getBasisfunction', 'basisIter', partial(BasisFunction, lr))
        self.lr = lr

    def edge(self, *edge, depth=1):
        for w in self.lr.w.getEdgeFunctionsIter(_check_edge(edge)):
            yield BasisFunction(self.lr, w)


class MeshLineView(ListLikeView):

    def __init__(self, lr):
        super().__init__(lr.w, 'nMeshlines', 'getMeshline', 'meshlineIter', partial(MeshLine, lr))


class MeshRectView(ListLikeView):

    def __init__(self, lr):
        super().__init__(lr.w, 'nMeshRectangles', 'getMeshRectangle', 'meshrectIter', partial(MeshRect, lr))


class LRSplineObject:

    def __init__(self, w):
        w.generateIDs()
        self.w = w
        self.elements = ElementView(self)
        self.basis = BasisView(self)

    @staticmethod
    def read_many(stream):
        objects = []
        while True:
            try:
                cls = _constructor(stream)
                objects.append(cls(stream))
            except raw.EOFError:
                break
        return objects

    def __len__(self):
        return len(self.basis)

    @property
    def pardim(self):
        return next(iter(self.elements)).pardim

    @property
    def dimension(self):
        return len(next(iter(self.basis)).controlpoint)

    @dimension.setter
    def dimension(self, value):
        self.w.rebuildDimension(value)

    @property
    def shape(self):
        return (len(self),)

    @property
    def controlpoints(self):
        return np.array([bf.controlpoint for bf in self.basis])

    @controlpoints.setter
    def controlpoints(self, value):
        if value.shape[0] != len(self):
            raise ValueError(f'Incorrect number of control points: expected {len(self)}')
        if value.ndim != 2:
            raise ValueError(f'Incorrect number of dimensions: expected 2')
        _, newdim = value.shape
        if newdim != self.dimension:
            self.dimension = newdim
        self.w.setControlpoints(value.flat)

    def write(self, stream):
        return self.w.write(stream)

    def start(self, direction=None):
        if direction is None:
            return tuple(self.w.startparam(d) for d in range(self.pardim))
        return self.w.startparam(_check_direction(direction, self.pardim))

    def end(self, direction=None):
        if direction is None:
            return tuple(self.w.endparam(d) for d in range(self.pardim))
        return self.w.endparam(_check_direction(direction, self.pardim))

    def span(self, direction=None):
        if direction is None:
            return tuple((self.w.startparam(d), self.w.endparam(d)) for d in range(self.pardim))
        direction = _check_direction(direction, self.pardim)
        return (self.w.startparam(direction), self.w.endparam(direction))

    def order(self, direction=None):
        if direction is None:
            return tuple(self.w.order(d) for d in range(self.pardim))
        return self.w.order(_check_direction(direction, self.pardim))

    def knots(self, direction=None):
        if direction is None:
            return self.w.getGlobalUniqueKnotVector()
        direction = _check_direction(direction, self.pardim)
        return self.w.getGlobalUniqueKnotVector()[direction]

    def refine(self, objects, beta=None):
        if not objects:
            raise ValueError('Refinement list must be non-empty')
        elif isinstance(objects[0], float):
            self.w.refineByDimensionIncrease(objects, beta)
        elif isinstance(objects[0], BasisFunction):
            ids = [bf.id for bf in objects]
            self.w.refineBasisFunction(ids)
        elif isinstance(objects[0], Element):
            ids = [bf.id for bf in objects]
            self.w.refineElement(ids)
        else:
            raise TypeError('List of unknown objects: expected float, BasisFunction or Element')
        self.w.generateIDs()

    def configure(self, **kwargs):
        if 'aspect_ratio' in kwargs:
            r = kwargs.pop('aspect_ratio')
            posteriori = kwargs.pop('posteriori_fix', True)
            self.w.setMaxAspectRatio(r, posteriori)
        for key, val in kwargs.items():
            attr = {
                'strategy': 'setRefStrat',
                'symmetry': 'setRefSymmetry',
                'multiplicity': 'setRefMultiplicity',
                'max_tjoints': 'setMaxTjoints',
                'close_gaps': 'setCloseGaps',
            }[key]
            getattr(self.w, key)(val)

    def generate_ids(self):
        self.w.generateIDs()

    def __mul__(self, x):
        new = self.clone()
        new.controlpoints *= x
        return new

    def __rmul__(self, x):
        return self * x

    def __getitem__(self, i):
        return self.basis[i].controlpoint


class LRSplineSurface(LRSplineObject):

    def __init__(self, *args):
        if len(args) == 0:
            w = raw.LRSurface()
        elif isinstance(args[0], raw.LRSurface):
            w = arg
        elif isinstance(args[0], (io.IOBase,str,bytes)):
            w = raw.LRSurface()
            w.read(args[0])
        elif len(args) == 2: # specify (n1,n2)
            w = raw.LRSurface(args[0], args[1], 2, 2)
        elif len(args) == 4: # specify (n1,n2) and (p1,p2)
            w = raw.LRSurface(args[0], args[1], args[2], args[3])
        elif len(args) == 6: # specify (n1,n2), (p1,p2) and (knot1,knot2)
            w = raw.LRSurface(args[0], args[1], args[2], args[3], args[4], args[5])
        elif len(args) == 7: # specify controlpoints in addition
            w = raw.LRSurface(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        else:
            w = raw.LRSurface()
        super().__init__(w)
        self.meshlines = MeshLineView(self)

    def corners(self):
        return np.array([
            next(self.basis.edge('east', 'south')).controlpoint,
            next(self.basis.edge('west', 'south')).controlpoint,
            next(self.basis.edge('east', 'north')).controlpoint,
            next(self.basis.edge('west', 'north')).controlpoint,
        ])

    def clone(self):
        return LRSplineSurface(self.w.copy())

    def write_postscript(self, stream, **kwargs):
        return self.w.writePostscriptElements(stream, **kwargs)

    def insert(self, *args, direction=None, value=None, start=None, end=None, multiplicity=None):
        if len(args) > 1:
            raise TypeError('Expected at most one positional argument')

        if len(args) == 1:
            ml, = args
            if not isinstance(ml, MeshLine):
                raise TypeError(f'Expected MeshLine, got {type(ml)}')
            if direction is None:
                direction = ml.constant_direction
            if value is None:
                value = ml.value
            if start is None:
                start = ml.start(ml.variable_direction)
            if end is None:
                end = ml.end(ml.variable_direction)
            if multiplicity is None:
                multiplicity = ml.multiplicity

        if multiplicity is None:
            multiplicity = 1
        direction = _check_direction(direction, self.pardim)

        if direction == 0:
            self.w.insert_const_u_edge(value, start, end, multiplicity)
        else:
            self.w.insert_const_v_edge(value, start, end, multiplicity)

    def evaluate(self, u, v, iel=-1):
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
            retval = np.array([self.w.point(up, vp, iEl=iel) for up, vp in zip(u.flat, v.flat)])
            return retval.reshape(u.shape)
        return self.w.point(u, v, iEl=iel)

    def derivative(self, u, v, d=(1,1), iel=-1):
        wrapper = lambda u,v,n: self.w.point(u, v, n, iEl=iel)
        return _derivative_helper((u, v), d, wrapper)

    def bezier_extraction(self, iEl):
        return self.w.getBezierExtraction(iEl)

    __call__ = evaluate


class LRSplineVolume(LRSplineObject):

    def __init__(self, *args):
        if len(args) == 0:
            w = raw.LRVolume()
        elif isinstance(args[0], raw.LRVolume):
            w = arg
        elif isinstance(args[0], (io.IOBase,str,bytes)):
            w = raw.LRVolume()
            w.read(args[0])
        elif len(args) == 3: # only specify number of functions (n1,n2,n3)
            w = raw.LRVolume(args[0], args[1], args[2], 2, 2, 2)
        elif len(args) == 6: # specity n & p for 3 directions
            w = raw.LRVolume(*args)
        elif len(args) == 9: # specify n,p and knotvector for 3 directions
            w = raw.LRVolume(*args)
        elif len(args) == 10: # specify all above in addition to controlpoints
            w = raw.LRVolume(*args)
        else:
            w = raw.LRVolume()
        super().__init__(w)
        self.meshrects = MeshRectView(self)

    def corners(self):
        return np.array([
            next(self.basis.edge('east', 'south', 'bottom')).controlpoint,
            next(self.basis.edge('west', 'south', 'bottom')).controlpoint,
            next(self.basis.edge('east', 'north', 'bottom')).controlpoint,
            next(self.basis.edge('west', 'north', 'bottom')).controlpoint,
            next(self.basis.edge('east', 'south', 'top')).controlpoint,
            next(self.basis.edge('west', 'south', 'top')).controlpoint,
            next(self.basis.edge('east', 'north', 'top')).controlpoint,
            next(self.basis.edge('west', 'north', 'top')).controlpoint,
        ])

    def clone(self):
        return LRSplineVolume(self.w.copy())

    def insert(self, mr):
        self.w.insert(mr)

    def evaluate(self, u, v, w, iel=-1):
        if isinstance(u, np.ndarray) and isinstance(u, np.ndarray) and isinstance(u, np.ndarray):
            retval = np.array([self.w.point(up, vp, wp, iEl=iel) for up, vp, wp in zip(u.flat, v.flat, w.flat)])
            return retval.reshape(u.shape)
        return self.w.point(u, v, w, iEl=iel)

    def bezier_extraction(self, iEl):
        return self.w.getBezierExtraction(iEl)

    def derivative(self, u, v, w, d=(1,1,1), iel=-1):
        wrapper = lambda u,v,w,n: self.w.point(u, v, w, n, iEl=iel)
        return _derivative_helper((u, v, w), d, wrapper)

    __call__ = evaluate
