from functools import partial
import operator as op
import numpy as np
from itertools import combinations_with_replacement, repeat, chain

from . import raw


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
    peek = stream.peek(20)
    if not peek:
        raise raw.EOFError('')
    if isinstance(peek, bytes):
        peek = peek.decode('utf-8')
    if peek.startswith('# LRSPLINE SURFACE'):
        return LRSplineSurface
    if peek.startswith('# LRSPLINE VOLUME'):
        return LRSplineVolume
    raise ValueError("Unknown LRSpline object type: '{}'".format(peek[:20]))


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

    def derivative(self, u, v, d=(1,1)):
        nderivs = sum(d)
        index = nderivs * (nderivs + 1) // 2 + d[1]
        retval = np.array([
            self.w.evaluate(up, vp, nderivs, True, True)[index]
            for up, vp in zip(u.flat, v.flat)
        ])
        return retval.reshape(u.shape)

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


class LRSplineSurface(LRSplineObject):

    def __init__(self, arg=None):
        if isinstance(arg, raw.LRSurface):
            w = arg
        else:
            w = raw.LRSurface()
            if arg is not None:
                w.read(arg)
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
        nderivs = sum(d)
        index = sum(dd + 1 for dd in range(nderivs)) + d[1]
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
            retval = []
            for up, vp in zip(u.flat, v.flat):
                r = self.w.point(up, vp, nderivs, iEl=iel)
                retval.append(r[index])
            return np.array(retval).reshape(u.shape)
        return self.w.point(u, v, nderivs, iEl=iel)[index]

    __call__ = evaluate


class LRSplineVolume(LRSplineObject):

    def __init__(self, arg=None):
        if isinstance(arg, raw.LRVolume):
            w = arg
        else:
            w = raw.LRVolume()
            if arg is not None:
                w.read(arg)
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

    def derivative(self, u, v, w, d=(1,1,1), iel=-1):
        nderivs = sum(d)
        index = nderivs * (nderivs + 1) * (nderivs + 2) // 6
        tgt = tuple(chain.from_iterable(repeat(i,r) for i,r in enumerate(d)))
        index += next(i for i,t in enumerate(combinations_with_replacement(range(len(d)), nderivs)) if t == tgt)
        if isinstance(u, np.ndarray) and isinstance(u, np.ndarray) and isinstance(u, np.ndarray):
            retval = []
            for up, vp, wp in zip(u.flat, v.flat, w.flat):
                r = self.w.point(up, vp, wp, nderivs, iEl=iel)
                retval.append(r[index])
            return np.array(retval).reshape(u.shape)
        return self.w.point(u, v, w, nderivs, iEl=iel)[index]

    __call__ = evaluate
