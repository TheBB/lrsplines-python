import numpy as np

from . import raw


def check_direction(direction, pardim):
    if direction in {0, 'u', 'U'} and 0 < pardim:
        return 0
    elif direction in {1, 'v', 'V'} and 1 < pardim:
        return 1
    elif direction in {2, 'w', 'W'} and 2 < pardim:
        return 2
    raise ValueError('Invalid direction')


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
        return self.w.getParmin(check_direction(direction, self.pardim))

    def end(self, direction=None):
        if direction is None:
            return tuple(self.w.getParmax(d) for d in range(self.pardim))
        return self.w.getParmax(check_direction(direction, self.pardim))

    def span(self, direction=None):
        if direction is None:
            return tuple((self.w.getParmin(d), self.w.getParmax(d)) for d in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return (self.w.getParmin(direction), self.w.getParmax(direction))

    def support(self):
        for w in self.w.supportIter():
            yield BasisFunction(self.lr, w)


class MeshInterface(SimpleWrapper):

    def start(self, direction=None):
        if direction is None:
            return tuple(k[0] for k in self.span())
        return self.span(direction)[0]

    def end(self, direction=None):
        if direction is None:
            return tuple(k[1] for k in self.span())
        return self.span(direction)[1]


class MeshLine(MeshInterface):

    def __str__(self):
        parname = 'u' if self.w.is_spanning_u() else 'v'
        return f'MeshLine({self.w.start_} < {parname} < {self.w.stop_})'

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
        direction = check_direction(direction, self.lr.pardim)
        if direction == self.variable_direction:
            return (self.w.start_, self.w.stop_)
        return (self.w.const_par_, self.w.const_par_)


class ElementView:

    def __init__(self, lr):
        self.lr = lr

    def __len__(self):
        return self.lr.w.nElements()

    def __getitem__(self, idx):
        return self.lr.w.getElement(idx)

    def __iter__(self):
        for w in self.lr.w.elementIter():
            yield Element(self.lr, w)


class BasisView:

    def __init__(self, lr):
        self.lr = lr

    def __len__(self):
        return self.lr.w.nBasisFunctions()

    def __getitem__(self, idx):
        return self.lr.w.getBasisfunction(idx)

    def __iter__(self):
        for w in self.lr.w.basisIter():
            yield BasisFunction(self.lr, w)


class MeshLineView:

    def __init__(self, lr):
        self.lr = lr

    def __len__(self):
        return self.lr.w.nMeshlines()

    def __getitem__(self, idx):
        return self.lr.w.getMeshline(idx)

    def __iter__(self):
        for w in self.lr.w.meshlineIter():
            yield Meshline(self.lr, w)


class LRSplineObject:

    def __init__(self, w):
        self.w = w
        self.elements = ElementView(self)
        self.basis = BasisView(self)

    def __len__(self):
        return len(self.basis)

    @property
    def pardim(self):
        return next(self.elements()).pardim

    @property
    def dimension(self):
        return len(next(self.basis()).controlpoint)

    @dimension.setter
    def dimension(self, value):
        self.w.rebuildDimension(value)

    @property
    def shape(self):
        return (len(self),)

    @property
    def controlpoints(self):
        return np.array([bf.controlpoint for bf in self.basis()])

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
        return self.w.startparam(check_direction(direction, self.pardim))

    def end(self, direction=None):
        if direction is None:
            return tuple(self.w.endparam(d) for d in range(self.pardim))
        return self.w.endparam(check_direction(direction, self.pardim))

    def span(self, direction=None):
        if direction is None:
            return tuple((self.w.startparam(d), self.w.endparam(d)) for d in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return (self.w.startparam(direction), self.w.endparam(direction))

    def order(self, direction=None):
        if direction is None:
            return tuple(self.w.order(d) for d in range(self.pardim))
        return self.w.order(check_direction(direction, self.pardim))

    def edge(self, *args):
        side = raw.parameterEdge.NONE
        for arg in args:
            side |= {
                'west': raw.parameterEdge.WEST,
                'east': raw.parameterEdge.EAST,
                'south': raw.parameterEdge.SOUTH,
                'north': raw.parameterEdge.NORTH,
                'top': raw.parameterEdge.TOP,
                'bottom': raw.parameterEdge.BOTTOM,
            }[arg]

        for w in self.w.getEdgeFunctionsIter(side):
            yield BasisFunction(self, w)

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

    def clone(self):
        return LRSplineSurface(self.w.copy())

    def write_postscript(self, stream, **kwargs):
        return self.w.writePostscriptElements(stream, **kwargs)

    def meshlines(self):
        for w in self.w.meshlineIter():
            yield MeshLine(self, w)

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
        direction = check_direction(direction, self.pardim)

        if direction == 0:
            self.w.insert_const_u_edge(value, start, end, multiplicity)
        else:
            self.w.insert_const_v_edge(value, start, end, multiplicity)

    def evaluate(self, u, v, iel=-1):
        retval = np.array([self.w.point(up, vp, iEl=iel) for up, vp in zip(u.flat, v.flat)])
        return retval.reshape(u.shape)

    def derivative(self, u, v, d=(1,1), iel=-1):
        nderivs = sum(d)
        index = sum(dd + 1 for dd in range(nderivs)) + d[1]
        retval = []
        for up, vp in zip(u.flat, v.flat):
            r = self.w.point(up, vp, nderivs, iEl=iel)
            retval.append(r[index])
        return np.array(retval).reshape(u.shape)

    __call__ = evaluate
