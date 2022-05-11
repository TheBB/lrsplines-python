# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as preinc

import numpy as np
def check_direction(direction, pardim):
    if direction in {0, 'u', 'U'} and 0 < pardim:
        return 0
    elif direction in {1, 'v', 'V'} and 1 < pardim:
        return 1
    elif direction in {2, 'w', 'W'} and 2 < pardim:
        return 2
    raise ValueError('Invalid direction')


cdef extern from '<iostream>' namespace 'std':
    cdef cppclass istream:
        pass
    cdef cppclass ostream:
        pass

cdef extern from '<fstream>' namespace 'std':
    cdef cppclass ifstream(istream):
        ifstream(const char *) except +
        bool is_open()
        void close()
    cdef cppclass ofstream(ostream):
        ofstream(const char *) except +
        bool is_open()
        void close()

cdef extern from '<sstream>' namespace 'std':
    cdef cppclass istringstream(istream):
        istringstream(const string& str) except +

cdef extern from 'LRSpline/HashSet.h':
    cdef cppclass HashSet_iterator[T]:
        T operator*()
        HashSet_iterator[T] operator++()
        bool equal(HashSet_iterator[T])

cdef extern from 'LRSpline/Basisfunction.h' namespace 'LR':
    cdef cppclass Basisfunction_ 'LR::Basisfunction':
        int getId()
        void getControlPoint(vector[double]&)
        int nVariate() const
        double evaluate(double u, double v, bool u_from_right, bool v_from_right) const
        void evaluate(vector[double]& results, double u, double v, int derivs, bool u_from_right, bool v_from_right) const

cdef extern from 'LRSpline/Element.h' namespace 'LR':
    cdef cppclass Element_ 'LR::Element':
        int getId()
        int getDim()
        double getParmin(int)
        double getParmax(int)
        HashSet_iterator[Basisfunction_*] supportBegin()
        HashSet_iterator[Basisfunction_*] supportEnd()

cdef extern from 'LRSpline/Meshline.h' namespace 'LR':
    cdef cppclass Meshline_ 'LR::Meshline':
        bool is_spanning_u()
        double const_par_
        double start_
        double stop_
        int multiplicity_

cdef extern from 'LRSpline/LRSpline.h' namespace 'LR':
    cdef enum parameterEdge:
        NONE   =  0
        WEST   =  1
        EAST   =  2
        SOUTH  =  4
        NORTH  =  8
        TOP    = 16
        BOTTOM = 32
    cdef cppclass LRSpline_ 'LR::LRSpline':
        int dimension()
        int nVariate()
        int nBasisFunctions()
        double startparam(int)
        double endparam(int)
        int order(int)
        void generateIDs()
        void getEdgeFunctions(vector[Basisfunction_*]& edgeFunctions, parameterEdge edge, int depth)
        vector[Element_*].iterator elementBegin()
        vector[Element_*].iterator elementEnd()
        HashSet_iterator[Basisfunction_*] basisBegin()
        HashSet_iterator[Basisfunction_*] basisEnd()
        vector[Meshline_*].iterator meshlineBegin()
        vector[Meshline_*].iterator meshlineEnd()
        bool setControlPoints(vector[double]& cps)
        void rebuildDimension(int dimvalue)

cdef extern from 'LRSpline/LRSplineSurface.h' namespace 'LR':
    cdef cppclass LRSplineSurface_ 'LR::LRSplineSurface' (LRSpline_):
        LRSplineSurface() except +
        LRSplineSurface_* copy()
        void read(istream) except +
        void write(ostream) except +
        void point(vector[double]& pt, double u, double v, int iEl) const
        void point(vector[vector[double]]& pts, double u, double v, int derivs, int iEl) const
        void getGlobalUniqueKnotVector(vector[double]& knot_u, vector[double]& knot_v) const
        Meshline_* insert_const_u_edge(double u, double start_v, double stop_v, int multiplicity)
        Meshline_* insert_const_v_edge(double v, double start_u, double stop_u, int multiplicity)
        void writePostscriptElements(ostream, int, int, bool, vector[int]*) const
        void writePostscriptMesh(ostream, bool, vector[int]*) const
        void writePostscriptMeshWithControlPoints(ostream, int, int) const
        double makeIntegerKnots()


cdef class BasisFunction:

    cdef Basisfunction_* bf

    @property
    def id(self):
        return self.bf.getId()

    @property
    def controlpoint(self):
        cdef vector[double] data
        self.bf.getControlPoint(data)
        return list(data)

    @property
    def nvariate(self):
        return self.bf.nVariate()

    # TODO: Implement for trivariate
    def __call__(self, double u, double v, d=None):
        if d is None:
            return self.bf.evaluate(u, v, <bool> True, <bool> True)
        assert len(d) == 2
        derivs = sum(d)
        index = derivs * (derivs + 1) // 2 + d[1]
        cdef vector[double] results
        self.bf.evaluate(results, u, v, <int> derivs, <bool> True, <bool> True)
        return results[index]


cdef class Element:

    cdef Element_* el

    @property
    def id(self):
        return self.el.getId()

    @property
    def pardim(self):
        return self.el.getDim()

    def start(self, direction=None):
        if direction is None:
            return tuple(self.el.getParmin(i) for i in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return self.el.getParmin(direction)

    def end(self, direction=None):
        if direction is None:
            return tuple(self.el.getParmax(i) for i in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return self.el.getParmax(direction)

    def basis_functions(self):
        cdef HashSet_iterator[Basisfunction_*] it = self.el.supportBegin()
        cdef HashSet_iterator[Basisfunction_*] end = self.el.supportEnd()
        while not it.equal(end):
            bf = BasisFunction()
            bf.bf = deref(it)
            yield bf
            preinc(it)


cdef class Meshline:

    cdef Meshline_* ml

    @property
    def is_spanning_u(self):
        return self.ml.is_spanning_u()

    @property
    def const_par(self):
        return self.ml.const_par_

    @property
    def start(self):
        return self.ml.start_

    @property
    def stop(self):
        return self.ml.stop_

    @property
    def multiplicity(self):
        return self.ml.multiplicity_


cdef class ParameterEdge:
    NONE   =  0
    WEST   =  1
    EAST   =  2
    SOUTH  =  4
    NORTH  =  8
    TOP    = 16
    BOTTOM = 32


cdef class LRSplineObject:

    cdef LRSpline_* lr

    def __del__(self):
        if self.lr:
            del self.lr

    @property
    def pardim(self):
        return self.lr.nVariate()

    @property
    def dimension(self):
        return self.lr.dimension()

    def set_dimension(self, dim):
        self.lr.rebuildDimension(dim)

    @property
    def controlpoints(self):
        cps = np.empty((len(self), self.dimension))
        for i, bf in enumerate(self.basis_functions()):
            cps[i,:] = bf.controlpoint
        return cps

    def __len__(self):
        return self.lr.nBasisFunctions()

    def start(self, direction=None):
        if direction is None:
            return tuple(self.lr.startparam(i) for i in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return self.lr.startparam(direction)

    def end(self, direction=None):
        if direction is None:
            return tuple(self.lr.endparam(i) for i in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return self.lr.endparam(direction)

    def order(self, direction=None):
        if direction is None:
            return tuple(self.lr.order(i) for i in range(self.pardim))
        direction = check_direction(direction, self.pardim)
        return self.lr.order(direction)

    def elements(self):
        cdef vector[Element_*].iterator it = self.lr.elementBegin()
        cdef vector[Element_*].iterator end = self.lr.elementEnd()
        while it != end:
            el = Element()
            el.el = deref(it)
            yield el
            preinc(it)

    def basis_functions(self):
        cdef HashSet_iterator[Basisfunction_*] it = self.lr.basisBegin()
        cdef HashSet_iterator[Basisfunction_*] end = self.lr.basisEnd()
        while not it.equal(end):
            bf = BasisFunction()
            bf.bf = deref(it)
            yield bf
            preinc(it)

    def edge_functions(self, edge):
        cdef vector[Basisfunction_*] bfs
        self.lr.getEdgeFunctions(bfs, edge, 1)
        it = bfs.begin()
        while it != bfs.end():
            bf = BasisFunction()
            bf.bf = deref(it)
            yield bf
            preinc(it)

    def set_controlpoints(self, cps):
        assert len(cps) % len(self) == 0
        self.lr.rebuildDimension(len(cps) // len(self))
        cdef vector[double] vec_cps
        vec_cps = list(cps)
        self.lr.setControlPoints(cps)

    def corners(self):
        return [
            next(self.edge_functions(ParameterEdge.SOUTH | ParameterEdge.WEST)).controlpoint,
            next(self.edge_functions(ParameterEdge.SOUTH | ParameterEdge.EAST)).controlpoint,
            next(self.edge_functions(ParameterEdge.NORTH | ParameterEdge.WEST)).controlpoint,
            next(self.edge_functions(ParameterEdge.NORTH | ParameterEdge.EAST)).controlpoint,
        ]

    def generate_ids(self):
        self.lr.generateIDs()


cdef class LRSurface(LRSplineObject):

    @staticmethod
    def from_file(str filename):
        cdef ifstream* stream
        cdef LRSplineSurface_* lr
        stream = new ifstream(filename.encode())
        lr = new LRSplineSurface_()
        surf = LRSurface()
        if stream.is_open():
            lr.read(deref(stream))
            surf.lr = lr
            stream.close()
            del stream
            return surf
        raise FileNotFoundError()

    @staticmethod
    def from_bytes(bytes bytestring):
        cdef string cppstring = bytestring
        cdef istringstream* stream
        stream = new istringstream(cppstring)
        cdef LRSplineSurface_* lr = new LRSplineSurface_()
        lr.read(deref(stream))
        del stream
        surf = LRSurface()
        surf.lr = lr
        return surf

    def to_file(self, str filename):
        cdef ofstream* stream
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        stream = new ofstream(filename.encode())
        if stream.is_open():
            lr.write(deref(stream))
            stream.close()
        del stream

    def to_postscript(self, str filename):
        cdef ofstream* stream
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        stream = new ofstream(filename.encode())
        if stream.is_open():
            lr.writePostscriptElements(deref(stream), 2, 2, True, NULL)
            stream.close()
        del stream

    def clone(self):
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        cdef LRSplineSurface_* copy = lr.copy()
        surf = LRSurface()
        surf.lr = copy
        return surf

    def knots(self):
        cdef vector[double] knots_u
        cdef vector[double] knots_v
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        lr.getGlobalUniqueKnotVector(knots_u, knots_v)
        return (tuple(knots_u), tuple(knots_v))

    def __call__(self, double u, double v, d=(0,0)):
        assert len(d) == 2
        derivs = sum(d)
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        cdef vector[vector[double]] data
        lr.point(data, u, v, derivs, -1)
        index = sum(dd + 1 for dd in range(derivs)) + d[1]
        return list(data[index])

    def meshlines(self):
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        cdef vector[Meshline_*].iterator it = lr.meshlineBegin()
        cdef vector[Meshline_*].iterator end = lr.meshlineEnd()
        while it != end:
            ml = Meshline()
            ml.ml = deref(it)
            yield ml
            preinc(it)

    def insert_const_u_edge(self, double u, double start_v, double stop_v, int multiplicity=1):
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        lr.insert_const_u_edge(u, start_v, stop_v, multiplicity)

    def insert_const_v_edge(self, double v, double start_u, double stop_u, int multiplicity=1):
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        lr.insert_const_v_edge(v, start_u, stop_u, multiplicity)
