# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as preinc

import numpy as np
from splipy.utils import check_direction


cdef extern from '<iostream>' namespace 'std':
    cdef cppclass istream:
        pass

cdef extern from '<fstream>' namespace 'std':
    cdef cppclass ifstream(istream):
        ifstream(const char *) except +
        bool is_open()
        void close()

cdef extern from '<sstream>' namespace 'std':
    cdef cppclass istringstream(istream):
        istringstream(const string& str) except +

cdef extern from 'HashSet.h':
    cdef cppclass HashSet_iterator[T]:
        T operator*()
        HashSet_iterator[T] operator++()
        bool equal(HashSet_iterator[T])

cdef extern from 'Basisfunction.h' namespace 'LR':
    cdef cppclass Basisfunction_ 'LR::Basisfunction':
        int getId()
        void getControlPoint(vector[double]&)

cdef extern from 'Element.h' namespace 'LR':
    cdef cppclass Element_ 'LR::Element':
        int getId()
        int getDim()
        double getParmin(int)
        double getParmax(int)
        HashSet_iterator[Basisfunction_*] supportBegin()
        HashSet_iterator[Basisfunction_*] supportEnd()

cdef extern from 'LRSpline.h' namespace 'LR':
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
        void getEdgeFunctions(vector[Basisfunction_*]& edgeFunctions, parameterEdge edge, int depth)
        vector[Element_*].iterator elementBegin()
        vector[Element_*].iterator elementEnd()
        HashSet_iterator[Basisfunction_*] basisBegin()
        HashSet_iterator[Basisfunction_*] basisEnd()
        bool setControlPoints(vector[double]& cps)
        void rebuildDimension(int dimvalue)

cdef extern from 'LRSplineSurface.h' namespace 'LR':
    cdef cppclass LRSplineSurface_ 'LR::LRSplineSurface' (LRSpline_):
        LRSplineSurface() except +
        void read(istream) except +
        void point(vector[double]& pt, double u, double v, int iEl) const
        void point(vector[vector[double]]& pts, double u, double v, int derivs, int iEl) const


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

    @property
    def pardim(self):
        return self.lr.nVariate()

    def dimension(self):
        return self.lr.dimension()

    def controlpoints(self):
        cps = np.empty((len(self), self.dimension()))
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

    def __call__(self, double u, double v, d=(0,0)):
        assert len(d) == 2
        derivs = sum(d)
        cdef LRSplineSurface_* lr = <LRSplineSurface_*> self.lr
        cdef vector[vector[double]] data
        lr.point(data, u, v, derivs, -1)
        index = sum(dd + 1 for dd in range(derivs)) + d[1]
        return list(data[index])
