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
    cdef cppclass Basisfunction 'LR::Basisfunction':
        int getId()
        void getControlPoint(vector[double]&)

cdef extern from 'Element.h' namespace 'LR':
    cdef cppclass Element 'LR::Element':
        int getId()
        int getDim()
        double getParmin(int)
        double getParmax(int)
        HashSet_iterator[Basisfunction*] supportBegin()
        HashSet_iterator[Basisfunction*] supportEnd()

cdef extern from 'LRSpline.h' namespace 'LR':
    cdef enum parameterEdge:
        NONE   =  0
        WEST   =  1
        EAST   =  2
        SOUTH  =  4
        NORTH  =  8
        TOP    = 16
        BOTTOM = 32
    cdef cppclass LRSpline 'LR::LRSpline':
        int dimension()
        int nVariate()
        int nBasisFunctions()
        double startparam(int)
        double endparam(int)
        int order(int)
        void getEdgeFunctions(vector[Basisfunction*]& edgeFunctions, parameterEdge edge, int depth)
        vector[Element*].iterator elementBegin()
        vector[Element*].iterator elementEnd()
        HashSet_iterator[Basisfunction*] basisBegin()
        HashSet_iterator[Basisfunction*] basisEnd()
        bool setControlPoints(vector[double]& cps)
        void rebuildDimension(int dimvalue)

cdef extern from 'LRSplineSurface.h' namespace 'LR':
    cdef cppclass LRSplineSurface 'LR::LRSplineSurface' (LRSpline):
        LRSplineSurface()
        LRSplineSurface(int n1, int n2, int order_u, int order_v)
        void read(istream)
        void point(vector[double]& pt, double u, double v, int iEl) const
        void point(vector[vector[double]]& pts, double u, double v, int derivs, int iEl) const

cdef extern from 'LRSplineVolume.h' namespace 'LR':
    cdef cppclass LRSplineVolume 'LR::LRSplineVolume' (LRSpline):
        LRSplineVolume()
        LRSplineVolume(int n1, int n2, int n3, int order_u, int order_v, int order_w)
        void read(istream)
        void point(vector[double]& pt, double u, double v, double w, int iEl) const
        void point(vector[vector[double]]& pts, double u, double v, double w, int derivs, int iEl) const


cdef class BasisFunction:

    cdef Basisfunction* bf

    @property
    def id(self):
        return self.bf.getId()

    @property
    def controlpoint(self):
        cdef vector[double] data
        self.bf.getControlPoint(data)
        return list(data)


cdef class pyElement:

    cdef Element* el

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
        cdef HashSet_iterator[Basisfunction*] it = self.el.supportBegin()
        cdef HashSet_iterator[Basisfunction*] end = self.el.supportEnd()
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

    cdef LRSpline* lr

    @property
    def pardim(self):
        return self.lr.nVariate()

    @property
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
        cdef vector[Element*].iterator it = self.lr.elementBegin()
        cdef vector[Element*].iterator end = self.lr.elementEnd()
        while it != end:
            el = pyElement()
            el.el = deref(it)
            yield el
            preinc(it)

    def basis_functions(self):
        cdef HashSet_iterator[Basisfunction*] it = self.lr.basisBegin()
        cdef HashSet_iterator[Basisfunction*] end = self.lr.basisEnd()
        while not it.equal(end):
            bf = BasisFunction()
            bf.bf = deref(it)
            yield bf
            preinc(it)

    def edge_functions(self, edge):
        cdef vector[Basisfunction*] bfs
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


cdef class Surface(LRSplineObject):

    def __init__(self,n1,n2,p1,p2):
        cdef LRSplineSurface* lr = new LRSplineSurface(n1,n2,p1,p2)
        self.lr = lr

    @staticmethod
    def from_file(str filename):
        cdef ifstream* stream
        cdef LRSplineSurface* lr
        stream = new ifstream(filename.encode())
        lr = new LRSplineSurface()
        surf = Surface()
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
        cdef LRSplineSurface* lr = new LRSplineSurface()
        lr.read(deref(stream))
        del stream
        surf = Surface()
        surf.lr = lr
        return surf

    def __call__(self, double u, double v, d=(0,0)):
        assert len(d) == 2
        derivs = sum(d)
        cdef LRSplineSurface* lr = <LRSplineSurface*> self.lr
        cdef vector[vector[double]] data
        lr.point(data, u, v, derivs, -1)
        index = sum(dd + 1 for dd in range(derivs)) + d[1]
        return list(data[index])


cdef class Volume(LRSplineObject):

    def __init__(self,n1,n2,n3,p1,p2,p3):
        cdef LRSplineVolume* lr = new LRSplineVolume(n1,n2,n3,p1,p2,p3)
        self.lr = lr

    @staticmethod
    def from_file(str filename):
        cdef ifstream* stream
        cdef LRSplineVolume* lr
        stream = new ifstream(filename.encode())
        lr = new LRSplineVolume()
        vol = Volume()
        if stream.is_open():
            lr.read(deref(stream))
            vol.lr = lr
            stream.close()
            del stream
            return vol
        raise FileNotFoundError()

    @staticmethod
    def from_bytes(bytes bytestring):
        cdef string cppstring = bytestring
        cdef istringstream* stream
        stream = new istringstream(cppstring)
        cdef LRSplineVolume* lr = new LRSplineVolume()
        lr.read(deref(stream))
        del stream
        vol = Volume()
        vol.lr = lr
        return vol

    def __call__(self, double u, double v, double w, d=(0,0,0)):
        assert len(d) == 3
        derivs = sum(d)
        cdef LRSplineVolume* lr = <LRSplineVolume*> self.lr
        cdef vector[vector[double]] data
        lr.point(data, u, v, w, derivs, -1)
        index = sum(dd + 1 for dd in range(derivs)) + d[1]
        return list(data[index])
