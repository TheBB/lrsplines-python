# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as preinc

import numpy as np


cdef extern from '<iostream>' namespace 'std':
    cdef cppclass istream:
        pass
    cdef cppclass ostream:
        pass

cdef extern from '<sstream>' namespace 'std':
    cdef cppclass istringstream(istream):
        istringstream(const string& str) except +
    cdef cppclass ostringstream(ostream):
        ostringstream() except +
        string str()

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
        double evaluate(double u, double v, double w, bool u_from_right, bool v_from_right, bool w_from_right) const
        void evaluate(vector[double]& results, double u, double v, double w, int derivs, bool u_from_right, bool v_from_right, bool w_from_right) const

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

cdef extern from 'LRSpline/LRSpline.h':
    cdef enum refinementStrategy_ 'refinementStrategy':
        LR_MINSPAN
        LR_FULLSPAN
        LR_STRUCTURED_MESH

cdef extern from 'LRSpline/LRSpline.h' namespace 'LR':
    cdef enum parameterEdge_ 'LR::parameterEdge':
        NONE
        WEST
        EAST
        SOUTH
        NORTH
        TOP
        BOTTOM
    cdef cppclass LRSpline_ 'LR::LRSpline':
        int dimension()
        int nVariate()
        int nBasisFunctions()
        int nElements()
        double startparam(int)
        double endparam(int)
        int order(int)
        void generateIDs()
        void getEdgeFunctions(vector[Basisfunction_*]& edgeFunctions, parameterEdge_ edge, int depth) const
        void getEdgeElements(vector[Element_*]& edgeElements, parameterEdge_ edge) const
        vector[Element_*].iterator elementBegin()
        vector[Element_*].iterator elementEnd()
        Element_* getElement(int i)
        Basisfunction_* getBasisfunction(int i)
        HashSet_iterator[Basisfunction_*] basisBegin()
        HashSet_iterator[Basisfunction_*] basisEnd()
        bool setControlPoints(vector[double]& cps)
        void rebuildDimension(int dimvalue)
        void refineBasisFunction(int i)
        void refineBasisFunction(const vector[int]& i)
        void refineElement(int i)
        void refineElement(const vector[int]& i)
        void refineByDimensionIncrease(const vector[double]& error, double beta)

cdef extern from 'LRSpline/LRSplineSurface.h' namespace 'LR':
    cdef cppclass LRSplineSurface_ 'LR::LRSplineSurface' (LRSpline_):
        LRSplineSurface() except +
        LRSplineSurface_* copy()
        void read(istream) except +
        void write(ostream) except +
        void point(vector[double]& pt, double u, double v, int iEl) const
        void point(vector[double]& pt, double u, double v, int iEl, bool u_from_right, bool v_from_right) const
        void point(vector[vector[double]]& pts, double u, double v, int derivs, int iEl) const
        void point(vector[vector[double]]& pts, double u, double v, int derivs, bool u_from_right, bool v_from_right, int iEl) const
        void getGlobalUniqueKnotVector(vector[double]& knot_u, vector[double]& knot_v) const
        Meshline_* insert_const_u_edge(double u, double start_v, double stop_v, int multiplicity)
        Meshline_* insert_const_v_edge(double v, double start_u, double stop_u, int multiplicity)
        void writePostscriptElements(ostream, int, int, bool, vector[int]*) const
        void writePostscriptMesh(ostream, bool, vector[int]*) const
        void writePostscriptMeshWithControlPoints(ostream, int, int) const
        vector[Meshline_*].iterator meshlineBegin()
        vector[Meshline_*].iterator meshlineEnd()
        int nMeshlines() const
        Meshline_* getMeshline(int i)


cdef class Basisfunction:

    cdef Basisfunction_* w

    def getId(self):
        return self.w.getId()

    def getControlPoint(self):
        cdef vector[double] data
        self.w.getControlPoint(data)
        return np.array(data)

    def nVariate(self):
        return self.w.nVariate()

    def evaluate(self, *args):
        cdef vector[double] results
        if len(args) == 4:
            u, v, ufr, vfr = args
            return self.w.evaluate(<double> u, <double> v, <bool> ufr, <bool> vfr)
        if len(args) == 5:
            u, v, d, ufr, vfr = args
            self.w.evaluate(results, <double> u, <double> v, <int> d, <bool> ufr, <bool> vfr)
            return np.array(results)
        if len(args) == 6:
            u, v, w, ufr, vfr, wfr = args
            return self.w.evaluate(<double> u, <double> v, <double> w, <bool> ufr, <bool> vfr, <bool> wfr)
        if len(args) == 7:
            u, v, w, d, ufr, vfr, wfr = args
            self.w.evaluate(results, <double> u, <double> v, <double> w, <int> d, <bool> ufr, <bool> vfr, <bool> wfr)
            return np.array(results)
        raise TypeError("evaluate() expected 4, 5, 6 or 7 arguments")


cdef class Element:

    cdef Element_* w

    def getId(self):
        return self.w.getId()

    def getDim(self):
        return self.w.getDim()

    def getParmin(self, i):
        return self.w.getParmin(i)

    def getParmax(self, i):
        return self.w.getParmax(i)

    def supportIter(self):
        cdef HashSet_iterator[Basisfunction_*] it = self.w.supportBegin()
        cdef HashSet_iterator[Basisfunction_*] end = self.w.supportEnd()
        while not it.equal(end):
            bf = Basisfunction()
            bf.w = deref(it)
            yield bf
            preinc(it)


cdef class Meshline:

    cdef Meshline_* w

    def is_spanning_u(self):
        return self.w.is_spanning_u()

    @property
    def const_par_(self):
        return self.w.const_par_

    @property
    def start_(self):
        return self.w.start_

    @property
    def stop_(self):
        return self.w.stop_

    @property
    def multiplicity_(self):
        return self.w.multiplicity_


cdef class parameterEdge:
    NONE   = parameterEdge_.NONE
    WEST   = parameterEdge_.WEST
    EAST   = parameterEdge_.EAST
    SOUTH  = parameterEdge_.SOUTH
    NORTH  = parameterEdge_.NORTH
    TOP    = parameterEdge_.TOP
    BOTTOM = parameterEdge_.BOTTOM


cdef class refinementStrategy:
    LR_MINSPAN = refinementStrategy_.LR_MINSPAN
    LR_FULLSPAN = refinementStrategy_.LR_FULLSPAN
    LR_STRUCTURED_MESH = refinementStrategy_.LR_STRUCTURED_MESH


cdef class LRSplineObject:

    cdef LRSpline_* w

    def __dealloc__(self):
        del self.w

    def nVariate(self):
        return self.w.nVariate()

    def dimension(self):
        return self.w.dimension()

    def rebuildDimension(self, int dim):
        self.w.rebuildDimension(dim)

    def nBasisFunctions(self):
        return self.w.nBasisFunctions()

    def nElements(self):
        return self.w.nElements()

    def startparam(self, int i):
        return self.w.startparam(i)

    def endparam(self, int i):
        return self.w.endparam(i)

    def order(self, int i):
        return self.w.order(i)

    def elementIter(self):
        cdef vector[Element_*].iterator it = self.w.elementBegin()
        cdef vector[Element_*].iterator end = self.w.elementEnd()
        while it != end:
            el = Element()
            el.w = deref(it)
            yield el
            preinc(it)

    def getElement(self, int i):
        el = Element()
        el.w = self.w.getElement(i)
        return el

    def getEdgeElementsIter(self, edge):
        cdef vector[Element_*] els
        self.w.getEdgeElements(els, edge)
        it = els.begin()
        while it != els.end():
            bf = Element()
            bf.w = deref(it)
            yield bf
            preinc(it)

    def basisIter(self):
        cdef HashSet_iterator[Basisfunction_*] it = self.w.basisBegin()
        cdef HashSet_iterator[Basisfunction_*] end = self.w.basisEnd()
        while not it.equal(end):
            bf = Basisfunction()
            bf.w = deref(it)
            yield bf
            preinc(it)

    def getBasisfunction(self, int i):
        bf = Basisfunction()
        bf.w = self.w.getBasisfunction(i)
        return bf

    def getEdgeFunctionsIter(self, edge, int depth=1):
        cdef vector[Basisfunction_*] bfs
        self.w.getEdgeFunctions(bfs, edge, depth)
        it = bfs.begin()
        while it != bfs.end():
            bf = Basisfunction()
            bf.w = deref(it)
            yield bf
            preinc(it)

    def setControlpoints(self, cpts):
        self.w.setControlPoints(cpts)

    def generateIDs(self):
        self.w.generateIDs()

    def refineBasisFunction(self, i):
        cdef vector[int] indices
        if isinstance(i, int):
            self.w.refineBasisFunction(<int> i)
        elif isinstance(i, list):
            indices = i
            self.w.refineBasisFunction(indices)

    def refineElement(self, i):
        cdef vector[int] indices
        if isinstance(i, int):
            self.w.refineElement(<int> i)
        elif isinstance(i, list):
            indices = i
            self.w.refineElement(indices)

    def refineByDimensionIncrease(self, errors, double beta):
        cdef vector[double] cpperrors
        cpperrors = errors
        self.w.refineByDimensionIncrease(errors, beta)


cdef class LRSurface(LRSplineObject):

    def __cinit__(self, w=None):
        self.w = new LRSplineSurface_()

    cdef _set_w(self, LRSplineSurface_* w):
        del self.w
        self.w = w

    def read(self, stream):
        cdef string cppstring
        if hasattr(stream, 'read'):
            stream = stream.read()
        if isinstance(stream, str):
            cppstring = stream.encode()
        elif isinstance(stream, bytes):
            cppstring = stream
        cdef istringstream* cppstream
        cppstream = new istringstream(cppstring)
        (<LRSplineSurface_*> self.w).read(deref(cppstream))
        del stream

    def write(self, stream):
        cdef ostringstream* cppstream
        cppstream = new ostringstream()
        (<LRSplineSurface_*> self.w).write(deref(cppstream))
        stream.write(cppstream.str())
        del cppstream

    def writePostscriptElements(self, stream, int nu=2, int nv=2, bool close=True, colorElements=None):
        cdef ostringstream* cppstream
        cdef vector[int] colors
        cppstream = new ostringstream()
        if colorElements is None:
            (<LRSplineSurface_*> self.w).writePostscriptElements(deref(cppstream), nu, nv, close, NULL)
        else:
            colors = colorElements
            (<LRSplineSurface_*> self.w).writePostscriptElements(deref(cppstream), nu, nv, close, &colors)
        stream.write(cppstream.str())
        del cppstream

    def copy(self):
        cdef LRSplineSurface_* copy = (<LRSplineSurface_*> self.w).copy()
        retval = LRSurface()
        retval._set_w(copy)
        return retval

    def getGlobalUniqueKnotVector(self):
        cdef vector[double] knots_u
        cdef vector[double] knots_v
        (<LRSplineSurface_*> self.w).getGlobalUniqueKnotVector(knots_u, knots_v)
        return np.array(knots_u), np.array(knots_v)

    def point(self, *args, iEl=-1):
        cdef vector[double] point
        cdef vector[vector[double]] results
        if len(args) == 2:
            u, v = args
            (<LRSplineSurface_*> self.w).point(point, u, v, iEl)
            return np.array(point)
        if len(args) == 4:
            u, v, ufr, vfr = args
            (<LRSplineSurface_*> self.w).point(point, u, v, <bool> ufr, <bool> vfr, iEl)
            return np.array(point)
        if len(args) == 3:
            u, v, d = args
            (<LRSplineSurface_*> self.w).point(results, u, v, d, iEl)
            return np.array(results)
        if len(args) == 5:
            u, v, d, ufr, vfr = args
            (<LRSplineSurface_*> self.w).point(results, u, v, d, iEl)
            return np.array(results)
        raise TypeError("point() expected 2, 3, 4 or 5 arguments")

    def meshlineIter(self):
        cdef vector[Meshline_*].iterator it = (<LRSplineSurface_*> self.w).meshlineBegin()
        cdef vector[Meshline_*].iterator end = (<LRSplineSurface_*> self.w).meshlineEnd()
        while it != end:
            ml = Meshline()
            ml.w = deref(it)
            yield ml
            preinc(it)

    def nMeshlines(self):
        return (<LRSplineSurface_*> self.w).nMeshlines()

    def getMeshline(self, int i):
        ml = Meshline()
        ml.w = (<LRSplineSurface_*> self.w).getMeshline(i)
        return ml

    def insert_const_u_edge(self, double u, double start_v, double stop_v, int multiplicity=1):
        (<LRSplineSurface_*> self.w).insert_const_u_edge(u, start_v, stop_v, multiplicity)

    def insert_const_v_edge(self, double v, double start_u, double stop_u, int multiplicity=1):
        (<LRSplineSurface_*> self.w).insert_const_v_edge(v, start_u, stop_u, multiplicity)
