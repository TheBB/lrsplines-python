from pytest import fixture
from pathlib import Path
import numpy as np
import lrspline as lr
from lrspline import raw
from splipy import BSplineBasis

path = Path(__file__).parent

@fixture
def srf3():
    p = [4,4]
    n = [7,7]
    ans = lr.LRSplineSurface(n[0], n[1], p[0], p[1])
    ans.basis[10].refine()
    return ans

@fixture
def srf():
    with open(path / 'mesh01.lr', 'rb') as f:
        return lr.LRSplineSurface(f)

@fixture
# quadratic function, random function refined three times
def srf2():
    p = np.array([3,3]) # order=3
    n = p + 7           # 8 elements
    srf = lr.LRSplineSurface(n[0], n[1], p[0], p[1])
    srf.basis[34].refine()
    srf.basis[29].refine()
    srf.basis[100].refine()
    with open('ex.eps', 'wb') as f:
        srf.write_postscript(f)
    return srf


def test_raw_constructors():
    srf = raw.LRSurface()
    assert srf.nBasisFunctions() == 0
    assert srf.nElements() == 0

    srf = raw.LRSurface(2,2,2,2)
    assert srf.nBasisFunctions() == 4
    assert srf.nElements() == 1

    srf = raw.LRSurface(n1=5, n2=4, order_u=3, order_v=2)
    assert srf.nBasisFunctions() == 20
    assert srf.nElements() == 9

    knot1 = [0,0,0,1,2,3,4,4,4]
    knot2 = [1,1,1,2,2,3,3,3]
    srf = raw.LRSurface(n1=6, n2=5, order_u=3, order_v=3, knot1=knot1, knot2=knot2)
    assert srf.nBasisFunctions() == 30
    assert srf.nElements() == 8

    knot1 = [0,0,0,1,2,3,4,4,4]
    knot2 = [1,1,1,2,2,3,3,3]
    # choose greville points as controlpoints gives linear mapping x(u,v) = u, y(u,v)=v
    cp = [[[(x0+x1)/2, (y0+y1)/2] for x0,x1 in zip(knot1[1:-3], knot1[2:-2])] for y0,y1 in zip(knot2[1:-3], knot2[2:-2])]
    cp = np.ndarray.flatten(np.array(cp))
    srf = raw.LRSurface(n1=6, n2=5, order_u=3, order_v=3, knot1=knot1, knot2=knot2, coef=cp)
    assert srf.nBasisFunctions() == 30
    assert srf.nElements() == 8

    ### the tests below result in segmentation fault. Most probably due to the
    #   defintion HAS_GOTOOLS which previously have caused issues with the
    #   point()-method

    # np.testing.assert_allclose(srf.point(0.123, 1.2), [0.123, 1.2])
    # np.testing.assert_allclose(srf.point(1.456, 2.2), [0.456, 2.2])
    # np.testing.assert_allclose(srf.point(3.199, 2.8), [3.199, 2.8])

def test_bezier_extraction():
    # single linear element: 4x4 identity matrix
    srf = raw.LRSurface(2,2,2,2)
    np.testing.assert_allclose(srf.getBezierExtraction(0), np.identity(4))

    # single quadratic element: 9x9 identity matrix
    srf = raw.LRSurface(3,3,3,3)
    np.testing.assert_allclose(srf.getBezierExtraction(0), np.identity(9))

    # two quadratic elements (Note that LR basisfunctions are "randomly" ordered
    # which means that the rows of this matrix can be permuted at a later point)
    srf = lr.LRSplineSurface(4,3,3,3)
    example_C = [[1,  0,  0,  0,  0,  0,  0,  0,  0, ],# the leftmost columns
                 [0,  0,  0,  1,  0,  0,  0,  0,  0, ],# should have exactly one 1
                 [0,  0,  0,  0,  0,  0,  1,  0,  0, ],
                 [0,  0,  0.5,0,  0,  0,  0,  0,  0, ],# the rightmost columns
                 [0,  0,  0,  0,  0,  0.5,0,  0,  0, ],# should have one 0.5's
                 [0,  0,  0,  0,  0,  0,  0,  0,  0.5],
                 [0,  1,  0.5,0,  0,  0,  0,  0,  0, ],# the center column
                 [0,  0,  0,  0,  1,  0.5,0,  0,  0, ],# should contain [1,.5]
                 [0,  0,  0,  0,  0,  0,  0,  1,  0.5]]

    C = srf.bezier_extraction(0)
    for i,bf in enumerate(srf.elements[0].support()):
        if bf[0][1] == 0.5: # rightmost function of left element
            assert np.sum(C[i]) == 0.5
            assert np.max(C[i]) == 0.5
            assert np.count_nonzero(C[i]) == 1
        elif bf[0][2] == 0.5: # center column
            assert np.sum(C[i]) == 1.5
            assert np.max(C[i]) == 1.0
            assert np.count_nonzero(C[i]) == 2
        else: # leftmost column
            assert np.sum(C[i]) == 1
            assert np.max(C[i]) == 1
            assert np.count_nonzero(C[i]) == 1

def test_bezier_evaluation(srf):
  b1 = BSplineBasis(srf.order(0))
  b2 = BSplineBasis(srf.order(1))
  nviz = 4
  # takes forever to test all elements, so to speed production, we only test the 11 first ones
  # for el in srf.elements:
  for el in [srf.elements[i] for i in range(11)]:
    B = el.bezier_extraction()
    cp = np.array([func.controlpoint for func in el.support()])
    u0 = np.array(el.start())
    u1 = np.array(el.end())
    de = u1-u0
    for u in np.linspace(0,1,nviz):
      for v in np.linspace(0,1,nviz):
        bez = np.kron(b2(v), b1(u))
        xi = u0 + de*np.array([u,v])
        val = bez @ B.T @ cp

        bezier_eval = np.ndarray.flatten(np.array(val))
        direct_eval = srf(*xi)

        np.testing.assert_allclose(direct_eval, bezier_eval)

def test_srf_from_file(srf):
    np.testing.assert_allclose(srf(0.0, 0.0), [0.0, 0.0])
    assert len(srf.basis) == 1229
    assert len(list(srf.basis.edge('south'))) == 15
    assert len(srf.elements) == 1300
    assert len(list(srf.elements.edge('south'))) == 14
    assert len(srf.meshlines) == 130


def test_evaluate(srf3):
    # testing identity mapping x(u,v) = u; y(u,v) = v
    np.testing.assert_allclose(srf3(0.123, 0.323), [0.123, 0.323])
    np.testing.assert_allclose(srf3(0.123, 0.323), [0.123, 0.323])
    np.testing.assert_allclose(srf3(0.987, 0.555), [0.987, 0.555])

    # testing vector evaluation
    u = np.linspace(0,1,13)
    v = np.linspace(0,1,13)
    result = srf3(u,v)
    assert result.shape == (13,2)

    # testing meshgrid evaluation
    U,V = np.meshgrid(u,v)
    result = srf3(U,V)
    assert result.shape == (13,13,2)


def test_derivative():
    srf = lr.LRSplineSurface(2, 2, 2, 2)

    # Single point, single derivative
    np.testing.assert_allclose(srf.derivative(0.123, 0.323, d=(0,0)), [0.123, 0.323])
    np.testing.assert_allclose(srf.derivative(0.123, 0.323, d=(1,0)), [1.0, 0.0])
    np.testing.assert_allclose(srf.derivative(0.123, 0.323, d=(0,1)), [0.0, 1.0])
    np.testing.assert_allclose(srf.derivative(0.123, 0.323, d=(1,1)), [0.0, 0.0])
    np.testing.assert_allclose(srf.basis[0].derivative(0.123, 0.323, d=(0,0)), 0.593729)
    np.testing.assert_allclose(srf.basis[1].derivative(0.123, 0.323, d=(1,0)), 0.677)
    np.testing.assert_allclose(srf.basis[2].derivative(0.123, 0.323, d=(0,1)), 0.877)
    np.testing.assert_allclose(srf.basis[3].derivative(0.123, 0.323, d=(1,1)), 1.0)

    # Multiple points, single derivative
    pt = (np.array([0.123, 0.821]), np.array([0.323, 0.571]))
    np.testing.assert_allclose(srf.derivative(*pt, d=(0,0)), [[0.123, 0.323], [0.821, 0.571]])
    np.testing.assert_allclose(srf.derivative(*pt, d=(1,0)), [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(srf.derivative(*pt, d=(0,1)), [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(srf.derivative(*pt, d=(1,1)), [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(srf.basis[0].derivative(*pt, d=(0,0)), [0.593729, 0.076791])
    np.testing.assert_allclose(srf.basis[1].derivative(*pt, d=(1,0)), [0.677, 0.429])
    np.testing.assert_allclose(srf.basis[2].derivative(*pt, d=(0,1)), [0.877, 0.179])
    np.testing.assert_allclose(srf.basis[3].derivative(*pt, d=(1,1)), [1.0, 1.0])

    # Single point, multiple derivatives
    np.testing.assert_allclose(
        srf.derivative(0.123, 0.323, d=[(0,0), (1,0), (0,1), (1,1)]),
        [[0.123, 0.323], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )
    np.testing.assert_allclose(
        srf.basis[3].derivative(0.123, 0.323, d=[(0,0), (1,0), (0,1), (1,1)]),
        [0.039729, 0.323, 0.123, 1.0]
    )

    # Multiple points, multiple derivatives
    np.testing.assert_allclose(
        srf.derivative(*pt, d=[(0,0), (1,0), (0,1), (1,1)]),
        [
            [[0.123, 0.323], [0.821, 0.571]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    np.testing.assert_allclose(
        srf.basis[3].derivative(*pt, d=[(0,0), (1,0), (0,1), (1,1)]),
        [
            [0.039729, 0.468791],
            [0.323, 0.571],
            [0.123, 0.821],
            [1.0, 1.0],
        ]
    )


def test_get_controlpoint():
    srf = lr.LRSplineSurface(3,3,3,3)
    # all controlpoints srf[i] should equal the greville absiccae
    for i,bf in enumerate(srf.basis):
        np.testing.assert_allclose(srf[i], [np.mean(bf[0][1:3]), np.mean(bf[1][1:3])])


def test_equality(srf):
    bf = srf.basis[0]
    for b in srf.elements[0].support():
        if b.id == 0:
            assert b == bf
        else:
            assert not b == bf


def test_element_at(srf):
    for el in srf.elements:
        midpoint = (np.array(el.start()) + np.array(el.end())) / 2.0
        el2 = srf.element_at(*midpoint)
        assert el == el2

    el1 = srf.elements[0]
    el2 = srf.elements[1]
    assert not el1 == el2

    pt = (np.array(el2.start()) + np.array(el2.end())) / 2.0
    el2 = srf.element_at(*pt)
    assert not el1 == el2

    pt = (np.array(el1.start()) + np.array(el1.end())) / 2.0
    el2 = srf.element_at(*pt)
    assert el1 == el2


def test_support(srf2):
    # check that all element -> basisfunction pointers are consistent
    for bf in srf2.basis:
        for el in bf.support():
            assert bf in el.support()

    # check that all basisfunction -> element pointers are consistent
    for el in srf2.elements:
        for bf in el.support():
            assert el in bf.support()

    # check that the 'in' call does as intended
    assert not srf2.basis[0] in srf2.elements[1].support()

