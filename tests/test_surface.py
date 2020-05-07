from pytest import fixture
from pathlib import Path
import numpy as np
import lrspline as lr
from lrspline import raw

path = Path(__file__).parent

@fixture
def srf():
    with open(path / 'mesh01.lr', 'rb') as f:
        return lr.LRSplineSurface(f)


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

def test_srf_from_file(srf):
    np.testing.assert_allclose(srf(0.0, 0.0), [0.0, 0.0])
    assert len(srf.basis) == 1229
    assert len(list(srf.basis.edge('south'))) == 15
    assert len(srf.elements) == 1300
    assert len(list(srf.elements.edge('south'))) == 14
    assert len(srf.meshlines) == 130


def test_evaluate():
    # testing identity mapping x(u,v) = u; y(u,v) = v
    srf = lr.LRSplineSurface(2,2,2,2)
    np.testing.assert_allclose(srf(0.123, 0.323), [0.123, 0.323])

    srf = lr.LRSplineSurface(6,5,4,3)
    np.testing.assert_allclose(srf(0.123, 0.323), [0.123, 0.323])
    np.testing.assert_allclose(srf(0.987, 0.555), [0.987, 0.555])


def test_get_controlpoint():
    srf = lr.LRSplineSurface(3,3,3,3)
    # all controlpoints srf[i] should equal the greville absiccae
    for i,bf in enumerate(srf.basis):
        np.testing.assert_allclose(srf[i], [np.mean(bf[0][1:3]), np.mean(bf[1][1:3])])
