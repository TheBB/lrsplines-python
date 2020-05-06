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

def test_srf_from_file(srf):
    np.testing.assert_allclose(srf(0.0, 0.0), [0.0, 0.0])
    assert len(srf.basis) == 1229
    assert len(list(srf.basis.edge('south'))) == 15
    assert len(srf.elements) == 1300
    assert len(list(srf.elements.edge('south'))) == 14
    assert len(srf.meshlines) == 130

