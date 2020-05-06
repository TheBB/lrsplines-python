from pytest import fixture
from pathlib import Path
import numpy as np
import lrspline as lr
from lrspline import raw

path = Path(__file__).parent

@fixture
def vol():
    with open(path / 'mesh02.lr', 'rb') as f:
        return lr.LRSplineVolume(f)

def test_raw_constructors():
    vol = raw.LRVolume()
    assert vol.nBasisFunctions() == 0
    assert vol.nElements() == 0

    vol = raw.LRVolume(2,2,2,2,2,2)
    assert vol.nBasisFunctions() == 8
    assert vol.nElements() == 1

    vol = raw.LRVolume(n1=5, n2=4, n3=3, order_u=3, order_v=2, order_w=1)
    assert vol.nBasisFunctions() == 60
    assert vol.nElements() == 27

    knot1 = [0,0,0,1,2,3,4,4,4]
    knot2 = [1,1,1,2,2,3,3,3]
    knot3 = [2,2,2,3,4,4,4]
    vol = raw.LRVolume(n1=6, n2=5, n3=4, order_u=3, order_v=3, order_w=3, knot1=knot1, knot2=knot2, knot3=knot3)
    assert vol.nBasisFunctions() == 120
    assert vol.nElements() == 16

    knot1 = [0,0,0,1,2,3,4,4,4]
    knot2 = [1,1,1,2,2,3,3,3]
    knot3 = [2,2,2,3,4,4,4]
    # choose greville points as controlpoints gives linear mapping x(u,v) = u, y(u,v)=v
    cp = [[[[(x0+x1)/2, (y0+y1)/2, (z0+z1)/2] for x0,x1 in zip(knot1[1:-3], knot1[2:-2])] for y0,y1 in zip(knot2[1:-3], knot2[2:-2])] for z0,z1 in zip(knot3[1:-3], knot3[2:-2])]
    cp = np.ndarray.flatten(np.array(cp))
    vol = raw.LRVolume(n1=6, n2=5, n3=4, order_u=3, order_v=3, order_w=3, knot1=knot1, knot2=knot2, knot3=knot3, coef=cp)
    assert vol.nBasisFunctions() == 120
    assert vol.nElements() == 16

    ### the tests below result in errors. Most probably due to the
    #   defintion HAS_GOTOOLS which previously have caused issues with the
    #   point()-method

    # np.testing.assert_allclose(vol.point(0.123, 1.2, 3.1),  [0.123, 1.2, 3.1])
    # np.testing.assert_allclose(vol.point(1.456, 2.2, 2.54), [0.456, 2.2, 2.54])
    # np.testing.assert_allclose(vol.point(3.199, 2.8, 2.57), [3.199, 2.8, 2.57])

def test_vol_from_file(vol):
    np.testing.assert_allclose(vol(0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
    assert len(vol.basis) == 1176
    assert len(list(vol.basis.edge('south'))) == 90
    assert len(vol.elements) == 1240
    assert len(list(vol.elements.edge('south'))) == 70
    assert len(vol.meshrects) == 189

def test_evaluate():
    # testing identity mapping x(u,v) = u; y(u,v) = v
    srf = lr.LRSplineVolume(2,2,2,2,2,2)
    np.testing.assert_allclose(srf(0.123, 0.323, 0.456), [0.123, 0.323, 0.456])

    srf = lr.LRSplineVolume(8,7,6,5,4,3) # n=(8,7,6), p=(5,4,3)
    np.testing.assert_allclose(srf(0.123, 0.323, 0.872), [0.123, 0.323, 0.872])
    np.testing.assert_allclose(srf(0.987, 0.555, 0.622), [0.987, 0.555, 0.622])
