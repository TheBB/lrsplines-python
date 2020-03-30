from pytest import fixture
from pathlib import Path
import numpy as np
import lrspline as lr


path = Path(__file__).parent

@fixture
def srf():
    with open(path / 'mesh01.lr', 'rb') as f:
        return lr.LRSplineSurface(f)


@fixture
def vol():
    with open(path / 'mesh02.lr', 'rb') as f:
        return lr.LRSplineVolume(f)


def test_srf_numbers(srf):
    np.testing.assert_allclose(srf(0.0, 0.0), [0.0, 0.0])
    assert len(srf.basis) == 1229
    assert len(list(srf.basis.edge('south'))) == 15
    assert len(srf.elements) == 1300
    assert len(list(srf.elements.edge('south'))) == 14
    assert len(srf.meshlines) == 130


def test_vol_numbers(vol):
    np.testing.assert_allclose(vol(0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
    assert len(vol.basis) == 1176
    assert len(list(vol.basis.edge('south'))) == 90
    assert len(vol.elements) == 1240
    assert len(list(vol.elements.edge('south'))) == 70
    assert len(vol.meshrects) == 189
