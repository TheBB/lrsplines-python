from __future__ import annotations

from pathlib import Path

import numpy as np
from pytest import fixture

import lrspline as lr
from lrspline import raw

path = Path(__file__).parent


@fixture
def vol3():
    p = [4, 4, 4]
    n = [7, 7, 7]
    ans = lr.LRSplineVolume(n[0], n[1], n[2], p[0], p[1], p[2])
    ans.basis[83].refine()
    return ans


@fixture
def vol():
    with (path / "mesh02.lr").open("rb") as f:
        return lr.LRSplineVolume(f)


def test_raw_constructors():
    vol = raw.LRVolume()
    assert vol.nBasisFunctions() == 0
    assert vol.nElements() == 0

    vol = raw.LRVolume(2, 2, 2, 2, 2, 2)
    assert vol.nBasisFunctions() == 8
    assert vol.nElements() == 1

    vol = raw.LRVolume(n1=5, n2=4, n3=3, order_u=3, order_v=2, order_w=1)
    assert vol.nBasisFunctions() == 60
    assert vol.nElements() == 27

    knot1 = [0, 0, 0, 1, 2, 3, 4, 4, 4]
    knot2 = [1, 1, 1, 2, 2, 3, 3, 3]
    knot3 = [2, 2, 2, 3, 4, 4, 4]
    vol = raw.LRVolume(
        n1=6, n2=5, n3=4, order_u=3, order_v=3, order_w=3, knot1=knot1, knot2=knot2, knot3=knot3
    )
    assert vol.nBasisFunctions() == 120
    assert vol.nElements() == 16

    knot1 = [0, 0, 0, 1, 2, 3, 4, 4, 4]
    knot2 = [2, 2, 2, 4, 6, 6, 6]
    knot3 = [4, 4, 4, 8, 8, 8]

    rng = np.random.default_rng()
    cp = list(rng.random(216))
    vol = raw.LRVolume(
        n1=6, n2=4, n3=3, order_u=3, order_v=3, order_w=3, knot1=knot1, knot2=knot2, knot3=knot3, coef=cp
    )
    assert vol.nBasisFunctions() == 72
    assert vol.nElements() == 8


def test_vol_from_file(vol):
    np.testing.assert_allclose(vol(0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
    assert len(vol.basis) == 1176
    assert len(list(vol.basis.edge("south"))) == 90
    assert len(vol.elements) == 1240
    assert len(list(vol.elements.edge("south"))) == 70
    assert len(vol.meshrects) == 189


def test_element_at(vol):
    for el in vol.elements:
        midpoint = (np.array(el.start()) + np.array(el.end())) / 2.0
        el2 = vol.element_at(*midpoint)
        assert el == el2

    el1 = vol.elements[0]
    el2 = vol.elements[1]
    assert el1 != el2

    pt = (np.array(el2.start()) + np.array(el2.end())) / 2.0
    el2 = vol.element_at(*pt)
    assert el1 != el2

    pt = (np.array(el1.start()) + np.array(el1.end())) / 2.0
    el2 = vol.element_at(*pt)
    assert el1 == el2


def test_evaluate(vol3):
    # testing identity mapping x(u,v,w) = u; y(u,v,w) = v; z(u,v,w) = w
    np.testing.assert_allclose(vol3(0.123, 0.323, 0.456), [0.123, 0.323, 0.456])
    np.testing.assert_allclose(vol3(0.123, 0.323, 0.872), [0.123, 0.323, 0.872])
    np.testing.assert_allclose(vol3(0.987, 0.555, 0.622), [0.987, 0.555, 0.622])

    # testing vector evaluation
    u = np.linspace(0, 1, 7)
    v = np.linspace(0, 1, 7)
    w = np.linspace(0, 1, 7)
    result = vol3(u, v, w)
    assert result.shape == (7, 3)

    # testing meshgrid evaluation
    U, V, W = np.meshgrid(u, v, w)
    result = vol3(U, V, W)
    assert result.shape == (7, 7, 7, 3)


def test_derivative():
    vol = lr.LRSplineVolume(2, 2, 2, 2, 2, 2)

    # Single point, single derivative
    np.testing.assert_allclose(vol.derivative(0.123, 0.323, 0.456, d=(0, 0, 0)), [0.123, 0.323, 0.456])
    np.testing.assert_allclose(vol.derivative(0.123, 0.323, 0.456, d=(1, 0, 0)), [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(vol.derivative(0.123, 0.323, 0.456, d=(0, 1, 0)), [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(vol.derivative(0.123, 0.323, 0.456, d=(0, 0, 1)), [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(vol.basis[7].derivative(0.123, 0.323, 0.456, d=(0, 0, 0)), 0.018116424)
    np.testing.assert_allclose(vol.basis[7].derivative(0.123, 0.323, 0.456, d=(1, 0, 0)), 0.147288)
    np.testing.assert_allclose(vol.basis[7].derivative(0.123, 0.323, 0.456, d=(0, 1, 0)), 0.056088)
    np.testing.assert_allclose(vol.basis[7].derivative(0.123, 0.323, 0.456, d=(0, 0, 1)), 0.039729)

    # Multiple points, single derivative
    pt = (np.array([0.123, 0.821]), np.array([0.323, 0.571]), np.array([0.456, 0.617]))
    np.testing.assert_allclose(
        vol.derivative(*pt, d=(0, 0, 0)), [[0.123, 0.323, 0.456], [0.821, 0.571, 0.617]]
    )
    np.testing.assert_allclose(
        vol.derivative(*pt, d=(1, 0, 0)), [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-15
    )
    np.testing.assert_allclose(
        vol.derivative(*pt, d=(0, 1, 0)), [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-15
    )
    np.testing.assert_allclose(
        vol.derivative(*pt, d=(0, 0, 1)), [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1e-15
    )
    np.testing.assert_allclose(vol.basis[7].derivative(*pt, d=(0, 0, 0)), [0.018116424, 0.289244047])
    np.testing.assert_allclose(vol.basis[7].derivative(*pt, d=(1, 0, 0)), [0.147288, 0.352307])
    np.testing.assert_allclose(vol.basis[7].derivative(*pt, d=(0, 1, 0)), [0.056088, 0.506557])
    np.testing.assert_allclose(vol.basis[7].derivative(*pt, d=(0, 0, 1)), [0.039729, 0.468791])

    # Single point, multiple derivatives
    np.testing.assert_allclose(
        vol.derivative(0.123, 0.323, 0.456, d=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        [
            [0.123, 0.323, 0.456],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        vol.basis[7].derivative(0.123, 0.323, 0.456, d=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        [0.018116424, 0.147288, 0.056088, 0.039729],
    )

    # Multiple points, multiple derivatives
    np.testing.assert_allclose(
        vol.derivative(*pt, d=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        [
            [[0.123, 0.323, 0.456], [0.821, 0.571, 0.617]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        vol.basis[7].derivative(*pt, d=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        [
            [0.018116424, 0.289244047],
            [0.147288, 0.352307],
            [0.056088, 0.506557],
            [0.039729, 0.468791],
        ],
    )
