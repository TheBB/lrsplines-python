import unittest
import pytest
import numpy as np
import lrspline as lr
import numpy as np
from splipy import BSplineBasis
from scipy import linalg

def get_volume(p, n, nref):
  vol    = lr.LRSplineVolume(*n, *p)
  for i in range(nref):
    el = vol.element_at(.24, .32, .16)
    el2 = vol.element_at(.74, .62, .16)
    functions = [func for func in el.support()] + [func for func in el2.support()]
    vol.refine(functions)
  return vol

def get_bezier_basis(p):
  b1 = BSplineBasis(p[0])
  b2 = BSplineBasis(p[1])
  b3 = BSplineBasis(p[2])
  return (b1,b2,b3)

def get_all_bezier_evaluations(p,nviz):
  b1 = BSplineBasis(p[0])
  b2 = BSplineBasis(p[1])
  b3 = BSplineBasis(p[2])
  result = []
  for u in np.linspace(0,1,nviz):
    for v in np.linspace(0,1,nviz):
      for w in np.linspace(0,1,nviz):
        bez = np.kron(b3(w), np.kron(b2(v), b1(u)))
        result += [bez]
  return result

def eval_bezier(spline, bezier, nviz):
  for el in spline.elements:
    cp = np.array([func.controlpoint for func in el.support()])
    B = el.bezier_extraction()
    k = 0
    for u in np.linspace(0,1,nviz):
      for v in np.linspace(0,1,nviz):
        for w in np.linspace(0,1,nviz):
          bez = bezier[k]
          k += 1
          value = bez @ B.T @ cp

def eval_call(spline, nviz):
  for el in spline.elements:
    for u in np.linspace(el.start(0), el.end(0), nviz):
      for v in np.linspace(el.start(1), el.end(1), nviz):
        for w in np.linspace(el.start(2), el.end(2), nviz):
          value = spline(u,v,w)

def eval_single_element_bezier(spline, bezier, nviz):
  el = np.random.choice(spline.elements)
  B = el.bezier_extraction()
  cp = np.array([func.controlpoint for func in el.support()])
  for u in np.linspace(0,1,nviz):
    for v in np.linspace(0,1,nviz):
      for w in np.linspace(0,1,nviz):
        bez = np.kron(bezier[2](w), np.kron(bezier[1](v), bezier[0](u)))
        value = bez @ B.T @ cp

def eval_single_element_call(spline, nviz):
  el = np.random.choice(spline.elements)
  for u in np.linspace(el.start(0), el.end(0), nviz):
    for v in np.linspace(el.start(1), el.end(1), nviz):
      for w in np.linspace(el.start(2), el.end(2), nviz):
        value = spline(u,v,w)

@pytest.mark.benchmark(group="eval-vol-single-nviz5")
def test_eval_single_nviz5(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 2)
  benchmark(eval_single_element_call, spline, 5)

@pytest.mark.benchmark(group="eval-vol-single-nviz5")
def test_bezier_single_nviz5(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 2)
  bezier = get_bezier_basis(p)
  benchmark(eval_single_element_bezier, spline, bezier, 5)

@pytest.mark.benchmark(group="eval-vol-single-nviz2")
def test_eval_single_nviz2(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 2)
  benchmark(eval_single_element_call, spline, 2)

@pytest.mark.benchmark(group="eval-vol-single-nviz2")
def test_bezier_single_nviz2(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 2)
  bezier = get_bezier_basis(p)
  benchmark(eval_single_element_bezier, spline, bezier, 2)

@pytest.mark.benchmark(group="eval-vol-nviz2")
def test_eval_nviz2(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 2)
  benchmark(eval_call, spline, 2)

@pytest.mark.benchmark(group="eval-vol-nviz2")
def test_bezier_nviz2(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 2)
  bezier = get_all_bezier_evaluations(p, 2)
  benchmark(eval_bezier, spline, bezier, 2)

@pytest.mark.benchmark(group="eval-vol-nviz5")
def test_eval_nviz5(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 1)
  benchmark(eval_call, spline, 5)

@pytest.mark.benchmark(group="eval-vol-nviz5")
def test_bezier_nviz5(benchmark):
  p = np.array([3,3,3]) # quadratic functions (this is order, polynomial degree+1)
  n = p + 3             # number of basis functions, start with 4x4 elements
  spline = get_volume(p,n, 1)
  bezier = get_all_bezier_evaluations(p, 5)
  benchmark(eval_bezier, spline, bezier, 5)
