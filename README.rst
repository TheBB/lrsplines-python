=========
LRSplines
=========

.. image:: https://badge.fury.io/py/LRSplines.svg
   :target: https://badge.fury.io/py/LRSplines

.. image:: https://github.com/TheBB/lrsplines-python/workflows/Python%20package/badge.svg?branch=master
   :target: https://github.com/TheBB/lrsplines-python/actions?query=workflow%3A%22Python+package%22+branch%3Amaster


This is a cython-based Python wrapper around the `LRSplines
<https://github.com/VikingScientist/LRsplines>`_ library, a C++ implementation
of locally refined B-splines.


Installing
----------

LRSplines is available on PyPi.::

    pip install lrsplines


Usage
-----

There are two modules for interacting with LRSplines.

- The ``lrspline.raw`` module contains an API that is identical (so
  far as feasible) to the C++ interface of the backend library.

- The ``lrspline`` module contains a Pythonic interface to the same
  library.

In addition, the ``lrsplines`` module is available, but is deprecated
and should not be used.
