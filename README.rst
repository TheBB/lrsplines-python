=========
LRSplines
=========

.. image:: https://badge.fury.io/py/LRSplines.svg
   :target: https://badge.fury.io/py/LRSplines

.. image:: https://travis-ci.org/TheBB/lrsplines-python.svg?branch=master
   :target: https://travis-ci.org/TheBB/lrsplines-python


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
