=========
LRSplines
=========

This is a cython-based Python wrapper around the `LRSplines
<https://github.com/VikingScientist/LRsplines>`_ library, a C++ implementation
of locally refined B-splines.


Dependencies
------------

You will need a Python installation with numpy and cython, as well as LRSplines
itself installed.

Installing
----------

Make sure the submodules are updated, i.e. ``git submodule init`` followed by
``git submodule update``. To install, use::

    pip install .
