#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='LRSplines',
    version='0.0.1',
    description='Python bindings for the LRSplines library',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    ext_modules=cythonize(Extension(
        'lrsplines',
        ['lrsplines.pyx'],
        include_dirs=['submodules/LRSplines/include/LRSpline'],
        libraries=['LRSpline'],
    )),
    install_requires=['numpy'],
)
