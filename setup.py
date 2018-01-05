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
    ext_modules=cythonize(Extension('lrsplines',
        ['lrsplines.pyx'],
        language="c++",
        include_dirs=['submodules/LRSplines/include/LRSpline'],
        extra_compile_args=["-DHAS_BOOST=1","-DHAS_GOTOOLS=1","-std=gnu++0x"],
        extra_link_args=["-DHAS_BOOST=1","-DHAS_GOTOOLS=1","-std=gnu++0x"],
        libraries=['LRSpline'],
    )),
    install_requires=['numpy'],
)
