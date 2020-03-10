#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension
from subprocess import run
from os import path, makedirs
from Cython.Build import cythonize


LRSPLINES = path.abspath(path.join(path.dirname(__file__), 'submodules', 'LRSplines'))
SOURCES = [
    'submodules/LRSplines/src/Basisfunction.cpp',
    'submodules/LRSplines/src/Element.cpp',
    'submodules/LRSplines/src/Meshline.cpp',
    'submodules/LRSplines/src/LRSpline.cpp',
    'submodules/LRSplines/src/LRSplineSurface.cpp',
]


setup(
    name='LRSplines',
    version='0.1',
    description='Python bindings for the LRSplines library',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=find_packages(),
    ext_modules=cythonize([
        Extension(
            'lrsplines',
            ['lrsplines.pyx', *SOURCES],
            include_dirs=[path.join(LRSPLINES, 'include')],
            extra_compile_args=['-std=c++11'],
        ),
        Extension(
            'lrspline.raw',
            ['lrspline/raw.pyx', *SOURCES],
            include_dirs=[path.join(LRSPLINES, 'include')],
            extra_compile_args=['-std=c++11'],
        )
    ]),
    install_requires=['numpy'],
)
