#!/usr/bin/env python

from pathlib import Path
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
    'submodules/LRSplines/src/MeshRectangle.cpp',
    'submodules/LRSplines/src/LRSpline.cpp',
    'submodules/LRSplines/src/LRSplineSurface.cpp',
    'submodules/LRSplines/src/LRSplineVolume.cpp',
]

with open(Path(__file__).parent / 'README.rst') as f:
    desc = f.read()


setup(
    name='LRSplines',
    version='1.5.0',
    description='Python bindings for the LRSplines library',
    long_description_content_type='text/x-rst',
    long_description=desc,
    keywords=['Splines', 'LR', 'Locally refined'],
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    license='GNU public license v3',
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
)
