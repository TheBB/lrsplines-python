#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension
from subprocess import run
from os import path, makedirs
import sys


# Path of .pyx files, without extension
EXTENSION_FILES = ['lrsplines', 'lrspline/raw']


# If Cython is installed AND all pyx files are included, use Cython.
# This will generate C++ sources at install/build time.  Cython
# sources are not included in a source package, and we don't rely on
# Cython being available in the installing environment anyway.
if all(path.exists(fn + '.pyx') for fn in EXTENSION_FILES):
    try:
        from Cython.Build import cythonize
        HAS_CYTHON = True
    except ImportError:
        HAS_CYTHON = False
else:
    HAS_CYTHON = False


# If we're not using Cython, the cythonized sources must be in the
# file tree.  This is taken care of by sdist (see MANIFEST.in)
if not HAS_CYTHON:
    try:
        assert all(path.exists(fn + '.cpp') for fn in EXTENSION_FILES)
    except AssertionError:
        print("Could not find either cython or cythonized source files.", file=sys.stderr)
        raise


# Specify the C++ library files that must also be included
LRSPLINES_PATH = path.abspath(path.join(path.dirname(__file__), 'submodules', 'LRSplines'))
LRSPLINES_SOURCES = [
    path.join(LRSPLINES_PATH, 'src', 'Basisfunction.cpp'),
    path.join(LRSPLINES_PATH, 'src', 'Element.cpp'),
    path.join(LRSPLINES_PATH, 'src', 'Meshline.cpp'),
    path.join(LRSPLINES_PATH, 'src', 'MeshRectangle.cpp'),
    path.join(LRSPLINES_PATH, 'src', 'LRSpline.cpp'),
    path.join(LRSPLINES_PATH, 'src', 'LRSplineSurface.cpp'),
    path.join(LRSPLINES_PATH, 'src', 'LRSplineVolume.cpp'),
]


# Create extension objects, using either pyx or cpp file extension as required
EXTENSIONS = [
    Extension(
        'lrsplines',
        ['lrsplines.' + ('pyx' if HAS_CYTHON else 'cpp'), *LRSPLINES_SOURCES],
        include_dirs=[path.join(LRSPLINES_PATH, 'include')],
        extra_compile_args=['-std=c++11'],
    ),
    Extension(
        'lrspline.raw',
        ['lrspline/raw.' + ('pyx' if HAS_CYTHON else 'cpp'), *LRSPLINES_SOURCES],
        include_dirs=[path.join(LRSPLINES_PATH, 'include')],
        extra_compile_args=['-std=c++11'],
    ),
]

if HAS_CYTHON:
    EXTENSIONS = cythonize(EXTENSIONS)


with open(Path(__file__).parent / 'README.rst') as f:
    desc = f.read()


setup(
    name='LRSplines',
    version='1.9.0',
    description='Python bindings for the LRSplines library',
    long_description_content_type='text/x-rst',
    long_description=desc,
    keywords=['Splines', 'LR', 'Locally refined'],
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    license='GNU public license v3',
    packages=find_packages(),
    ext_modules=EXTENSIONS,
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
)
