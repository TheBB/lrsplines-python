#!/usr/bin/env python

from setuptools import setup
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension
from subprocess import run
from os import path, makedirs
from Cython.Build import cythonize


LRSPLINES = path.abspath(path.join(path.dirname(__file__), 'submodules', 'LRSplines'))
BUILDPATH = path.join(LRSPLINES, 'build')


class CustomBuild(build_ext):

    def run(self):
        makedirs(BUILDPATH, exist_ok=True)
        run(['cmake', '..',
             '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_POSITION_INDEPENDENT_CODE=1',
             '-DBUILD_SHARED_LIBS=0'],
            cwd=BUILDPATH)
        run(['make'], cwd=BUILDPATH)
        super().run()


setup(
    name='LRSplines',
    version='0.0.1',
    description='Python bindings for the LRSplines library',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    ext_modules=cythonize(Extension(
        'lrsplines',
        ['lrsplines.pyx'],
        library_dirs=[path.join(BUILDPATH, 'lib')],
        include_dirs=[path.join(LRSPLINES, 'include', 'LRSpline')],
        libraries=['LRSpline'],
    )),
    install_requires=['numpy'],
    cmdclass={'build_ext': CustomBuild},
)
