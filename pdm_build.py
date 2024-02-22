from pathlib import Path
from setuptools import Extension
from Cython.Build import cythonize

def pdm_build_update_setup_kwargs(context, setup_kwargs):
    lrsplines_path = Path('submodules') / 'LRSplines'
    lrsplines_src_path = lrsplines_path / 'src'
    lrsplines_cpp = [
        lrsplines_src_path / 'Basisfunction.cpp',
        lrsplines_src_path / 'Element.cpp',
        lrsplines_src_path / 'Meshline.cpp',
        lrsplines_src_path / 'MeshRectangle.cpp',
        lrsplines_src_path / 'LRSpline.cpp',
        lrsplines_src_path / 'LRSplineSurface.cpp',
        lrsplines_src_path / 'LRSplineVolume.cpp',
    ]

    extensions = [
        Extension(
            'lrsplines',
            ['lrsplines.pyx', *map(str, lrsplines_cpp)],
            include_dirs=[str(lrsplines_path / 'include')],
            extra_compile_args=['-std=c++11'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
        Extension(
            'lrspline.raw',
            ['lrspline/raw.pyx', *map(str, lrsplines_cpp)],
            include_dirs=[str(lrsplines_path / 'include')],
            extra_compile_args=['-std=c++11'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
    ]

    setup_kwargs.update(
        ext_modules=cythonize(extensions),
        packages=["lrspline"],
    )
