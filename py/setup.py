# import os
# from os.path import join as pjoin
# from setuptools import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import subprocess
# import numpy as np


# try:
#     numpy_include = np.get_include()
# except AttributeError:
#     numpy_include = np.get_numpy_include()


# class custom_build_ext(build_ext):
#     def build_extensions(self):
#         build_ext.build_extensions(self)

# ext_modules = [
#     Extension(
#         name='cython_t',
#         sources=[
#             'cython_t.pyx'
#         ],
#         extra_compile_args=[
#             '-Wno-cpp'
#         ],
#         include_dirs=[
#             numpy_include
#         ]
#     ),
# ]

# setup(
#     name='fast_rcnn',
#     ext_modules=ext_modules,
#     cmdclass={'build_ext': custom_build_ext},
# )


##################### from Detectron #######################3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()


# Extension modules
ext_modules = [
    Extension(
        name='cython_bbox',
        sources=[
            'cython_bbox.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
]

setup(
    name='Detectron',
    packages=['.'],
    ext_modules=cythonize(ext_modules)
)
