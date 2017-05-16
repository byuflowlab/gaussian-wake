#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

module1 = Extension('_porteagel_fortran', sources=['src/gaussianwake/gaussianwake.f90', 'src/gaussianwake/adStack.c',
                                                   'src/gaussianwake/adBuffer.f'], extra_compile_args=['-O2', '-c'])

setup(
    name='GaussianWake',
    version='0.0.1',
    description='Gaussian wake model published by Bastankhah and Porte Agel 2016',
    install_requires=['openmdao>=1.7'],
    package_dir={'': 'src'},
    ext_modules=[module1],
    dependency_links=['http://github.com/OpenMDAO/OpenMDAO.git@master'],
    packages=['gaussianwake'],
    license='Apache License, Version 2.0',
)