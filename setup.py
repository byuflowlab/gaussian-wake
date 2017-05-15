#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

module1 = Extension('_porteagel_fortran', sources=['src/gaussianwake/gaussianwake.f90'], extra_compile_args=['-O2', '-c'])

setup(
    name='GaussianWake',
    version='0.0.1',
    description='Wind farm optimization interface allowing wake models to be switched out',
    install_requires=['openmdao>=1.6.3'],
    package_dir={'': 'src'},
    ext_modules=[module1],
    dependency_links=['http://github.com/OpenMDAO/OpenMDAO.git@master'],
    packages=['gaussianwake'],
    license='Apache License, Version 2.0',
)