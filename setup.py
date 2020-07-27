"""
setup.py

The basic outline is stolen from something written
by Ivan Oreshnikov.

Sorry Ivan :)
"""

import os

try:
    from setuptools import setup, convert_path
except ImportError:
    from distutils.core import setup
    from distutils.util import convert_path

setup(name='odefilters',
      version='0.99dev',
      packages=['odefilters'],
      author='Nicholas Kraemer, Hans Kersting',
      description='Probabilistic ODE Solver via Bayesian Filtering',
      license='LICENSE.txt',
      long_description=open('README.md').read())