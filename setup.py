
import os

try:
    from setuptools import setup, convert_path
except ImportError:
    from distutils.core import setup
    from distutils.util import convert_path

setup(name='difflikelihoods',
      version='0.99dev',
      packages=['difflikelihoods'],
      author='Nicholas Kraemer, Hans Kersting',
      description='Differentiable likelihoods for fast inversion of likelihood-free dynamical systems',
      license='LICENSE.txt',
      long_description=open('README.md').read())