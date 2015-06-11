# py_matlab_randn
Implementation of various random number generators for Python
Currently there is:
1) MATLAB's randn() for normal distribution
2) Parallel generator that uses Intel's VSL

###### Installation:
1. Set up MKLROOT environment variable if you want VSL version to be installed. This variable should contain a path to Intel MKL library.
2. run 'python setup.py install'


###### Usage:
There are two modules installed matlab_random and vsl_random.

matlab_random.randn(shape) - generate normally distributed numbers unsing sequential MATLAB algorithm

  shape - a tuple indicating shape of the array, or an integer indicating size of 1D array
  
matlab_random.rng(seed) - seed the MATLAB's generator

  seed - integer to seed the generator


vsl_random.randn(shape) - generate normally distributed numbers using parallel VSL and openMP algorithms

  shape - a tuple indicating shape of the array, or an integer indicating size of 1D array
  
vsl_random.randn(shape) - generate numbers uniformly distributed over [0, 1) using parallel VSL and openMP algorithms

  shape - a tuple indicating shape of the array, or an integer indicating size of 1D array
  
vsl_random.rng(seed) - seed the generator

  seed - integer to seed the generator
