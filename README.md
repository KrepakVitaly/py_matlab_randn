# py_matlab_randn
Implementation of MATLAB's randn function for Python

###### Installation:
1. Open random.prj with MATLAB and generate static library and C sources with MATLAB Coder.
2. Go to "Build" tab. (#1 on the picture)
3. Make shure library name is "libmlrandn" (#2 on the picture)
4. Press the "Build button" (#3 on the picture)
5. Run 'python setup.py install'. The pacakge will be installed as matlab_randn module.
![](http://oi57.tinypic.com/jp8uu9.jpg)


###### Usage:
matlab_randn.randn(shape, use_seed=False, seed=0)

  shape - a tuple indicating shape of the array, or an integer indicating size of 1D array
  
  use_seed - use specified seed (default value is False)
  
  seed - seed for generator (default value is 0)
