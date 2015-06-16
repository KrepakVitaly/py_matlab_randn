from distutils.core import setup, Extension
import os
import distutils.ccompiler as cc
import numpy

mkl_root_dir = os.environ.get('MKLROOT')

if mkl_root_dir is None:
    print 'MKLROOT environment variable is not defined. VSL version will not be installed'
    mkl_include_path = ''
    mkl_libs_path = ''
else:
    mkl_include_path = os.path.join(mkl_root_dir, 'include')
    mkl_libs_path = os.path.join(mkl_root_dir, 'lib', 'intel64')

if cc.get_default_compiler() is 'msvc':
    cflags = ['/Ox']
    vsl_compile_args = ['/openmp']
    vsl_link_args = ['mkl_intel_lp64.lib', 'mkl_core.lib', 'mkl_intel_thread.lib', 'libiomp5md.lib']
    omp_path = os.path.split(mkl_root_dir)[0]
    omp_path = os.path.join(omp_path, 'compiler', 'lib', 'intel64')
else:
    mkl_intel_lp64 = os.path.join(mkl_libs_path, 'libmkl_intel_lp64.a')
    mkl_core = os.path.join(mkl_libs_path, 'libmkl_core.a')
    mkl_gnu_thread = os.path.join(mkl_libs_path, 'libmkl_gnu_thread.a')
    vsl_compile_args = ['-fopenmp', '-m64']
    omp_path = ''
    vsl_link_args=['-Wl,--start-group', mkl_intel_lp64, mkl_core, mkl_gnu_thread,\
    '-Wl,--end-group', '-ldl','-lpthread', '-lgomp']
    cflags = ['-O3']

matlab_random = Extension('matlab_random', sources=['py_matlab_random.cpp', 'matlab_random.cpp'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=cflags,
                          language='c++')
vsl_random = Extension('vsl_random', sources=['py_vsl_random.cpp', 'vsl_random.cpp'],
                        include_dirs=[numpy.get_include(), mkl_include_path],
                        library_dirs=[mkl_libs_path, omp_path],
                        extra_compile_args=vsl_compile_args,
                        extra_link_args=vsl_link_args,
                        language='c++')

modules = [matlab_random]
if mkl_root_dir is not None:
    modules.append(vsl_random)

setup(name='random_number',
      version='0.0.2',
      ext_modules=modules)
