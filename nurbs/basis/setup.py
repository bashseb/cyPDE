# setup.py 
from distutils.core import setup
from Cython.Build import cythonize
 
setup(
  name = 'Bspline functions',
  ext_modules = cythonize("bspline.pyx")
  # ext_modules = cythonize("*.pyx")
)
