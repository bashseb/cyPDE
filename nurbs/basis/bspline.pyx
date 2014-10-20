# file: nurbs.pyx
from __future__ import division
import numpy as np
cimport numpy as np
#@cython.boundscheck(False) # turn of bounds-checking for entire function 
# type of array     
DTYPE = np.float64  
ctypedef np.float64_t DTYPE_t 

DTYPEui = np.uint
ctypedef np.uint_t DTYPEui_t



include "findspan.pyx"
include "findspan_cast.pyx" # testing until I learn how to write proper ufuncs
include "basisfun.pyx"
include "basisfunder.pyx"
include "bspeval.pyx"
include "bspderiv.pyx"
# include "nrbeval.pyx"
