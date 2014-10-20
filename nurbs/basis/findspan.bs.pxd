import numpy as np
cimport numpy as np
DTYPE = np.float64  
ctypedef np.float64_t DTYPE_t 
cpdef unsigned int findspan( unsigned int n, unsigned int p, double u, np.ndarray[DTYPE_t, ndim=1] U)
