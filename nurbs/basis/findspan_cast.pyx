# function s = findspan(n,p,u,U)                 
# FINDSPAN  Find the span of a B-Spline knot vector at a parametric point 
# ------------------------------------------------------------------------- 
# ADAPTATION of FINDSPAN from C 
# ------------------------------------------------------------------------- 
# 
# Calling Sequence: 
#  
#   s = findspan(n,p,u,U) 
#  
#  INPUT: 
#  
#    n - number of control points - 1 
#    p - spline degree 
#    u - parametric point 
#    U - knot sequence 
#  
#  RETURN: 
#  
#    s - knot span 
#  
#  Algorithm A2.1 from 'The NURBS BOOK' pg68 
                                                 
# wrapper until I learn how to write unit functions
cpdef np.ndarray[DTYPEui_t, ndim=1] _findspan( unsigned int n, unsigned int p, np.ndarray[DTYPE_t, ndim=1] u, np.ndarray[DTYPE_t, ndim=1] U):    # int findspan(int n, int p, double u, double *U) { 
	assert U.dtype == DTYPE
	assert u.dtype == DTYPE

	cdef np.ndarray[DTYPEui_t, ndim=1] retv = np.zeros(u.size, dtype=DTYPEui)
	cdef unsigned int i
	
	for i in range(u.size):
		retv[i] = findspan(n, p, u[i], U)

	return retv                                        

