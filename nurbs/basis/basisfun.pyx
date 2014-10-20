# function N = basisfun(i,u,p,U)                 
# BASISFUN  Basis # function for B-Spline 
# ------------------------------------------------------------------------- 
# ADAPTATION of BASISFUN from C Routine 
# ------------------------------------------------------------------------- 
# 
# Calling Sequence: 
#  
#   N = basisfun(i,u,p,U) 
#    
#    INPUT: 
#    
#      i - knot span  ( from FindSpan() ) 
#      u - parametric point 
#      p - spline degree 
#      U - knot sequence 
#    
#    OUTPUT: 
#    
#      N - Basis functions vector[p+1] 
#    
#    Algorithm A2.2 from 'The NURBS BOOK' pg70. 
                                                 
#   void basisfun(int i, double u, int p, double *U, double *N) { 
cpdef np.ndarray[DTYPE_t, ndim=1] basisfun(unsigned int i, double u, unsigned int p, np.ndarray[DTYPE_t, ndim=1]  U):
	assert U.dtype == DTYPE
    
	cdef unsigned int j, r									#   int j,r; 
	cdef double saved, temp									#   double saved, temp; 

															#   double *left  = (double*) mxMalloc((p+1)*sizeof(double)); 
	cdef np.ndarray[DTYPE_t, ndim=1] left = np.zeros(p+1, dtype=DTYPE)
															#   double *right = (double*) mxMalloc((p+1)*sizeof(double)); 
	cdef np.ndarray[DTYPE_t, ndim=1] right = np.zeros(p+1, dtype=DTYPE)

	cdef np.ndarray[DTYPE_t, ndim=1] N = np.zeros(p+1, dtype=DTYPE)
                                                
	N[0] = 1.                                     			#   N[0] = 1.0; 
	for j in range(1,p+1):                        			#   for (j = 1; j <= p; j++) { 
		left[j] = u - U[i+1-j]                    			#   left[j]  = u - U[i+1-j]; 
		right[j] = U[i+j] - u                     			#   right[j] = U[i+j] - u; 
		saved = 0.                                			#   saved = 0.0; 
		for r in range(j):									#   for (r = 0; r < j; r++) { 
			temp = N[r] / (right[r+1] + left[j-r])			#   temp = N[r] / (right[r+1] + left[j-r]); 
			N[r] = saved + right[r+1] * temp      			#   N[r] = saved + right[r+1] * temp;
			saved = left[j-r] * temp              			#   saved = left[j-r] * temp;
	                                              			#   } 
		N[j] = saved                              			#   N[j] = saved; 
															#   } 

															#   mxFree(left); 
															#   mxFree(right); 
															#   } 
	if __debug__:
		print "DBG: basisfun i={}, u={}, p={} U={}, result N={}".format(i,u,p,U,N)
	return N



