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
                                                 
cpdef unsigned int findspan( unsigned int n, unsigned int p, double u, np.ndarray[DTYPE_t, ndim=1] U):    # int findspan(int n, int p, double u, double *U) { 
	assert U.dtype == DTYPE

	
	cdef unsigned int low, high, mid                #   int low, high, mid;                                                 
	                                                #   // special case 
	if u == U[n+1]: 
		if __debug__:
			print "DBG: findspan n={}, p={}, u={}, knots/U={}. res = {}".format(n, p, u, U, n)
		return n						            #   if (u == U[n+1]) return(n); 
	                                                #   // do binary search 
	low = p                                        #   low = p; 
	high = n + 1                                   #   high = n + 1; 
	mid = (low + high) // 2; 	                    #   mid = (low + high) / 2; 
	while u < U[mid] or u >= U[mid+1]:           #   while (u < U[mid] || u >= U[mid+1])  { 
		if u < U[mid]:                           #     if (u < U[mid]) 
			high = mid                             #       high = mid; 
		else:                                        #     else 
			low = mid                              #       low = mid;                   
		mid = (low + high) // 2              #     mid = (low + high) / 2; 
	                                             #   } 
	                                                # 
	if __debug__:
		print "DBG: findspan n={}, p={}, u={}, knots/U={}. res = {}".format(n, p, u, U, mid)
	return mid                                        #   return(mid); 
                                                #   } 

# cpdef np.ndarray[DTYPEui_t, ndim=1] findspan_( unsigned int n, unsigned int p, np.ndarray[DTYPE_t, ndim=1] u, np.ndarray[DTYPE_t, ndim=1] U):    
# 	assert U.dtype == DTYPE
# 	
# 	cdef unsigned int low, high, mid                #   int low, high, mid;                                                 
# 	                                                #   // special case 
# 	if u == U[n+1]: 
# 		return n						            #   if (u == U[n+1]) return(n); 
# 	                                                #   // do binary search 
# 	low = p                                        #   low = p; 
# 	high = n + 1                                   #   high = n + 1; 
# 	mid = (low + high) // 2; 	                    #   mid = (low + high) / 2; 
# 	while u < U[mid] or u >= U[mid+1]:           #   while (u < U[mid] || u >= U[mid+1])  { 
# 		if u < U[mid]:                           #     if (u < U[mid]) 
# 			high = mid                             #       high = mid; 
# 		else:                                        #     else 
# 			low = mid                              #       low = mid;                   
# 		mid = (low + high) // 2              #     mid = (low + high) / 2; 
# 	                                             #   } 
# 	                                                # 
# 	return mid                                        #   return(mid); 
#                                                 #   } 
