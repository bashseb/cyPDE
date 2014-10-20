# function [dc,dk] = bspderiv(d,c,k)

# BSPDERIV:  B-Spline derivative.
# 
#  MATLAB SYNTAX:
# 
#         [dc,dk] = bspderiv(d,c,k)
#  
#  INPUT:
# 
#    d - degree of the B-Spline
#    c - control points          double  matrix(mc,nc)
#    k - knot sequence           double  vector(nk)
# 
#  OUTPUT:
# 
#    dc - control points of the derivative     double  matrix(mc,nc)
#    dk - knot sequence of the derivative      double  vector(nk)
# 
#  Modified version of Algorithm A3.3 from 'The NURBS BOOK' pg98.
#
#    Copyright (C) 2000 Mark Spink, 2007 Daniel Claxton
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# if not __debug__:
# 	@cython.boundscheck(False) # turn of bounds-checking for entire function, no negative indices may be used.

# I'm not specifying the return type here, as numpy arrays are python objects anyways.
cpdef bspderiv(unsigned int d, np.ndarray[DTYPE_t, ndim=2] c, np.ndarray[DTYPE_t, ndim=1] k):

	if __debug__:
		print "DBG: BSPDERIV. derivative of bspline "
	assert c.dtype == DTYPE
	assert k.dtype == DTYPE

	cdef unsigned int i, j, mc, nc, nk
	cdef double tmp

	mc = c.shape[0]
	nc = c.shape[1]
	nk = k.size
	cdef np.ndarray[DTYPE_t, ndim=2] dc = np.zeros([mc, nc-1], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] dk = np.zeros(nk-2, dtype=DTYPE)

	for i in range(nc-1):
		tmp = d/ (k[i+d+1] - k[i+1])
		for j in range(mc):
			dc[j,i] = tmp * (c[j,i+1] - c[j,i])
	
	for i in range(nk-2):
		dk[i] = k[i+1];

	return dc, dk

