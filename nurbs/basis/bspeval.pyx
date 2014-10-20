# function p = bspeval(d,c,k,u)

# BSPEVAL:  Evaluate B-Spline at parametric points.
# 
# Calling Sequence:
# 
#   p = bspeval(d,c,k,u)
# 
#    INPUT:
# 
#       d - Degree of the B-Spline.
#       c - Control Points, matrix of size (dim,nc).
#       k - Knot sequence, row vector of size nk.
#       u - Parametric evaluation points, row vector of size nu.
# 
#    OUTPUT:
#
#       p - Evaluated points, matrix of size (dim,nu)
# 
#    Copyright (C) 2000 Mark Spink, 2007 Daniel Claxton, 2010 C. de Falco
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# include "findspan.pyx"
# cimport findspan

cpdef np.ndarray[DTYPE_t, ndim=2] bspeval(unsigned int d, np.ndarray[DTYPE_t, ndim=2] c, np.ndarray[DTYPE_t, ndim=1] k, np.ndarray[DTYPE_t, ndim=1] u):
	assert c.dtype == DTYPE
	assert k.dtype == DTYPE
	assert u.dtype == DTYPE
	if __debug__:
		print"DBG: BSPEVAL evaluation of parametric points. d={}, coefs={}, knots={} pts/u={}".format(d, c, k, u)

	cdef unsigned int mc, nc, nu, s, i, tmp1
	cdef double tmp2

	mc = c.shape[0]
	nc = c.shape[1]

	cdef np.ndarray[DTYPE_t, ndim=2] ret = np.zeros([mc, u.size], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] N = np.zeros(d+1, dtype=DTYPE)

	for col in range(u.size):
		s =  findspan(k.size-d-2, d, u[col], k)
		# print "nc-1 = {}, d={}, u={}, k={}".format( nc-1, d, u[col], k)
		# s =  findspan(nc-1, d, u[col], k)
		N = basisfun(s, u[col], d, k)
		tmp1 = s - d
		for row in range (mc):
			tmp2 = 0.
			for i in range(d+1):
				# print "N[{}] = {}".format(i,N[i])
				# print c[row, tmp1+i]
				tmp2 += N[i] * c[row, tmp1+i]
			# print "res " + str(tmp2)
			ret[row,col] = tmp2

	if __debug__:
		print "DBG: bspeval, result ret={}".format(ret)
	return ret
