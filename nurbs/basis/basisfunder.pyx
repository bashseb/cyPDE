# function dersv = basisfunder (ii, pl, uu, u_knotl, nders)
# BASISFUNDER:  B-Spline Basis function derivatives.
#
# Calling Sequence:
# 
#   ders = basisfunder (ii, pl, uu, k, nd)
#
#    INPUT:
#   
#      ii  - knot span index (see findspan)
#      pl  - degree of curve
#      uu  - parametric points (u, below)
#      k   - knot vector (U below)
#      nd  - number of derivatives to compute (k below)
#
#    OUTPUT:
#   
#      ders - ders(n, i, :) (i-1)-th derivative at n-th point
#   
#    Adapted from Algorithm A2.3 from 'The NURBS BOOK' pg72.
#
# See also: 
#
#    numbasisfun, basisfun, findspan
#
#    Copyright (C) 2009,2011 Rafael Vazquez
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


# I'd need a ufunc which accepts vectors/arrays for u
cpdef np.ndarray[DTYPE_t, ndim=2] basisfunder(unsigned int i, unsigned int p, double u,  unsigned int k, np.ndarray[DTYPE_t, ndim=1] U):

	assert U.dtype == DTYPE
	assert k <= p

	cdef unsigned int l, j, s1, s2
	cdef int j1, j2, pk, rk, r
	cdef double temp, saved, der

	cdef np.ndarray[DTYPE_t, ndim=1] left = np.zeros(p+1, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] right = np.zeros(p+1, dtype=DTYPE)

	cdef np.ndarray[DTYPE_t, ndim=2] a = np.zeros([2,p+1], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=2] ndu = np.zeros([p+1,p+1], dtype=DTYPE)

	cdef np.ndarray[DTYPE_t, ndim=2] ders = np.zeros([k+1,p+1], dtype=DTYPE)

	if __debug__:
		print "DBG: BASISFUNDER: degree k={} (<= p={}) at u={}, located in knot span i={}. Output is k by p ({}x{}) rows are for p=const".format(k, p, u, i, k+1, p+1)
		#print "output is k by p ({}x{}) rows are for p=const".format(k+1,p+1)

	ndu[0,0] = 1.
	
	for j in range(1,p+1):
		left[j] = u - U[i+1-j]
		right[j] = U[i+j] - u
		saved = 0.

		for r in range(j):
			# lower triangle
			ndu[j,r] = right[r+1] + left[j-r]
			temp = ndu[r,j-1]/ndu[j,r]
			# upper triangle
			ndu[r,j] = saved + right[r+1] * temp
			saved = left[j-r] * temp

		ndu[j,j] = saved


	for j in range(p+1): # basis functions, ok
		ders[0,j] = ndu[j,p]

	# compute derivs
	for r in range(p+1):
		s1=0
		s2=1
		a[0,0]=1.

		for l in range(1, k+1): #### i=>l, n=>k
			der = 0.
			rk = r-l
			pk = p-l

			if r >= l:
				a[s2,0] = a[s1,0] / ndu[pk+1,rk]
				der = a[s2,0] * ndu[rk,pk]

			if rk >= -1:
				j1 = 1
			else:
				j1 = -rk

			if r-1 <= pk:
				j2 = l-1
			else:
				j2 = p - r

			for j in range(j1, j2+1):
				a[s2,j] = ( a[s1,j] - a[s1,j-1] ) / ndu[pk+1,rk+j]
				der += a[s2,j] * ndu[rk+j,pk]

			if r <= pk:
				a[s2,l] = -a[s1, l-1] / ndu[pk+1,r]
				der += a[s2,l] * ndu[r,pk]

			ders[l,r] = der
			j=s1
			s1=s2
			s2=j
	r = p
	for l in range(1,k+1):
		for j in range(p+1):
			ders[l,j]  *= r
		r *= (p-l)


	return ders


#-orig  dersv = zeros(numel(uu), nders+1, pl+1);
#-orig 
#-orig   for jj = 1:numel(uu)
#-orig 
#-orig     i = ii(jj)+1; ## convert to base-1 numbering of knot spans
#-orig     u = uu(jj);
#-orig 
#-orig     ders = zeros(nders+1,pl+1);
#-orig     ndu = zeros(pl+1,pl+1);
#-orig     left = zeros(pl+1);
#-orig     right = zeros(pl+1);
#-orig     a = zeros(2,pl+1);
#-orig     ndu(1,1) = 1;
#-orig     for j = 1:pl
#-orig       left(j+1) = u - u_knotl(i+1-j);
#-orig       right(j+1) = u_knotl(i+j) - u;
#-orig       saved = 0;
#-orig       for r = 0:j-1
#-orig         ndu(j+1,r+1) = right(r+2) + left(j-r+1);
#-orig         temp = ndu(r+1,j)/ndu(j+1,r+1);
#-orig         ndu(r+1,j+1) = saved + right(r+2)*temp;
#-orig         saved = left(j-r+1)*temp;
#-orig       end
#-orig       ndu(j+1,j+1) = saved;
#-orig     end   
#-orig     for j = 0:pl
#-orig       ders(1,j+1) = ndu(j+1,pl+1);
#-orig     end
#-orig     for r = 0:pl
#-orig       s1 = 0;
#-orig       s2 = 1;
#-orig       a(1,1) = 1;
#-orig       for k = 1:nders #compute kth derivative
#-orig         d = 0;
#-orig         rk = r-k;
#-orig         pk = pl-k;
#-orig         if (r >= k)
#-orig           a(s2+1,1) = a(s1+1,1)/ndu(pk+2,rk+1);
#-orig           d = a(s2+1,1)*ndu(rk+1,pk+1);
#-orig         end
#-orig         if (rk >= -1)
#-orig           j1 = 1;
#-orig         else 
#-orig           j1 = -rk;
#-orig         end
#-orig         if ((r-1) <= pk)
#-orig           j2 = k-1;
#-orig         else 
#-orig           j2 = pl-r;
#-orig         end
#-orig         for j = j1:j2
#-orig           a(s2+1,j+1) = (a(s1+1,j+1) - a(s1+1,j))/ndu(pk+2,rk+j+1);
#-orig           d = d + a(s2+1,j+1)*ndu(rk+j+1,pk+1);
#-orig         end
#-orig         if (r <= pk)
#-orig           a(s2+1,k+1) = -a(s1+1,k)/ndu(pk+2,r+1);
#-orig           d = d + a(s2+1,k+1)*ndu(r+1,pk+1);
#-orig         end
#-orig         ders(k+1,r+1) = d;
#-orig         j = s1;
#-orig         s1 = s2;
#-orig         s2 = j;
#-orig       end
#-orig     end
#-orig     r = pl;
#-orig     for k = 1:nders
#-orig       for j = 0:pl
#-orig         ders(k+1,j+1) = ders(k+1,j+1)*r;
#-orig       end
#-orig       r = r*(pl-k);
#-orig     end
#-orig 
#-orig     dersv(jj, :, :) = ders;
#-orig     
#-orig   end
#-orig 
#-orig end

#!test
#! k    = [0 0 0 0 1 1 1 1];
#! p    = 3;
#! u    = rand (1);
#! i    = findspan (numel(k)-p-2, p, u, k);
#! ders = basisfunder (i, p, u, k, 1);
#! sumders = sum (squeeze(ders), 2);
#! assert (sumders(1), 1, 1e-15);
#! assert (sumders(2:end), 0, 1e-15);

#!test
#! k    = [0 0 0 0 1/3 2/3 1 1 1 1];
#! p    = 3;
#! u    = rand (1);
#! i    = findspan (numel(k)-p-2, p, u, k);
#! ders = basisfunder (i, p, u, k, 7); 
#! sumders = sum (squeeze(ders), 2);
#! assert (sumders(1), 1, 1e-15);
#! assert (sumders(2:end), zeros(rows(squeeze(ders))-1, 1), 1e-13);

#!test
#! k    = [0 0 0 0 1/3 2/3 1 1 1 1];
#! p    = 3;
#! u    = rand (100, 1);
#! i    = findspan (numel(k)-p-2, p, u, k);
#! ders = basisfunder (i, p, u, k, 7);
#! for ii=1:10
#!   sumders = sum (squeeze(ders(ii,:,:)), 2);
#!   assert (sumders(1), 1, 1e-15);
#!   assert (sumders(2:end), zeros(rows(squeeze(ders(ii,:,:)))-1, 1), 1e-13);
#! end
#! assert (ders(:, (p+2):end, :), zeros(numel(u), 8-p-1, p+1), 1e-13)
#! assert (all(all(ders(:, 1, :) <= 1)), true)



