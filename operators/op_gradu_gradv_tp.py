#!/usr/bin/env python2
from __future__ import division
import numpy as np
import sys
sys.path.insert(0, '../nurbs_toolbox/')
#from nurbs import _findspan, basisfunder
# import nrbmak
#import geo

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix



def op_gradu_gradv_tp(space1, space2, msh, coeff):
	"tensor product gradu gradv assembly"

	if __debug__:
		print "DBG: tensor product assembly of gradu gradv operator"
	if space1.ndof == space2.ndof:
		isSquare = True
	else:
		isSquare = False
		print "WARNING: yet unexpected... "
	

	# lilmatrix usable, but default matlab format is csc
	# A = lil_matrix((space1.ndof, space2.ndof))
	# 
	# A = np.zeros((space1.ndof, space2.ndof))
	# csc_matrix((M, N), [dtype])
	A = csc_matrix((space1.ndof, space2.ndof))

	
	# assert msh.nel_dir[1] == 1 # to find out what vals >1 mean, c.f. manual columns in 2d msh slices in 3d msh

	for iel in np.arange(msh.nel_dir[0]):
		msh_col = msh.evaluate_col(iel)
		# stiffness matrix requires first derivative
		sp1_col = space1.evaluate_col(msh_col, value=False, gradient=True) # ask for gradient (shape_function_gradients) not for shape_functions (actually, shape functions are contained in shape_function_gradients in index [1,:,:,:]
		sp2_col = space2.evaluate_col(msh_col, value=False, gradient=True)

		x = []
		for idim in np.arange(msh.ndim):
			x.append( np.reshape( msh_col.geo_map[idim], (msh_col.nqn, msh_col.nel), order='F'))

		# simplification, only constant coefficient is allowed atm.
		# coeff = np.ones(x[0].shape)
		coe = coeff(x[0], x[1]) # evaluation of lambda function (2d)
		coe = coe.reshape(x[0].shape)


		# column contribution to the stiffness matrix.
		# print sp1_col.nsh_max
		# print sp2_col.nsh_max
		#import pdb; pdb.set_trace()
		if __debug__:
			print "DBG: Calling gradu gradv assembly of column {}".format(iel)
		A = A + op_gradu_gradv ( sp1_col, sp2_col, msh_col, coe)

		# TODO: rows, cols, vals... later
	
	#print A.todense()

	return A

# TODO: cythonize
def op_gradu_gradv(spu, spv, msh, coeff):
	"assemble stiffness matrix A = [(a_(i,j)] with a_(i,j) = (coeff grad u_j, grad v_j)"

	# 1. reshaping the precalculated column gradients
	gradu = np.reshape(spu.shape_function_gradients, (spu.ncomp, -1, msh.nqn, spu.nsh_max, msh.nel), order ='F')
	gradv = np.reshape(spv.shape_function_gradients, (spv.ncomp, -1, msh.nqn, spv.nsh_max, msh.nel), order ='F')
	#print gradu.shape
	print "gradient assembly, gradu==gradv: " + str(np.all(gradu==gradv))

	ndir = gradu.shape[1] #?? 

	# 2. empty fields
	rows = np.zeros(msh.nel * spu.nsh_max * spv.nsh_max)
	cols = np.zeros(msh.nel * spu.nsh_max * spv.nsh_max)
	values = np.zeros(msh.nel * spu.nsh_max * spv.nsh_max)

	ncounter = 0
	for iel in np.arange(msh.nel):
		# values in jacobian must be strictly non-zero
		# print iel
		if np.all(msh.jacdet[:,iel] != 0.):
			# jacdet_weights
			# print msh.nqn
			# print msh.nqn.dtype
			# print msh.jacdet.shape
			# print msh.quad_weights.shape
			# print coeff.shape

			jacdet_weights = np.reshape( msh.jacdet[:,iel] * msh.quad_weights[:,iel] * coeff[:,iel], (1, msh.nqn), order = 'F')
			# print gradu.shape
			gradu_iel = np.transpose(gradu[:,:,:, :spu.nsh[iel], iel], (0,1,3,2))
			gradu_iel = np.reshape(gradu_iel, (spu.ncomp * ndir, spu.nsh[iel], msh.nqn), order = 'F')
			gradu_iel = np.transpose(gradu_iel, (0,2,1))
			# print gradu_iel.shape

			gradv_iel = np.transpose(gradv[:,:,:, :spv.nsh[iel], iel], (0,1,3,2))
			gradv_iel = np.reshape(gradv_iel, (spv.ncomp * ndir, spv.nsh[iel], msh.nqn), order = 'F')
			gradv_iel = np.transpose(gradv_iel, (0,2,1))
			if __debug__:
				print "gradient assembly, gradu_iel==gradv_iel: " + str(np.all(gradu_iel==gradv_iel))

			# outch... this 'T' error was a bad one
			gradv_times_jw = jacdet_weights.T * gradv_iel # broadcasting
			# print gradv_times_jw.shape # ok, vals not checked
			for idof in np.arange(spv.nsh[iel]):
				rows[ncounter+np.arange(spu.nsh[iel])] = spv.connectivity[idof,iel]
				cols[ncounter+np.arange(spu.nsh[iel])] = spu.connectivity[np.arange(spu.nsh[iel]),iel]

				aux_val = gradv_times_jw[:,:,idof,np.newaxis] * gradu_iel # quad weights * jacdet * coeff * grad u * grad v
				values[ncounter+np.arange(spu.nsh[iel])] = np.sum( np.sum(aux_val, axis=1), axis=0) # axis1 is sum over quad nodes (i.e. 'l' sum in pdf), axis0 is over 'spu/v.ncomp * ndir' which I don't understand.

				ncounter += spu.nsh[iel]
		else:
			print "WARNING: jacdet zero ad quad node!!"

	#import pdb; pdb.set_trace()

	# Sparse matrix creation
	# csr_matrix( (data,(row,col)), shape=(3,3) )
	# csc_matrix( (data,(row,col)), shape=(3,3) )
	return csc_matrix ( (values[:ncounter],(rows[:ncounter], cols[:ncounter])), shape=(spv.ndof, spu.ndof))



	
# % OP_GRADU_GRADV_TP: assemble the stiffness matrix A = [a(i,j)], a(i,j) = (epsilon grad u_j, grad v_i), exploiting the tensor product structure.
# %
# %   mat = op_gradu_gradv_tp (spu, spv, msh, epsilon);
# %   [rows, cols, values] = op_gradu_gradv_tp (spu, spv, msh, epsilon);
# %
# % INPUT:
# %
# %   spu:     class representing the space of trial functions (see sp_bspline_2d)
# %   spv:     class representing the space of test functions (see sp_bspline_2d)
# %   msh:     class defining the domain partition and the quadrature rule (see msh_2d)
# %   epsilon: function handle to compute the diffusion coefficient
# %
# % OUTPUT:
# %
# %   mat:    assembled stiffness matrix
# %   rows:   row indices of the nonzero entries
# %   cols:   column indices of the nonzero entries
# %   values: values of the nonzero entries
# % 
# % Copyright (C) 2011, Carlo de Falco, Rafael Vazquez
# %
# %    This program is free software: you can redistribute it and/or modify
# %    it under the terms of the GNU General Public License as published by
# %    the Free Software Foundation, either version 3 of the License, or
# %    (at your option) any later version.
# 
# %    This program is distributed in the hope that it will be useful,
# %    but WITHOUT ANY WARRANTY; without even the implied warranty of
# %    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# %    GNU General Public License for more details.
# %
# %    You should have received a copy of the GNU General Public License
# %    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# function varargout = op_gradu_gradv_tp (space1, space2, msh, coeff)
# 
#   A = spalloc (space2.ndof, space1.ndof, 3*space1.ndof);
# 
#   ndim = numel (msh.qn);
# 
#   for iel = 1:msh.nel_dir(1)
# 	disp('Calling msh_evaluate_col from op_gradu_gradv_tp.m')
# 	disp(iel)
#     msh_col = msh_evaluate_col (msh, iel);
#     sp1_col = sp_evaluate_col (space1, msh_col, 'value', false, 'gradient', true);
#     sp2_col = sp_evaluate_col (space2, msh_col, 'value', false, 'gradient', true);
# 
#     for idim = 1:ndim
#       x{idim} = reshape (msh_col.geo_map(idim,:,:), msh_col.nqn, msh_col.nel);
#     end
# 
#     A = A + op_gradu_gradv (sp1_col, sp2_col, msh_col, coeff (x{:}));
#   end
# 
#   if (nargout == 1)
#     varargout{1} = A;
#   elseif (nargout == 3)
#     [rows, cols, vals] = find (A);
#     varargout{1} = rows;
#     varargout{2} = cols;
#     varargout{3} = vals;
#   end
# 
# end


if __name__ == '__main__':
	''' test
	'''
