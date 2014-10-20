#!/usr/bin/env python2
from __future__ import division
import numpy as np
import sys
sys.path.insert(0, '../nurbs_toolbox/')
#from nurbs import _findspan, basisfunder
# import nrbmak
#import geo

# from scipy.sparse import lil_matrix
# from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix




# cp # TODO: cythonize
# cp def op_gradu_gradv(spu, spv, msh, coeff):
# cp 	"assemble stiffness matrix A = [(a_(i,j)] with a_(i,j) = (coeff grad u_j, grad v_j)"
# cp 
# cp 	if __debug__:
# cp 		print "operator gradu gradv called"
# cp 	# 1. reshaping
# cp 	gradu = np.reshape(spu.shape_function_gradients, (spu.ncomp, -1, msh.nqn, spu.nsh_max, msh.nel), order ='F')
# cp 	gradv = np.reshape(spv.shape_function_gradients, (spv.ncomp, -1, msh.nqn, spv.nsh_max, msh.nel), order ='F')
# cp 	#print gradu.shape
# cp 
# cp 	ndir = gradu.shape[1]
# cp 
# cp 	# 2. empty fields
# cp 	rows = np.zeros(msh.nel * spu.nsh_max * spv.nsh_max)
# cp 	cols = np.zeros(msh.nel * spu.nsh_max * spv.nsh_max)
# cp 	values = np.zeros(msh.nel * spu.nsh_max * spv.nsh_max)
# cp 
# cp 	ncounter = 0
# cp 	for iel in np.arange(msh.nel):
# cp 		##### TODO if!
# cp 		# values in jacobian must be strictly non-zero
# cp 		# print iel
# cp 		if np.all(msh.jacdet[:,iel] != 0.):
# cp 			# jacdet_weights
# cp 			# print msh.nqn
# cp 			# print msh.nqn.dtype
# cp 			# print msh.jacdet.shape
# cp 			# print msh.quad_weights.shape
# cp 			# print coeff.shape
# cp 
# cp 			jacdet_weights = np.reshape( msh.jacdet[:,iel] * msh.quad_weights[:,iel] * coeff[:,iel], (1, msh.nqn), order = 'F')
# cp 			# print gradu.shape
# cp 			gradu_iel = np.transpose(gradu[:,:,:, :spu.nsh[iel], iel], (0,1,3,2))
# cp 			gradu_iel = np.reshape(gradu_iel, (spu.ncomp * ndir, spu.nsh[iel], msh.nqn), order = 'F')
# cp 			gradu_iel = np.transpose(gradu_iel, (0,2,1))
# cp 			# print gradu_iel.shape
# cp 
# cp 			gradv_iel = np.transpose(gradv[:,:,:, :spv.nsh[iel], iel], (0,1,3,2))
# cp 			gradv_iel = np.reshape(gradv_iel, (spv.ncomp * ndir, spv.nsh[iel], msh.nqn), order = 'F')
# cp 			gradv_iel = np.transpose(gradv_iel, (0,2,1))
# cp 
# cp 			gradv_times_jw = jacdet_weights * gradv_iel
# cp 			# print gradv_times_jw.shape # ok, vals not checked
# cp 			for idof in np.arange(spv.nsh[iel]):
# cp 				rows[ncounter+np.arange(spu.nsh[iel])] = spv.connectivity[idof,iel]
# cp 				cols[ncounter+np.arange(spu.nsh[iel])] = spu.connectivity[np.arange(spu.nsh[iel]),iel]
# cp 
# cp 				aux_val = gradv_times_jw[:,:,idof,np.newaxis] * gradu_iel
# cp 				# print aux_val.shape
# cp 				values[ncounter+np.arange(spu.nsh[iel])] = np.sum( np.sum(aux_val, axis=1), axis=0)
# cp 
# cp 				ncounter += spu.nsh[iel]
# cp 		else:
# cp 			print "WARNING: jacdet zero ad quad node!!"
# cp 
# cp 
# cp 	# Sparse matrix creation
# cp 	# csr_matrix( (data,(row,col)), shape=(3,3) )
# cp 	# csc_matrix( (data,(row,col)), shape=(3,3) )
# cp 	return csc_matrix ( (values[:ncounter],(rows[:ncounter], cols[:ncounter])), shape=(spv.ndof, spu.ndof))

# def op_u_v(spu, spv, msh, coeff, matrix=True):
# 	"OP_U_V: assemble the mass matrix M = [m(i,j)], m(i,j) = (mu u_j, v_i)."
# 
# 	if __debug__:
# 		print "Assembly of mass matrix"
# 	
# 	shpu = np.reshape(spu.shape_functions, [spu.ncomp, msh.nqn, spu.nsh_max, msh.nel], order = 'F')
# 	shpv = np.reshape(spv.shape_functions, [spv.ncomp, msh.nqn, spv.nsh_max, msh.nel], order = 'F')
# 
# 	rows = np.zeros (msh.nel * spu.nsh_max * spv.nsh_max, dtype=np.int)
# 	cols = np.zeros (msh.nel * spu.nsh_max * spv.nsh_max, dtype=np.int)
# 	values = np.zeros (msh.nel * spu.nsh_max * spv.nsh_max)
# 
# 	ncounter = 0
# 	for iel in np.arange(msh.nel):
# 		if np.all(msh.jacdet[:iel] != 0.):
#  			jacdet_weights = np.reshape( msh.jacdet[:,iel] * msh.quad_weights[:,iel] * coeff[:,iel], (1, msh.nqn), order = 'F')
# 
# 			# iel = 0 for 2d laplace problem ...  no of elements in partition...
# 			# print shpv.shape
# 			# print shpv[:,:,:spv.nsh[iel],0]
# 			shpv_iel = np.reshape(shpv[:,:,:spv.nsh[iel],iel], [spv.ncomp, msh.nqn, spv.nsh[iel]],order='F')
# 			shpu_iel = np.reshape(shpu[:,:,:spu.nsh[iel],iel], [spu.ncomp, msh.nqn, spu.nsh[iel]],order='F')
#  			shpv_times_jw = jacdet_weights * shpv_iel # broadcasting
# 
# 			for idof in np.arange(spv.nsh[iel]):
# 				print spv.connectivity
# 				rows[ncounter+np.arange(spu.nsh[iel])] = spv.connectivity[idof,iel].astype(np.int)
# 				cols[ncounter+np.arange(spu.nsh[iel])] = spu.connectivity[np.arange(spu.nsh[iel]),iel].astype(np.int)
# 
# 				aux_val = shpv_times_jw[:,:,idof,np.newaxis] * shpu_iel # broadcasting
# 				# print aux_val.shape
# 				values[ncounter+np.arange(spu.nsh[iel])] = np.sum( np.sum(aux_val, axis=1), axis=0)
# 
# 				ncounter += spu.nsh[iel]
#  		else:
#  			print "WARNING: jacdet zero ad quad node!!"
# 
# 
# 
# 	if matrix: # return assembled sparse matrix
# 		# csc_matrix( (data,(row,col)), shape=(3,3) )
# 		# testing
# 		return csc_matrix ( (values[:ncounter],(rows[:ncounter], cols[:ncounter])), shape=(spv.ndof, spu.ndof))
# 	else: # return rows, cols and values
# 
# 		return rows[:ncounter], cols[:ncounter], values[:ncounter]

def op_f_v(spv, msh, coeff):
	"OP_F_V: assemble the right-hand side vector r = [r(i)], with  r(i) = (f, v_i)."

	coeff = np.reshape(coeff, [spv.ncomp, msh.nqn, msh.nel], order='F')
	rhs   = np.zeros(spv.ndof)
	shpv  = np.reshape(spv.shape_functions, [spv.ncomp, msh.nqn, spv.nsh_max, msh.nel], order ='F')

	for iel in np.arange(msh.nel):
		if np.all(msh.jacdet[:,iel] != 0.):

			# import pdb; pdb.set_trace() # msh.jacdet.shape?
 			jacdet_weights = np.reshape( msh.jacdet[:,iel] * msh.quad_weights[:,iel], (1, msh.nqn), order = 'F')
			coeff_times_jw = jacdet_weights * coeff[:,:,iel] # broadcasting

			shpv_iel = np.reshape(shpv[:,:,:spv.nsh[iel],iel], [spv.ncomp, msh.nqn, spv.nsh[iel]],order='F')
			aux_val = coeff_times_jw.T * shpv_iel # broadcasting
			rhs_loc = np.sum( np.sum(aux_val, axis=1), axis=0)
			rhs[spv.connectivity[np.arange(spv.nsh[iel], dtype=np.uint),iel]] += rhs_loc[:]

 		else:
 			print "WARNING: jacdet zero at quad node!!"

	return rhs




if __name__ == '__main__':
	''' test
	'''
