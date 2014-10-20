#!/usr/bin/env python2
from __future__ import division
import numpy as np
import sys
# sys.path.insert(0, '../nurbs_toolbox/')
#from nurbs import _findspan, basisfunder
# import nrbmak
#import geo

def op_f_v_tp(space, msh, coeff):
	"OP_F_V_TP: assemble the right-hand side vector r = [r(i)], with  r(i) = (f, v_i), exploiting the tensor product structure."

	if __debug__:
		print "DBG: tensor product assembly of op_f_v"

	rhs = np.zeros (space.ndof)

	for iel in np.arange(msh.nel_dir[0]):
		msh_col = msh.evaluate_col(iel)
		# stiffness matrix requires first derivative
		sp_col = space.evaluate_col(msh_col, value=True) # ask for value of shape function

		# x are the positions of the gauss pts (probably). coefficient is evaluated at this pts
		x = []
		#for idim in np.arange(len(msh.qn)):
		for idim in np.arange(msh.ndim):
			# calls 
			x.append( np.reshape( msh_col.geo_map[idim], (msh_col.nqn, msh_col.nel), order='F'))
			# print x[-1]

		# simplification, only constant coefficient is allowed atm.
		# import pdb; pdb.set_trace()
		# coeff = np.ones(x[0].shape)
		coe = coeff(x[0], x[1]) # evaluation of lambda function

		# column contribution to the stiffness matrix.
		#import pdb; pdb.set_trace()
		if __debug__:
			print "DBG: Calling gradu gradv assembly of column {}".format(iel)
		rhs = rhs + op_f_v ( sp_col, msh_col, coe)

	return rhs

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
			rhs_loc = np.sum( np.sum(aux_val, axis=1), axis=0) # axis 1 is the quadrature sum, axis 0 is ncomp which is always 1, so nil impact
			rhs[spv.connectivity[np.arange(spv.nsh[iel], dtype=np.uint),iel]] += rhs_loc[:] # element sum

 		else:
 			print "WARNING: jacdet zero at quad node!!"

	return rhs




if __name__ == '__main__':
	''' test
	'''
