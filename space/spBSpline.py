#!/usr/bin/env python2
from __future__ import division
import numpy as np
import sys
from nurbs.bspline import _findspan, basisfunder
from util import util

from operators import op_u_v
from operators import op_f_v

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
# import nrbmak
#import geo

# SP_BSPLINE_2D: Constructor of the class of tensor-product spaces of B-Splines in 2D.
#
#     sp = sp_bspline_2d (knots, degree, msh)
#
# INPUTS:
#     
#     knots:     open knot vector    
#     degree:    b-spline polynomial degree
#     msh:       msh object that defines the quadrature rule (see msh_2d)
#
# OUTPUT:
#
#    sp: object representing the discrete function space, with the following fields and methods:
#
#        FIELD_NAME      (SIZE)                      DESCRIPTION
#        knots           (1 x 2 cell array)          knot vector in each parametric direction
#        degree          (1 x 2 vector)              splines degree in each parametric direction
#        spu             (struct)                    space of univariate splines in the first parametric direction
#        spv             (struct)                    space of univariate splines in the second parametric direction
#        ndof            (scalar)                    total number of degrees of freedom
#        ndof_dir        (1 x 2 vector)              degrees of freedom along each direction
#        nsh_max         (scalar)                    maximum number of shape functions per element
#        nsh_dir         (1 x 2 vector)              maximum number of univariate shape functions per element in each parametric direction
#        ncomp           (scalar)                    number of components of the functions of the space (actually, 1)
#        boundary        (1 x 4 struct array)        struct array representing the space of traces of basis functions on each edge
#
#       METHOD_NAME
#       sp_evaluate_col: compute the basis functions (and derivatives) in one column of the mesh (that is, fixing the element in the first parametric direction).
#       sp_evaluate_col_param: compute the basis functions (and derivatives) in one column of the mesh in the reference domain.
#       sp_eval_boundary_side: evaluate the basis functions in one side of the boundary.
#       sp_precompute:  compute any of the fields related to the discrete
#                       space (except boundary), in all the quadrature points,
#                       as in the space structure from previous versions.
#
# Copyright (C) 2009, 2010, 2011 Carlo de Falco
# Copyright (C) 2011 Rafael Vazquez
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

class OneSpace(object):
	''' TODO: think about this'''

class BspSpace(object):
	"b-spline space base class"
	spCnt = 0
	def __init__(self, knots, degree, msh):

		assert len(knots) == 2
		BspSpace.spCnt +=1
		self.knots = knots
		self.degree = degree
		self.qn = msh.qn  # nodes are not saved in ref... just applied to construct spu, spv spaces
		self.ndof = -1
		self.ndof_dir = -1
		self.nsh_max = -1
		self.nsh_dir = -1
		self.ncomp = 1
		if msh.boundary:
			self.boundary = msh.bnd[:]
		else:
			self.boundary = None

	# should this be a class function or outside helper function? Or inside OneSpace class?
	def sp_bspline_1d_param(self, knots, degree, qqn, gradient=True, hessian=False):
		'''Construct a space of B-Splines on the parametric domain in 1D.'''
		#Note: nodes is an array from qn

		if __debug__:
			print "DBG: bspline 1d param execution"
		
		nders = 0
		if hessian:
			nders = 2
		elif gradient:
			nders = 1

		sp = OneSpace()
		mknots = knots.size -1

		p = degree
		mcp = - p - 1 + mknots
		ndof = mcp + 1
		nel = qqn.shape[1] # same as qn! number of elements... = np.unique(refined knot vector).size-1 
		# must be 20 in case of post processing
		nqn = qqn.shape[0] # number of quadrature nodes per element

		nsh = np.ones((nel), dtype=np.int)*(-1)
		#assert nel == 1 # only one element....
		# import pdb; pdb.set_trace()

		connectivity = []
		# connectivity: numbers associated to the basis function which do not vanish in each element
		for iel in np.arange(nel):
			s = _findspan (mcp, p, qqn[:,iel].conj().transpose(), knots)
			# numbasisfun
			c = np.transpose((s-p) + np.reshape(np.arange(p+1,dtype=np.int), (p+1,1))).astype(np.int) # using broadcasting rules
			### I think '+1' is from matlab indexing
			c = np.unique(c.flatten(order='F'))+1
			# c = np.unique(c.flatten(order='F'))
			# connectivity[:,iel] = c
			nsh[iel] = (c != 0).sum().astype(np.int) # interesting: The number of shape functions can vary on a column basis. What does this mean?
			connectivity.append((c-1).astype(np.int))  # Fix it back

		# assumption: all 'c's above are of same size
		# assert nel == 1 # else needs to test connectivity code below
		ccn = np.ones((c.size, nel), dtype=np.int) * (-1)
		for iel in np.arange(nel):
			ccn[:,iel] = connectivity[iel].astype(np.int)
 # interesting: The number of shape functions can vary on a column basis. What does this mean? Nay, that' constant in my case .
		sp.nsh = nsh #.flatten()
		sp.nsh_max = np.max(nsh)
		sp.ndof = ndof
		sp.connectivity = ccn.astype(np.int)
		sp.ncomp = 1 #??

		s = _findspan(mcp, p, qqn.flatten(order='F'), knots)

		# Note: unpythonic...
		# needs ufunc... TODO
		tders = np.zeros((s.size,nders+1,p+1)) 
		# nqn 4, nel 2 

		# import pdb; pdb.set_trace()
		for i in np.arange(qqn.size): # loop over all elements in qqn vector.
			tders[i] = basisfunder( s[i], p, qqn.flatten(order='F')[i], nders, knots) 
			# e.g. the 2nd derivative of p=4 has the shape (3,5) (nders+1, p+1)
# 		if nqn == s.size:
# 			for i in np.arange(s.size):
# 				# print "s={}, p={}, node={}, knots ={}, nders={}".format(s[i], p, qqn[i], knots, nders)
# 				tders[i] = basisfunder( s[i], p, qqn[i,0], nders, knots) 
# 		elif nel == s.size:
# 			for i in np.arange(s.size):
# 				# print "s={}, p={}, node={}, knots ={}, nders={}".format(s[i], p, qqn[i], knots, nders)
# 				tders[i] = basisfunder( s[i], p, qqn[0,i], nders, knots)

		#nbf = np.transpose((s-p) + np.reshape(np.arange(p+1), (p+1,1))) # broadcasting
		#nbf = s[None,:].reshape(-1,nel).flatten(order='F')[:,None]-p + np.arange(p+1)
		nbf = (s-p)[:,None] + np.arange(p+1)

		# (s[:,None].reshape(-1,nel)-p)+ np.arange(p+1)[:,None]
		# nbf checked.
		# I'm not using s.shape because my s is not the same shape as qqn (nel x nqn)
		nbf = np.reshape(nbf, [nqn, nel, p+1], order='F') # MATLAB +1


		# if nel == s.size: # call by gradu gradv
		# 	nbf = np.reshape(nbf+1, (nel, nel, p+1), order ='F') # MATLAB +1?
		# elif nel == s.size: # call by post processor
		# 	nbf = np.reshape(nbf+1, (nel, nel, p+1), order ='F') # MATLAB +1?
		ders = np.zeros((qqn.size,nders+1,sp.nsh_max)) # size is (nqn x nel, nders+1 (=3 for gradient), nsh_max (depending on connectivity))

		for inqn in np.arange(qqn.size): # loop over quadrature points in vector
			# c ordering
			ir, iel = np.unravel_index(inqn,qqn.shape, order='F')
			ind = np.argwhere(connectivity[iel] == nbf[ir,iel,0])
			# the there are always p+1 basis functions (last index):
			ders[inqn,:,ind:ind+p+1] = tders[inqn] # first index of `ders` will be quadrature point/node index

		# need to do some reshaping, instead of '-1' one could also use p+1, I guess.
		shape_functions = np.reshape(ders[:,0,:], (nqn, nel, -1), order ='F') # get all points (first index), only 0-th derivative and all p+1 shape functions of this point
		sp.shape_functions = np.transpose(shape_functions, [0, 2, 1])

		if (gradient):
			# need to do some reshaping
			shape_function_gradients = np.reshape(ders[:,1,:], (nqn, nel,-1), order='F')
			sp.shape_function_gradients = np.transpose(shape_function_gradients, [0,2,1])
			

		if (hessian):
			shape_function_hessian = np.reshape(ders[:,2,:], (nqn, nel,-1), order='F')
			sp.shape_function_hessian = np.transpose(shape_function_hessian, [0,2,1])
				
		return sp

# def numbasisfun(i, u, p, U):

# INPUT:
#     
#     u:         vector of dof weights
#     space:     object defining the discrete space (see sp_bspline_3d)
#     msh:       object defining the points where to evaluate (see msh_3d)
#
# OUTPUT:
#
#     eu: the function evaluated in the given points 
#     F:  grid points in the physical domain, that is, the mapped points

	def eval_msh(self, u, msh): # tailored to my needs without evaluation of F or msh_col
		"SP_EVAL_MSH: Evaluate a function, given by its degrees of freedom, at the points given by a msh object."
		" do not use for gradient yet"
		ndim = len(msh.qn)
		
		F = np.zeros((ndim, msh.nqn, msh.nel))
		eu = np.zeros((self.ncomp, msh.nqn, msh.nel))

		for iel in np.arange(msh.nel_dir[0]):
			#msh_col = msh.evaluate_col(iel) 
			msh_col = DummyCol();
			msh_col.colnum = iel
			msh_col.nel = msh.nel_dir[1]
			msh_col.nel_dir = msh.nel_dir
			msh_col.nqn_dir = msh.nqn_dir
			msh_col.nqn = msh.nqn
			msh_col.geo_map = -2
			msh_col.elem_list =  iel + msh.nel_dir[0] * np.arange(msh.nel_dir[1]).reshape(1, msh.nel_dir[1])
			sp_col = self.evaluate_col(msh_col) # value=true, gradient=false

			# import pdb; pdb.set_trace() msh_col.nel =20, msh_col.nqn = 1 (iel=2,3). msh_col.elem_list = array([[  2,  22,  42,  62,  82, 102, 122, 142, 162, 182, 202, 222, 242, 262, 282, 302, 322, 342, 362, 382]])
			uc_iel = np.zeros( sp_col.connectivity.shape)
			uc_iel[sp_col.connectivity >= 0] = u[sp_col.connectivity[sp_col.connectivity >= 0]]
			weight = np.tile( np.reshape(uc_iel, [1, 1, sp_col.nsh_max, msh_col.nel] ,order='F'), [sp_col.ncomp, msh_col.nqn, 1, 1])
			# reshape the pre-calculated shape functions
			sp_col.shape_functions = np.reshape(sp_col.shape_functions, (sp_col.ncomp, msh_col.nqn, sp_col.nsh_max, msh_col.nel), order='F')


			###
			#import pdb; pdb.set_trace()
			F[:,:,msh_col.elem_list.flatten(order='F')] = msh_col.geo_map
			eu[:,:,msh_col.elem_list.flatten(order='F')] = np.reshape(np.sum(weight * sp_col.shape_functions, axis=2), [sp_col.ncomp, msh_col.nqn, msh_col.nel], order='F') # result product here

		if self.ncomp ==1:
			eu = np.reshape(eu, (msh.nqn, msh.nel), order ='F')
		
		# F correct, eu has error because vector u (uc_iel, weights) differs. so far, int_dof are correct (0), but Dirichlet vals are wrong
		return eu, F

	def eval_msh_original(self, u, msh):
		"SP_EVAL_MSH: Evaluate a function, given by its degrees of freedom, at the points given by a msh object."
		ndim = len(msh.qn)
		
		F = np.zeros((ndim, msh.nqn, msh.nel))
		eu = np.zeros((self.ncomp, msh.nqn, msh.nel))

		for iel in np.arange(msh.nel_dir[0]):
			msh_col = msh.evaluate_col(iel) 
			sp_col = self.evaluate_col(msh_col)

			# import pdb; pdb.set_trace() msh_col.nel =20, msh_col.nqn = 1 (iel=2,3). msh_col.elem_list = array([[  2,  22,  42,  62,  82, 102, 122, 142, 162, 182, 202, 222, 242, 262, 282, 302, 322, 342, 362, 382]])
			uc_iel = np.zeros( sp_col.connectivity.shape)
			uc_iel[sp_col.connectivity >= 0] = u[sp_col.connectivity[sp_col.connectivity >= 0]]
			weight = np.tile( np.reshape(uc_iel, [1, 1, sp_col.nsh_max, msh_col.nel] ,order='F'), [sp_col.ncomp, msh_col.nqn, 1, 1])
			sp_col.shape_functions = np.reshape(sp_col.shape_functions, (sp_col.ncomp, msh_col.nqn, sp_col.nsh_max, msh_col.nel), order='F')


			###
			#import pdb; pdb.set_trace()
			F[:,:,msh_col.elem_list.flatten(order='F')] = msh_col.geo_map
			eu[:,:,msh_col.elem_list.flatten(order='F')] = np.reshape(np.sum(weight * sp_col.shape_functions, axis=2), [sp_col.ncomp, msh_col.nqn, msh_col.nel], order='F') # result product here

		if self.ncomp ==1:
			eu = np.reshape(eu, (msh.nqn, msh.nel), order ='F')
		
		# F correct, eu has error because vector u (uc_iel, weights) differs. so far, int_dof are correct (0), but Dirichlet vals are wrong
		return eu, F


class BspSpace2d(BspSpace):
	"2d bspline space class"


	def __init__(self, knots, degree, msh):

		assert len(knots) == 2
		assert degree.size == 2
		assert msh.qn[0].ndim == 2
		assert msh.qn[1].ndim == 2
		# TODO: validate msh object
		self.ndim = 2
		super( BspSpace2d, self).__init__(knots, degree, msh)
		#kself.spu

		if __debug__:
		    print "DBG: new Bspline surface space of degree {} for {} nodes in u direction and {} nodes in v direction".format(degree, self.qn[0].size, self.qn[1].size)
		# Note: sp_bspline_1d_param is only capable of 1d nodes arrays
		hessian=True
		if np.any(degree) < 2:
			hessian = False
		self.spu = self.sp_bspline_1d_param(self.knots[0], self.degree[0], self.qn[0], gradient=True, hessian=hessian)
		self.spv = self.sp_bspline_1d_param(self.knots[1], self.degree[1], self.qn[1], gradient=True, hessian=hessian)
		

# 		test case which currently fails sp_bspline_1d_param
# 		knots = np.array([0, 0, 0, .5, 1, 1, 1])
# 		degree = 2
# 		points = np.array([[0, .1, .2],[.6, .7, .8]])
# 		testsp = self.sp_bspline_1d_param(knots, degree, points, hessian=True)

		self.nsh_max = self.spu.nsh_max * self.spv.nsh_max
		self.nsh_dir = np.array([self.spu.nsh_max, self.spv.nsh_max])
		self.ndof = self.spu.ndof * self.spv.ndof
		self.ndof_dir = np.array([self.spu.ndof, self.spv.ndof])
		self.ncomp = 1
		mcp = self.ndof_dir[0]
		ncp = self.ndof_dir[1]

		if self.boundary is not None:
			ind = np.array([1, 1, 0, 0])
			for iside in np.arange(len(self.boundary)):
				self.boundary[iside].ndof = self.ndof_dir[ind[iside]]
				self.boundary[iside].nsh_max = self.nsh_dir[ind[iside]]
				self.boundary[iside].ncomp = 1

    		#! boundary(1).dofs = sub2ind (sp.ndof_dir, ones(1,ncp), 1:ncp);
    		#! boundary(2).dofs = sub2ind (sp.ndof_dir, mcp*ones(1,ncp), 1:ncp);
    		#! boundary(3).dofs = sub2ind (sp.ndof_dir, 1:mcp, ones(1,mcp));
    		#! boundary(4).dofs = sub2ind (sp.ndof_dir, 1:mcp, ncp*ones(1,mcp));
			#print mcp*np.ones(ncp,dtype=np.uint)
			self.boundary[0].dofs = np.ravel_multi_index((np.zeros(ncp,dtype=np.int)		,np.arange(ncp)						),dims=self.ndof_dir, order='F')
			self.boundary[1].dofs = np.ravel_multi_index((mcp*np.ones(ncp,dtype=np.int)-1	,np.arange(ncp)						),dims=self.ndof_dir, order='F')
			self.boundary[2].dofs = np.ravel_multi_index((np.arange(mcp,dtype=np.int)		,np.zeros(mcp, dtype=np.int)		),dims=self.ndof_dir, order='F')
			self.boundary[3].dofs = np.ravel_multi_index((np.arange(mcp,dtype=np.int)		,ncp*np.ones(mcp,dtype=np.int)-1	),dims=self.ndof_dir, order='F') # typo here? ncp
		else:
			self.boundary = None

		self.nsh = []
		self.connectivity = []
		self.shape_functions = []
		self.shape_function_gradients = []

		#self.constructor = lambda (self.knots, self.__init__
		# needed? nay
		self.constructor = self.__init__

		
		# Actually, this function is not for 2d spaces only, but for n-d
	def evaluate_col(self, mmsh, value=True, gradient=False, hessian=False):
		"compute basis functions in one column of the mesh"
		if __debug__:
			print "DBG: space evaluate col. values = {}, gradients = {}".format(value, gradient)

		# returns discrete function space... 
		# if mmsh.colnum > 0:
		#import pdb; pdb.set_trace()
		sp = self.evaluate_col_param(mmsh, value=value, gradient=gradient, hessian=hessian)

		# 2. if(hessian)... TODO

		# 3. if gradient
		# - JinvT  = geopdes_invT__(msh.geo_map_jac) ?
		# 	in utils/ undocumented.
		# - geopdes_prod__(JinvT, self.shape_function_gradients) ?
		#	short
		#	in utils/ undocumented
		if gradient:
			# untested
			# import traceback
			# traceback.print_stack()
			# sys.exit(4)
			# untested yet
			JinvT, det = util.calc_JinvT(mmsh.geo_map_jac)
			JinvT = np.reshape(JinvT, (2,2, mmsh.nqn, mmsh.nel) ,order='F')
			sp.shape_function_gradients = util.calc_prod(JinvT, sp.shape_function_gradients)



		return sp

	
		# Actually, this function is not for 2d spaces only, but for n-d
	def evaluate_col_param (self, mmsh, value = True, gradient = False, hessian = False):
		"compute basis functions in the parametric domain, in one column of the mesh"

		spu = self.spu # apparently there is a diff in the refere
		spv = self.spv

		ndof = spu.ndof * spv.ndof
		ndof_dir = np.array([spu.ndof, spv.ndof])

		#nsh = spu.nsh[mmsh.colnum] * spv.nsh
		nsh = spu.nsh[mmsh.colnum] * spv.nsh
		# nsh = nsh.T

		conn_u = np.reshape(spu.connectivity[:,mmsh.colnum] , (spu.nsh_max, 1, 1), order = 'F')
		conn_u = np.tile(conn_u, [1, spv.nsh_max, mmsh.nel])
		conn_u = np.reshape(conn_u, (-1, mmsh.nel), order = 'F')

		# conn_v = np.reshape(spv.connectivity[:,mmsh.colnum] , (1, spv.nsh_max, mmsh.nel_dir[1]), order = 'F')
		conn_v = np.reshape(spv.connectivity , (1, spv.nsh_max, mmsh.nel_dir[1]), order = 'F')
		conn_v = np.tile(conn_v, [spu.nsh_max, 1, 1])
		conn_v = np.reshape(conn_v, (-1, mmsh.nel), order = 'F')

		connectivity = np.ones((self.nsh_max, mmsh.nel),dtype=np.int) * (-1)
		conn_geopdes = np.ones((self.nsh_max, mmsh.nel),dtype=np.int) * (-1)
		indices = np.logical_and(conn_u >= 0, conn_v >= 0) # this code relies on the fact that connectivity 0 is not allowed. in my code it is allowed, but negative indices are not.
		#import pdb; pdb.set_trace()

		conn_geopdes[indices] = np.ravel_multi_index((conn_u[indices].astype(np.int), conn_v[indices].astype(np.int)),dims=(spu.ndof, spv.ndof), order='F')
		conn_geopdes = np.reshape(conn_geopdes, (self.nsh_max, mmsh.nel), order = 'F')
		# using C order, such that I can use ravel()
		# use conn.ravel()[lin_index] 
		connectivity[indices] = np.ravel_multi_index((conn_u[indices].astype(np.int), conn_v[indices].astype(np.int)),dims=(spu.ndof, spv.ndof), order='C')
		connectivity = np.reshape(connectivity, (self.nsh_max, mmsh.nel), order = 'F')

		del conn_u, conn_v

		sp = OneSpace()
		sp.nsh_max = self.nsh_max

		#import pdb; pdb.set_trace()
		sp.nsh = nsh.flatten().astype(np.int) #flatten should not hurt..
		sp.ndof = ndof
		sp.ndof_dir = ndof_dir
		sp.connectivity=conn_geopdes
		sp.ncomp = 1

		# assert mmsh.colnum == 0

		#shp_u = np.reshape(spu.shape_functions[:,:,mmsh.colnum], (mmsh.nqn_dir[0], 1, spu.nsh_max, 1, 1))
		shp_u = np.reshape(spu.shape_functions[:,:,mmsh.colnum], (mmsh.nqn_dir[0], 1, spu.nsh_max, 1, 1), order = 'F')
		shp_u = np.tile (shp_u, [1, mmsh.nqn_dir[1], 1, spv.nsh_max, mmsh.nel]) # repeat spv.nsh_max times
		shp_u = np.reshape(shp_u, (mmsh.nqn, self.nsh_max, mmsh.nel), order = 'F')

		shp_v = np.reshape(spv.shape_functions, (1, mmsh.nqn_dir[1], 1, spv.nsh_max, mmsh.nel), order = 'F')
		shp_v = np.tile (shp_v, [mmsh.nqn_dir[0], 1, spu.nsh_max, 1, 1])
		shp_v = np.reshape(shp_v, (mmsh.nqn, self.nsh_max, mmsh.nel), order = 'F')

		if value:
			sp.shape_functions = shp_u * shp_v #why?

		if gradient or hessian:

			#FIXME shape_function_gradients
			# print spu.shape_function_gradients.shape
			#import pdb; pdb.set_trace()
			shg_u = np.reshape(spu.shape_function_gradients[:,:,mmsh.colnum], (mmsh.nqn_dir[0], 1, spu.nsh_max, 1, 1), order = 'F')
			shg_u = np.tile (shg_u, [1, mmsh.nqn_dir[1], 1, spv.nsh_max, mmsh.nel])
			shg_u = np.reshape(shg_u, (mmsh.nqn, self.nsh_max, mmsh.nel), order = 'F')
	
			shg_v = np.reshape(spv.shape_function_gradients, (1, mmsh.nqn_dir[1], 1, spv.nsh_max, mmsh.nel), order = 'F')
			shg_v = np.tile (shg_v, [mmsh.nqn_dir[0], 1, spu.nsh_max, 1, 1])
			shg_v = np.reshape(shg_v, (mmsh.nqn, self.nsh_max, mmsh.nel), order = 'F')

			# combine in 4d array: 0 gradients, 1 base
			sp.shape_function_gradients = np.array([shg_u * shp_v, shp_u * shg_v])
			# print sp.shape_function_gradients.shape # 2 16 16 2
			sp.shape_function_gradients = np.reshape(sp.shape_function_gradients, (2, mmsh.nqn, sp.nsh_max, mmsh.nel),order ='F')

			del shg_v, shg_u # unnecessaryNay, that' constant in my case

			# print sp.shape_function_gradients.shape # 2 16 16 2
			# print np.all(sp.shape_function_gradients[0] == shg_u * shg_v) True

			# if hessian and hasattr(....)
			# TODO


		return sp

	def sp_drchlt_l2_proj (self, mmsh, h, sides):
		"SP_DRCHLT_L2_PROJ: assign the degrees of freedom of Dirichlet boundaries through an L2 projection."

		if __debug__:
			print "DBG: Dirichlet l2 projection"
	
		rhs = np.zeros(self.ndof)
		# print h(rhs+1., rhs+np.pi/2, 0)

		nent = 0
		dofs = np.array([],dtype=np.uint)
		for iside in sides:
			nent = nent + mmsh.bnd[iside].nel * self.boundary[iside].nsh_max**2
			dofs = np.union1d(dofs, self.boundary[iside].dofs.astype(np.uint))

		rows = np.zeros(nent)
		cols = np.zeros(nent)
		vals = np.zeros(nent)
		ncounter = 0
		for iside in sides:

			msh_bnd = mmsh.eval_boundary_side(iside) 
			sp_bnd = self.eval_boundary_side(msh_bnd)

			if msh_bnd.geo_map.shape[0] == 2: # 2d
				x = np.squeeze (msh_bnd.geo_map[0])
				y = np.squeeze (msh_bnd.geo_map[1])
				hval = np.reshape( h(x,y, iside), [self.ncomp, msh_bnd.nqn, msh_bnd.nel], order ='F')

			elif msh_bnd.geo_map.shape[0] == 3: # 2d
				return -1

			# from pprint import pprint
			# pprint (vars(sp_bnd))
			#import pdb; pdb.set_trace()
			rs, cs, vs = op_u_v.op_u_v(sp_bnd, sp_bnd, msh_bnd, np.ones((msh_bnd.nqn, msh_bnd.nel)) ,matrix=False)
			# M = op_u_v.op_u_v(sp_bnd, sp_bnd, msh_bnd, np.ones((msh_bnd.nqn, msh_bnd.nel)) ,matrix=True)

			#print sp_bnd.dofs[rs]  # 1,2,3,4
			# TODO: this value comes from the connectivity array. but removing +1 above results in error... must be changed tough
			rows[ncounter+np.arange(rs.size)] = sp_bnd.dofs[rs]
			cols[ncounter+np.arange(rs.size)] = sp_bnd.dofs[cs]
			vals[ncounter+np.arange(rs.size)] = vs
			ncounter += rs.size
			#sys.exit(34)
			
			# TODO: check this values
			# import pdb; pdb.set_trace()
			rhs_side = op_f_v.op_f_v(sp_bnd, msh_bnd, hval)

			# import pdb; pdb.set_trace()
			rhs[sp_bnd.dofs] += rhs_side

	 	# csc_matrix( (data,(row,col)), shape=(3,3) )
		# import pdb; pdb.set_trace()
		M = csc_matrix( (vals, (rows, cols)))
		# print M[dofs,dofs].shape #  1,12
		# print rhs.shape # 16,
		# print dofs # 0-4 7-11 13-15

		
		from numpy.linalg import solve, norm
		# print M.todense()[dofs[:,None],dofs].shape
		u_ = solve(M.todense()[dofs[:,None],dofs], rhs[dofs])

		u = spsolve(M[dofs[:,None],dofs], rhs[dofs]) #,0 diff ref
		err = norm(u-u_)
		if err > 1e-10:
			print "WARNING: dense solution of Dirichlet conditions does not equal sparse solution"
		# print u_
		# print u
		# 
		# print M.todense()[dofs[None,:],dofs[:,None]]
		# print M[dofs[None,:],dofs[:,None]]
		# TODO: check this, but only once the vals array above is correct

			# rows, cols vals ncounter
			
		# M ... Mass matrix??? depends on rows, cols vals (I don't care for now), but u = M(dofs,dofs) \ rhs(dofs, 1)...
		# I don't yet get it... don't I simply have to evaluate the boundary nodes and set them to their given Dirichlet values...???
	
		# print u
		# print dofs
		return u, dofs

	def eval_boundary_side(self, msh_side):
		 "SP_EVAL_BOUNDARY_SIDE: Construct the space structure of one side of the boundary."
		 
		 iside = msh_side.side_number # 0, 1, 2, 3
		 ind = np.mod(np.floor((iside+2)/2),2).astype(np.uint) #+1
		 # print 'eval_bnd_side'
		 bnodes = np.reshape( np.squeeze(msh_side.quad_nodes[ind]), [msh_side.nqn, -1], order='F')
		 sp_side = self.sp_bspline_1d_param(self.knots[ind], self.degree[ind], bnodes, gradient=False)

		 sp_side.dofs = self.boundary[iside].dofs
		 return sp_side

	def sp_eval(self, u, geometry, npts, value=True, gradient=False, curl=False, divergence=False):
		"SP_EVAL: Compute the value or the derivatives of a function, given by its degrees of freedom, at a given set of points."

		ndim = len(npts)

		# import pdb; pdb.set_trace()
		if isinstance(npts, list):
			pts = npts[:]
			# list comprehension
			npts = np.array([ x.size for x in pts ], dtype=np.int )
		#elif # TODO

		brk = []
		for jj in np.arange(ndim):
			if pts[jj].size > 1:
				b = pts[jj][:-1] + np.diff(pts[jj])/2
				b = np.insert(b,0,0)
				b = np.append(b,1)
				# print b # ok
			else:
				b = np.array([0,1])
			brk.append(b)

		from msh import msh
		# TODO: move to BspSpace2d class
		if ndim == 2:
			# TESTING (seems to work)
			# print pts[0]
			pts[0] = pts[0][None,:]
			pts[1] = pts[1][None,:]
			
			# import pdb; pdb.set_trace()
			# one could ommit the creation of this mesh, as 'sp' just really requires `pts` and 
			mmsh = msh.Msh2d(breaks=brk, qn=pts, qw=None, geometry=geometry, boundary=False) # no quadrature weights
			sp = BspSpace2d(knots=self.knots, degree=self.degree, msh=mmsh) 
		elif ndim == 3:
			return -1

		if value: # default yes
			eu, F = sp.eval_msh(u, mmsh) 
			# import pdb; pdb.set_trace()
			F = np.reshape(F, np.insert(npts, 0 , ndim), order='F')
			eu = np.squeeze(np.reshape(eu, np.insert(npts, 0, sp.ncomp), order='F'))

		# if gradient, divergence, curl... TODO:

		return eu, F

	def h1_error(self, mmsh, u, uex, graduex):
		"SP_H1_ERROR: Evaluate the error in H^1 norm."
		if __debug__:
			print "DBG: H1 error"

		ndim = self.ndim
		errl2 = 0.
		errh1s = 0.

		valu = np.zeros((self.ncomp, mmsh.nqn, mmsh.nelcol))
		grad_valu = np.ones((self.ncomp, ndim, mmsh.nqn, mmsh.nelcol), dtype=np.float64) * -1.

		for iel in np.arange(mmsh.nel_dir[0]):
			msh_col = mmsh.evaluate_col(iel)
			sp_col = self.evaluate_col(msh_col, gradient=True)

			uc_iel = np.zeros(sp_col.connectivity.shape)
			uc_iel[sp_col.connectivity >= 0] = u[sp_col.connectivity[sp_col.connectivity >= 0]]

			weight = np.tile( np.reshape(uc_iel, [1, sp_col.nsh_max, msh_col.nel] ,order='F'), [msh_col.nqn, 1, 1])
			sp_col.shape_functions = np.reshape(sp_col.shape_functions, (sp_col.ncomp, msh_col.nqn, sp_col.nsh_max, msh_col.nel), order='F')
			sp_col.shape_function_gradients = np.reshape(sp_col.shape_function_gradients, (sp_col.ncomp, ndim, msh_col.nqn, sp_col.nsh_max, msh_col.nel), order='F')

			# shape of sp_col.shape_function_gradients[0,0] is 16, 16, 1
			# summing over axis 1 reduces it to 16,1 
			for icmp in np.arange(self.ncomp):
				for idim in np.arange(ndim): # idim loop because gradient has one dimension more (which is `ndim` in size, because of derivation)

					grad_valu[icmp,idim] = np.sum(weight * np.reshape(sp_col.shape_function_gradients[icmp, idim, :, :, :],
						[msh_col.nqn, sp_col.nsh_max, msh_col.nel],order='F'), axis=1)

				valu[icmp] = np.sum(weight * np.reshape(sp_col.shape_functions[icmp], 
					[msh_col.nqn, sp_col.nsh_max, msh_col.nel]), axis = 1) # weight is from solution vector. Understanding this product is very important. It's the space evaluation.

				## passt.

			x = []
			#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
			# import pdb; pdb.set_trace()
			for idim in np.arange(ndim):
				### shape of geo_map is correct, but values are mixed?
				x.append( np.reshape(msh_col.geo_map[idim], [msh_col.nqn, msh_col.nel],order='F'))
				### this results in an error at the evaluation of uex
				### most probably a wrong qnu/qnv is called in msh_evaluate_col

			w = msh_col.quad_weights * msh_col.jacdet

			valex = np.reshape( uex (x[0], x[1]), [self.ncomp, msh_col.nqn, msh_col.nel], order = 'F')
			# values on wrong positions shape correct
			grad_valex = np.reshape( graduex (x[0], x[1]), [self.ncomp, ndim, msh_col.nqn, msh_col.nel], order='F')
			# import pdb; pdb.set_trace()

			errh1s += np.sum( 
				np.reshape( 
					np.sum( np.sum(
						(grad_valu - grad_valex)**2 # shape 1 2 16 1
							,axis=0) 				# shape 2, 16, 1
							,axis=0), 				# shape 16, 1
					[msh_col.nqn, msh_col.nel], order='F') 
				* w )

			erraux = np.sum((valu - valex)**2, axis=0)
			errl2 += np.sum( w * erraux)


		errh1 = np.sqrt(errl2 + errh1s)
		errl2 = np.sqrt(errl2)
		return errh1, errl2



class DummyCol(object):
	'''dummy msh_col'''



	
if __name__ == '__main__':
	''' test
	'''
	
