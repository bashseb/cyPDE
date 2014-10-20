#!/usr/bin/env python2
from __future__ import division
import numpy as np
import sys
sys.path.insert(0, '../nurbs_toolbox/')
#from nurbs import _findspan, basisfunder

#  SP_DRCHLT_L2_PROJ: assign the degrees of freedom of Dirichlet boundaries through an L2 projection.
# 
#    [u, dofs] = sp_drchlt_l2_proj (sp, msh, h, sides)
# 
#  INPUT:
# 
#   sp:    object defining the space of discrete functions (see sp_bspline_2d)
#   msh:   object defining the domain partition and the quadrature rule (see msh_2d)
#   h:     function handle to compute the Dirichlet condition
#   sides: boundary sides on which a Dirichlet condition is imposed
# 
#  OUTPUT:
# 
#   u:    assigned value to the degrees of freedom
#   dofs: global numbering of the corresponding basis functions
# 
#  Copyright (C) 2010 Carlo de Falco, Rafael Vazquez
#  Copyright (C) 2011 Rafael Vazquez
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

def sp_drchlt_l2_proj (sp, msh, h, sides):

	rhs = np.zeros(sp.ndof)
	# print h(rhs+1., rhs+np.pi/2, 0)

	return rhs, rhs

# class BspSpace(object):
# 	"b-spline space base class"
# 	spCnt = 0
# 	def __init__(self, knots, degree, msh):
# 
# 		BspSpace.spCnt +=1
# 		self.knots = knots
# 		self.degree = degree
# 		self.nodes = msh.qn
# 		self.ndof = -1
# 		self.ndof_dir = -1
# 		self.nsh_max = -1
# 		self.nsh_dir = -1
# 		self.ncomp = 1
# 		self.boundary = msh.bnd
# 
# 	# should this be a class function or outside helper function? Or inside OneSpace class?
# 	def sp_bspline_1d_param(self, knots, degree, nodes, gradient=True, hessian=False):
# 		'''Construct a space of B-Splines on the parametric domain in 1D.'''
# 		
# 		nders = 0
# 		if hessian:
# 			nders = 2
# 		elif gradient:
# 			nders = 1
# 
# 		sp = OneSpace()
# 
# 		# print knots
# 		# print degree
# 		# print nodes.shape #2d shape=(4,1)
# 		# print nodes.flatten()
# 		mknots = knots.size -1
# 
# 		p = degree
# 		mcp = - p - 1 + mknots
# 		ndof = mcp + 1
# 		nel = nodes.shape[1]
# 		nqn = nodes.shape[0]
# 
# 		nsh = np.ones((1,nel))*(-1)
# 		assert nel == 1
# 
# 		connectivity = []
# 		# connectivity: numbers associated 
# 		for iel in np.arange(nel):
# 			s = _findspan (mcp, p, nodes[:,iel].conj().transpose(), knots)
# 			# numbasisfun
# 			c = np.transpose((s-p) + np.reshape(np.arange(p+1), (p+1,1))) # using broadcasting rules
# 			### I think '+1' is from matlab indexing
# 			c = np.unique(c.flatten(order='F'))+1
# 			if __debug__:
# 				print "WARN connectivity array +1"
# 			# c = np.unique(c.flatten(order='F'))
# 			connectivity.append(c) 
# 			# connectivity[:,iel] = c
# 			nsh[iel] = (c != 0).sum()
# 
# 		# assumption: all 'c's above are of same size
# 		assert nel == 1 # else needs to test connectivity code below
# 		ccn = np.zeros((c.size, nel))
# 		for iel in np.arange(nel):
# 			ccn[:,iel] = connectivity[iel]
# 
# 		sp.nsh = nsh
# 		sp.nsh_max = np.max(nsh)
# 		sp.ndof = ndof
# 		sp.connectivity = ccn
# 		sp.ncomp = 1 #??
# 
# 		assert nel == 1 # must pass 1d array to _findspan
# 		s = _findspan(mcp, p, nodes.flatten(), knots)
# 
# 		# Note: unpythonic...
# 		tders = np.zeros((s.size,nders+1,p+1)) 
# 		for i in np.arange(s.size):
# 			# print "s={}, p={}, node={}, knots ={}, nders={}".format(s[i], p, nodes[i], knots, nders)
# 			tders[i] = basisfunder( s[i], p, nodes[i], nders, knots)
# 		# print tders.shape
# 		# print tders
# 		nbf = np.transpose((s-p) + np.reshape(np.arange(p+1), (p+1,1))) # broadcasting
# 
# 		if __debug__:
# 			print "WARN connectivity array +1"
# 		# worried if this is correct
# 		nbf = np.reshape(nbf+1, (s.shape[0], 1, p+1), order ='F')
# 		# print nbf
# 		ders = np.zeros((nodes.size,nders+1,sp.nsh_max))
# 
# 		# for inqn in np.arrange(nodes.size):
# 		# 	ir, iel = ind2sub(
# 		### FIXME: skipping calcuation of this sht, because in my case ders == tders.
# 		#########################################
# 		if __debug__:
# 			print "WARNING: implicit assumption of ders==tders in spBspline.py"
# 		ders = tders
# 		#print ders[:,0,:]
# 		sp.shape_functions = ders[:,0,:]
# 
# 		if (gradient):
# 			sp.shape_function_gradients = ders[:,1,:]
# 
# 		if (hessian):
# 			sp.shape_function_hessians = ders[:,2,:]
# 				
# 		return sp
# 
# # def numbasisfun(i, u, p, U):
# 
# 
# 
# class BspSpace2d(BspSpace):
# 	"2d bspline space class"
# 	def __init__(self, knots, degree, msh):
# 
# 		assert len(knots) == 2
# 		assert degree.size == 2
# 		# TODO: validate msh object
# 		self.ndim = 2
# 		super( BspSpace2d, self).__init__(knots, degree, msh)
# 		#kself.spu
# 
# 		# TODO: sp_bspline_1d_param is only capable of 1d nodes arrays
# 		self.spu = self.sp_bspline_1d_param(self.knots[0], self.degree[0], self.nodes[0], gradient=True, hessian=True)
# 		self.spv = self.sp_bspline_1d_param(self.knots[1], self.degree[1], self.nodes[1], gradient=True, hessian=True)
# 
# # 		test case which currently fails sp_bspline_1d_param
# # 		knots = np.array([0, 0, 0, .5, 1, 1, 1])
# # 		degree = 2
# # 		points = np.array([[0, .1, .2],[.6, .7, .8]])
# # 		testsp = self.sp_bspline_1d_param(knots, degree, points, hessian=True)
# 
# 		self.nsh_max = self.spu.nsh_max * self.spv.nsh_max
# 		self.nsh_dir = np.array([self.spu.nsh_max, self.spv.nsh_max])
# 		self.ndof = self.spu.ndof * self.spv.ndof
# 		self.ndof_dir = np.array([self.spu.ndof, self.spv.ndof])
# 		self.ncomp = 1
# 		mcp = self.ndof_dir[0]
# 		ncp = self.ndof_dir[1]
# 
# 		# TODO: boundary
# 		if self.boundary is not None:
# 			ind = np.array([1, 1, 0, 0])
# 			for iside in np.arange(len(self.boundary)):
# 				print self.boundary[iside].breaks
# 				self.boundary[iside].ndof = self.ndof_dir[ind[iside]]
# 				self.boundary[iside].nsh_max = self.nsh_dir[ind[iside]]
# 				self.boundary[iside].ncomp = 1
# 
# 		# for the moment, hardcoded:
# 
#     	#! boundary(1).dofs = sub2ind (sp.ndof_dir, ones(1,ncp), 1:ncp);
#     	#! boundary(2).dofs = sub2ind (sp.ndof_dir, mcp*ones(1,ncp), 1:ncp);
#     	#! boundary(3).dofs = sub2ind (sp.ndof_dir, 1:mcp, ones(1,mcp));
#     	#! boundary(4).dofs = sub2ind (sp.ndof_dir, 1:mcp, ncp*ones(1,mcp));
# 		self.boundary[0] = np.array([0, 4, 8, 12])
# 		self.boundary[1] = np.array([3, 7, 11, 15])
# 		self.boundary[2] = np.array([0, 1, 2, 3])
# 		self.boundary[3] = np.array([12, 13, 14, 15])
# 
# 		self.nsh = []
# 		self.connectivity = []
# 		self.shape_functions = []
# 		self.shape_function_gradients = []
# 
# 		#self.constructor = lambda (self.knots, self.__init__
# 		# needed? nay
# 		self.constructor = self.__init__
# 
# 		
# 		# Actually, this function is not for 2d spaces only, but for n-d
# 	def evaluate_col(self, msh, value=True, gradient=False, hessian=False):
# 		"compute basis functions in one column of the mesh"
# 
# 		# returns discrete function space... 
# 
# 		sp = self.evaluate_col_param(msh, value=value, gradient=gradient, hessian=hessian)
# 
# 		# 2. if(hessian)... TODO
# 
# 		# 3. if gradient
# 		# - JinvT  = geopdes_invT__(msh.geo_map_jac) ?
# 		# 	in utils/ undocumented.
# 		# - geopdes_prod__(JinvT, self.shape_function_gradients) ?
# 		#	short
# 		#	in utils/ undocumented
# 		if gradient:
# 			JinvT, det = calc_JinvT(msh.geo_map_jac)
# 			JinvT = np.reshape(JinvT, (2,2, msh.nqn, msh.nel) ,order='F')
# 			#print JinvT.shape
# 			#print sp.shape_function_gradients.shape
# 			#print sp.shape_function_gradients[0,15,:]
# 			sp.shape_function_gradients = calc_prod(JinvT, sp.shape_function_gradients)
# 
# 
# 
# 		return sp
# 
# 	
# 
# 
# 		# Actually, this function is not for 2d spaces only, but for n-d
# 	def evaluate_col_param (self, msh, value = True, gradient = False, hessian = False):
# 		"compute basis functions in the parametric domain, in one column of the mesh"
# 
# 		spu = self.spu
# 		spv = self.spv
# 
# 		ndof = spu.ndof * spv.ndof
# 		ndof_dir = np.array([spu.ndof, spv.ndof])
# 
# 		nsh = spu.nsh[msh.colnum] * spv.nsh
# 		# nsp = nsh.T
# 
# 		# print spu.connectivity[:,msh.colnum]
# 		conn_u = np.reshape(spu.connectivity[:,msh.colnum] , (spu.nsh_max, 1, 1), order = 'F')
# 		conn_u = np.tile(conn_u, [1, spv.nsh_max, msh.nel])
# 		conn_u = np.reshape(conn_u, (-1, msh.nel), order = 'F')
# 		#print conn_u
# 
# 		conn_v = np.reshape(spv.connectivity[:,msh.colnum] , (1, spv.nsh_max, msh.nel_dir[1]), order = 'F')
# 		conn_v = np.tile(conn_v, [spv.nsh_max, 1, 1])
# 		conn_v = np.reshape(conn_v, (-1, msh.nel), order = 'F')
# 
# 		connectivity = np.zeros((self.nsh_max, msh.nel))
# 		conn_geopdes = np.zeros((self.nsh_max, msh.nel))
# 		indices = np.logical_and(conn_u != 0, conn_v != 0)
# 
# 		conn_geopdes[indices] = np.ravel_multi_index((conn_u[indices].astype(np.int)-1, conn_v[indices].astype(np.int)-1),dims=(spu.ndof, spv.ndof), order='F')
# 		conn_geopdes = np.reshape(conn_geopdes, (self.nsh_max, msh.nel), order = 'F')
# 		# using C order, such that I can use ravel()
# 		# use conn.ravel()[lin_index] 
# 		connectivity[indices] = np.ravel_multi_index((conn_u[indices].astype(np.int)-1, conn_v[indices].astype(np.int)-1),dims=(spu.ndof, spv.ndof), order='C')
# 		# connectivity[indices] = np.ravel_multi_index((0,1),dims=(spu.ndof, spv.ndof), order='F')
# 		connectivity = np.reshape(connectivity, (self.nsh_max, msh.nel), order = 'F')
# 
# 		del conn_u, conn_v
# 
# 		sp = OneSpace()
# 		sp.nsh_max = self.nsh_max
# 		sp.nsh = nsh.astype(np.int)
# 		sp.ndof = ndof
# 		sp.ndof_dir = ndof_dir
# 		sp.connectivity=conn_geopdes
# 		sp.ncomp = 1
# 
# 
# 		
# 		assert msh.colnum == 0
# 		# arrrrrrrrrrr bad bad bad. perhaps because of ders=tders, I'm have to hack this )
# 		# print spu.shape_functions.shape # can be three dimensional array, in 2d scalar laplace, it's 2d however.
# 
# 		#shp_u = np.reshape(spu.shape_functions[:,:,msh.colnum], (msh.nqn_dir[0], 1, spu.nsh_max, 1, 1))
# 		shp_u = np.reshape(spu.shape_functions, (msh.nqn_dir[0], 1, spu.nsh_max, 1, 1), order = 'F')
# 		shp_u = np.tile (shp_u, [1, msh.nqn_dir[1], 1, spv.nsh_max, msh.nel])
# 		shp_u = np.reshape(shp_u, (msh.nqn, self.nsh_max, msh.nel), order = 'F')
# 
# 		# print shp_u
# 		shp_v = np.reshape(spv.shape_functions, (1, msh.nqn_dir[1], 1, spv.nsh_max, msh.nel), order = 'F')
# 		shp_v = np.tile (shp_v, [msh.nqn_dir[0], 1, spu.nsh_max, 1, 1])
# 		shp_v = np.reshape(shp_v, (msh.nqn, self.nsh_max, msh.nel), order = 'F')
# 
# 		#print shp_v
# 
# 		if value:
# 			sp.shape_functions = shp_u * shp_v
# 
# 
# 		if gradient or hessian:
# 
# 			#FIXME shape_function_gradients[:,:,numcol]
# 			# print spu.shape_function_gradients.shape
# 			shg_u = np.reshape(spu.shape_function_gradients, (msh.nqn_dir[0], 1, spu.nsh_max, 1, 1), order = 'F')
# 			shg_u = np.tile (shg_u, [1, msh.nqn_dir[1], 1, spv.nsh_max, msh.nel])
# 			shg_u = np.reshape(shg_u, (msh.nqn, self.nsh_max, msh.nel), order = 'F')
# 	
# 			# print shp_u
# 			shg_v = np.reshape(spv.shape_function_gradients, (1, msh.nqn_dir[1], 1, spv.nsh_max, msh.nel), order = 'F')
# 			shg_v = np.tile (shg_v, [msh.nqn_dir[0], 1, spu.nsh_max, 1, 1])
# 			shg_v = np.reshape(shg_v, (msh.nqn, self.nsh_max, msh.nel), order = 'F')
# 
# 			# print shg_u.shape
# 			# print shg_v.shape
# 			# combine in 4d array: 0 gradients, 1 base
# 			sp.shape_function_gradients = np.array([shg_u * shg_v, shp_u * shp_v])
# 			# print sp.shape_function_gradients.shape # 2 16 16 2
# 			sp.shape_function_gradients = np.reshape(sp.shape_function_gradients, (2, msh.nqn, sp.nsh_max, msh.nel),order ='F')
# 
# 			del shg_v, shg_u # unnecessary
# 
# 			# print sp.shape_function_gradients.shape # 2 16 16 2
# 			# print np.all(sp.shape_function_gradients[0] == shg_u * shg_v) True
# 
# 			# if hessian and hasattr(....)
# 			# TODO
# 
# 
# 		return sp
# 
# 
# 
# def calc_JinvT(v):
# 	assert(v.shape[0] == 2)
# 	# 2D
# 	if v.shape[0] == 2:
# 		det = v[0,0,:,:] * v[1,1,:,:] - v[1,0,:,:] * v[0,1,:,:]
# 		#print det.shape
# 		# print v[1,1,:,:].shape
# 		JinvT = np.empty(v.shape)
# 
# 		JinvT[0,0,:,:] = v[1,1,:,:]/det
# 		JinvT[1,1,:,:] = v[0,0,:,:]/det
# 		JinvT[0,1,:,:] = -v[1,0,:,:]/det
# 		JinvT[1,0,:,:] = -v[0,1,:,:]/det
# 
# 		det = np.squeeze(det)
# 	return JinvT, det
# 
# def calc_prod(a, b):
# 	aux_dim = (a.shape[0], a.shape[2], b.shape[2], b.shape[3])
# 
# 	a = np.reshape(a, (a.shape[0], a.shape[1], a.shape[2], 1, a.shape[3]), order = 'F')
# 	b = np.reshape(b, np.insert(b.shape, 0, 1), order ='F')
# 	# print 'calc_prod'
# 	# row sum
# 	return np.reshape(np.sum(a*b,axis=1), newshape = aux_dim, order='F') # broadcasting
	
if __name__ == '__main__':
	''' test
	'''
	
