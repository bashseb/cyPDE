#!/usr/bin/env python2
from __future__ import division
import numpy as np
import sys
# sys.path.insert(0, '../cyPDEs/util/') # schiach!
# print sys.path
# TODO: I'd like to do a relative import... like from .. import util, but it doesnt work
from util import util

def msh_gauss_nodes(nnodes):
# MSH_GAUSS_NODES: define nodes and weights for a multi-dimensional tensor-product Gauss quadrature rule.
	
	if __debug__:
		print "msh_gauss_nodes, create gauss rule"
	# import pdb; pdb.set_trace()
	rule = []
	for idir in np.arange(nnodes.size):
		rule.append(np.array(grule(nnodes[idir])))
	
	return rule

def grule(n):
	# sourced from 
	# http://nullege.com/codes/show/src@w@a@wafo-0.11@src@wafo@integrate.py
	m = int(np.fix((n+1)/2))
	
	mm = 4*m-1
	t  = (np.pi/(4*n+2))*np.arange(3,mm+1,4)
	nn = (1-(1-1/n)/(8*n*n))
	xo = nn*np.cos(t)
	# Algorithm given by Davis and Rabinowitz in 'Methods
	# of Numerical Integration', page 365, Academic Press, 1975.
	  
	e1   = n*(n+1)
	
	for j in xrange(2):
	    pkm1 = 1
	    pk   = xo
	    for k in xrange(2,n+1):
	        t1   = xo*pk
	        pkp1 = t1-pkm1-(t1-pkm1)/k+t1
	        pkm1 = pk
	        pk   = pkp1
	
	    den = 1.-xo*xo
	    d1  = n*(pkm1-xo*pk)
	    dpn = d1/den
	    d2pn = (2.*xo*dpn-e1*pk)/den
	    d3pn = (4.*xo*d2pn+(2-e1)*dpn)/den
	    d4pn = (6.*xo*d3pn+(6-e1)*d2pn)/den
	    u = pk/dpn
	    v = d2pn/dpn
	    h = -u*(1+(.5*u)*(v+u*(v*v-u*d3pn/(3*dpn))))
	    p = pk+h*(dpn+(.5*h)*(d2pn+(h/3)*(d3pn+.25*h*d4pn)))
	    dp = dpn+h*(d2pn+(.5*h)*(d3pn+h*d4pn/3))
	    h  = h-p/dp
	    xo = xo+h
	
	x = -xo-h
	fx = d1-h*e1*(pk+(h/2)*(dpn+(h/3)*(d2pn+(h/4)*(d3pn+(.2*h)*d4pn))))
	w = 2*(1-x**2)/(fx**2)
	if (m+m) > n:
	    x[m-1] = 0.0
	
	if not ((m+m) == n):
	    m = m-1
	
	x = np.hstack((x,-x[m-1::-1]))
	w = np.hstack((w,w[m-1::-1]))
	return x,w

def set_quad_nodes(knots, rule, limits=None):
	" MSH_SET_QUAD_NODES: compute location and weights of quadrature nodes in a tensor product grid."
	if __debug__:
		print "DEBUG: set quad_nodes. location and weights of quadrature nodes in tp grid"
	if limits == None:
		limits = np.array([-1,1])
	# import pdb; pdb.set_trace()

	qn = []
	qw = [] 
	for idir in np.arange(len(rule)):
		uniqueknots = np.unique(knots[idir])
		uniqueknotsl = uniqueknots[0:-1]
		du = np.diff(uniqueknots, n=1, axis=0)
		nel = (uniqueknots.size-1)
		p1 = rule[idir][0] # quadrature nodes
		w1 = rule[idir][1] # quadrature weights
		nqn = p1.size

		# print np.transpose(np.tile( p1-limits[0], (nel, 1))) # + np.tile( du/np.diff(limits), (nqn, 1)))
		qn.append(	np.tile( uniqueknotsl, (nqn, 1)) + np.transpose(np.tile( p1-limits[0], (nel, 1))) * np.tile( du/np.diff(limits), (nqn, 1)))

		qw.append(  np.transpose(np.tile (w1, (nel, 1))) * np.tile ( du/np.diff(limits), (nqn,1)))

	return qn, qw

class Msh(object):
	"Mesh base class"
	mshCnt = 0
	def __init__(self, breaks, qn, qw, geometry, boundary=True, der2=False ):
		assert isinstance(qn, list)
		assert isinstance(qw, list) or qw is None
		assert isinstance(breaks, list)

		# TODO sanitize geometry

		Msh.mshCnt += 1

		self.boundary = boundary
		self.der2 = der2

		self.ndim = len(qn) # 2 for 2d
		self.nel = -1 # no of elements of the partition
		# self.nelcol = 0 # scalar no of elements ..
		self.nqn = -1 # no of quadrature nodes per element
		self.qn = qn # nqn x nel
		self.qw = qw
		self.geo = geometry
		self.quad_nodes = None
		self.quad_weights = None
		self.jacdet = []
	
	def geo_map(self, pts, grid=True):
		return self.geo.map_(pts, grid=grid)
	
	def geo_map_jac(self, pts, grid=True):
		return self.geo.map_der(pts, grid=grid)

	def geo_map_der2(self, pts, grid=True):
		# TODO
		self.geo.map_der2(pts, grid=grid)
		
		
#     breaks:   breaks along each direction in parametric space (repetitions are ignored)
#     qn:       quadrature nodes along each direction in parametric space
#     qw:       quadrature weights along each direction in parametric space
#     geometry: structure representing the geometrical mapping
#    'option', value: additional optional parameters, currently available options are:

class Msh2d(Msh):
	"2d mesh class"
	def __init__(self, breaks, qn, qw, geometry, boundary=True, der2=False):
		# call msh constructor

		if __debug__:
			print "DBG: new Msh2d object."
		super( Msh2d, self ).__init__(breaks, qn, qw, geometry, boundary=boundary, der2=der2)
		assert len(breaks) == len(qn)  == 2
		if qw is not None:
			assert len(qw) == 2

		self.breaks = [np.unique(breaks[0]), np.unique(breaks[1])] # paranoia
		self.nel_dir = np.array([ breaks[0].size-1, breaks[1].size-1]) 
		self.nel = (breaks[0].size-1) * (breaks[1].size-1)
		self.nelcol = self.nel_dir[1]

		# todo checks for qn!!
		qnu = qn[0] 
		qnv = qn[1]
		self.nqn_dir = np.array([ qn[0].shape[0] , qn[1].shape[0] ])
		# print 'nqndir'
		# print self.nqn_dir
		self.nqn = np.prod(self.nqn_dir)

		###?????
		# assert self.qn[0].shape[0] == self.nqn
		# assert self.qn[0].shape[1] == self.nel
		# assert self.qn[1].shape[0] == self.nqn
		# assert self.qn[1].shape[1] == self.nel

		if self.boundary:
			self.bnd = []
			ind = np.array([1,1,0,0])
			for iside in np.arange(4):

				b = Boundary() # empty object
				b.side_number = iside
				b.breaks = self.breaks[ind[iside]]
				b.nel = qn[ind[iside]].shape[1]
				b.nqn = qn[ind[iside]].shape[0]

				self.bnd.append( b )
		else:
			self.bnd = None

		#FIXME der2 (hessian)
	
	def evaluate_col(self, colnum):
		"""MSH_EVALUATE_COL: evaluate the parameterization in one column of the mesh."""
		msh_col = MeshCol() # TODO: actually, this should be also derived from the Msh class.
		msh_col.colnum = colnum 
		#FIXME nel_dir[1] >1
		msh_col.elem_list = colnum + self.nel_dir[0] * np.arange(self.nel_dir[1]).reshape(1, self.nel_dir[1])

		# print "elem_list = {} differs to geopdes".format(msh_col.elem_list)

		msh_col.nel_dir = self.nel_dir
		msh_col.nel = self.nelcol ##   this mesh has nel as many elements as there are columns
		msh_col.nqn = self.nqn
		msh_col.nqn_dir = self.nqn_dir

		
		#import pdb; pdb.set_trace()
		# hmmm assign local variable qnu the value of the super msh.
		qnu = self.qn[0][:,colnum].reshape(-1,1) ### qnu shape is not right (4,) instead of (4,1)
		# must pass ndim = 1 array to map, but flattening takes place later, as the following code depends on it.
		qnv = self.qn[1]
		if __debug__:
			print "DBG: msh evaluate col {} for qnu ({}), qnv({}).\t qnu = {}\t qnv={}".format(colnum, qnu.shape, qnv.shape, qnu, qnv)

		if self.quad_nodes is None:
			quad_nodes_u = np.reshape(qnu, (self.nqn_dir[0], 1, 1), order ='F')
			quad_nodes_u = np.tile (quad_nodes_u, [1, self.nqn_dir[1], self.nel_dir[1]])
			quad_nodes_u = np.reshape(quad_nodes_u, [-1, self.nel_dir[1]], order ='F')
			#pdb.set_trace()
			quad_nodes_v = np.reshape(qnv, (1, self.nqn_dir[1], self.nel_dir[1]), order ='F')
			quad_nodes_v = np.tile (quad_nodes_v, [self.nqn_dir[1], 1, 1])
			quad_nodes_v = np.reshape(quad_nodes_v, [-1, self.nel_dir[1]], order ='F')


			#import pdb; pdb.set_trace()
			msh_col.quad_nodes = np.empty(np.insert(quad_nodes_u.shape, 0, 2))
			#msh_col.quad_nodes = np.hstack((quad_nodes_u, quad_nodes_v)).T
			msh_col.quad_nodes[0] = quad_nodes_u
			msh_col.quad_nodes[1] = quad_nodes_u

			del quad_nodes_u, quad_nodes_v
		else:
			print "WARNING: entering untested code section"
			msh_col.quad_nodes = self.quad_nodes[:,:,msh_col.elem_list] # TODO: check.
			sys.exit(4)

		if self.qw is not None:
			if self.quad_weights is None:
				# create quad weights
				qwu = self.qw[0][:,colnum]
				# import pdb; pdb.set_trace()
				qwv = self.qw[1] # [:,colnum]

				quad_weights_u = np.reshape(qwu, (self.nqn_dir[0], 1), order ='F')
				quad_weights_u = np.tile (quad_weights_u, [1, self.nqn_dir[1], self.nel_dir[1]])
				quad_weights_u = np.reshape(quad_weights_u, [-1, self.nel_dir[1]], order ='F')
				# differently:
				quad_weights_v = np.reshape(qwv, (1, self.nqn_dir[1], self.nel_dir[1]), order ='F')
				quad_weights_v = np.tile (quad_weights_v, [self.nqn_dir[1], 1, 1])
				quad_weights_v = np.reshape(quad_weights_v, [-1, self.nel_dir[1]], order ='F')

				msh_col.quad_weights = quad_weights_u * quad_weights_v
				# print msh_col.quad_weights
				del quad_weights_u, quad_weights_v
			else:
				msh_col.quad_weights = self.quad_weights[:,msh_col.elem_list]

		# call maps, save in msh_col.geo_map
		# hmm.... think about this. Where evaluated map saved? should be one place in msh, right?

		# TODO: implement cached map!
		# if not cached...
		# eval_vect = np.vstack((qnu, qnv.T))
		eval_vect = [qnu.flatten(order='F'), qnv.flatten(order='F')]
		#if eval_vect.shape[0] != 2:
		## import pdb; pdb.set_trace() # h1 problem eval vect 
		F = self.geo_map(eval_vect, grid=True) 
		msh_col.geo_map = np.reshape(F , [2,  self.nqn, self.nel_dir[1]], order = 'F') # WARNING, this name will cause confusion
		# print F.shape

		# if not cached:
		jac = self.geo_map_jac(eval_vect)
		#print jac
		msh_col.geo_map_jac = np.reshape(jac, [2,2, self.nqn,self.nel_dir[1]], order = 'F')
		# this is a four dim array. I think the last (self.nel_dir[1] is for vectorial unknowns e.g. mechanics)
		# print jac.shape

		# jacobian matrix evaluated at quadrature nodes
		# TODO: caching

		# msh_side.jacdet = util.calc_norm(np.squeeze(msh_side.geo_map_jac[:,ind[iside],:,:]))[:,None]
		# import pdb; pdb.set_trace()
		msh_col.jacdet = np.abs(util.calc_jacdet(msh_col.geo_map_jac))
		msh_col.jacdet = np.reshape(msh_col.jacdet, (self.nqn, self.nel_dir[1]) ,order='F')

		# TODO msh.der2

		return msh_col

	def eval_boundary_side(self, iside):
		# MSH_EVAL_BOUNDARY_SIDE: evaluate the parameterization in one boundary side of the domain.

		assert iside < 4 # TODO
		ind = np.array([1, 1, 0, 0], dtype = np.int)
		ind2 = np.array([0, 0, 1, 1], dtype = np.int)

		msh_side = self.bnd[iside]
		msh_side.quad_weights = self.qw[ind[iside]] 

		msh_side.quad_nodes = np.zeros(np.insert(self.qn[ind[iside]].shape,0, 2))
		msh_side.quad_nodes[ind[iside],:,:] = self.qn[ind[iside]]

		# why?
		if iside == 1 or iside == 3:
			msh_side.quad_nodes[ind2[iside],:,:] = 1

		qn1 = msh_side.quad_nodes[0,:,:]
		qn2 = msh_side.quad_nodes[1,:,:]

		# pts = np.array([qn1, qn2])
		# print pts.shape # 2, 4, 1 
		# print np.squeeze(pts).shape # 2, 4

		# TODO: sort out dimensions
		# removing last dimension for map_ 
		pts = np.squeeze(np.array([qn1.flatten(order='F'), qn2.flatten(order='F')]))
		if __debug__:
			print 'calling geo_map with grid=False from msh/eval_boundary_side'

		#import pdb; pdb.set_trace()
		F = self.geo_map(pts, grid=False) # to not evaluate grid, but at scattered points

		jac = self.geo_map_jac(pts, grid=False)
		# print jac # totally different dimensions to reference octave. Perhaps calling wrong thing????
		# print jac.shape

		msh_side.geo_map = np.reshape(F, msh_side.quad_nodes.shape, order='F')
		msh_side.geo_map_jac = np.reshape(jac, [2, 2, msh_side.nqn, msh_side.nel], order='F')

		# import pdb; pdb.set_trace()
		# msh_side.jacdet = util.calc_norm(np.squeeze(msh_side.geo_map_jac[:,ind[iside],:,:]))[None,:]
		msh_side.jacdet = np.atleast_2d(util.calc_norm(np.squeeze(msh_side.geo_map_jac[:,ind[iside],:,:])))

		JinvT, jacdet = util.calc_JinvT(msh_side.geo_map_jac)
		JinvT = np.reshape(JinvT, [ 2, 2, msh_side.nqn, msh_side.nel], order='F')

		normal = np.zeros((2, msh_side.nqn, msh_side.nel))
		# +1 indexing?
		normal[ind2[iside]] = (-1)**(iside+1)
		normal = np.reshape(normal, [2, msh_side.nqn, 1, msh_side.nel], order ='F')
		normal = util.calc_prod(JinvT, normal)
		normal = np.reshape(normal, [2, msh_side.nqn, msh_side.nel], order = 'F')
		norms = np.tile( np.reshape(util.calc_norm(normal),[1, msh_side.nqn, msh_side.nel],order='F'), [2, 1, 1] )
		msh_side.normal = normal  / norms

		return msh_side




	

# should be derived from Msh
class MeshCol(object):
	"object containing quadrature rule in one column of the physical domain, which contains some fields" 



class Boundary(object):
	"boundary object"
	
	


  
if __name__ == '__main__':
	''' test
	'''
	print grule(5)
	

# % MSH_GAUSS_NODES: define nodes and weights for a multi-dimensional
# %                                 tensor-product Gauss quadrature rule.
# %
# %   rule = msh_gauss_nodes (nnodes)
# %
# % INPUT:
# %
# %     nnodes:   number of qadrature nodes along each direction
# %
# % OUTPUT:
# %
# %  rule:    cell array containing the nodes and weights of the rule  
# %           along each direction (rule{idir}(1,:) are the nodes and 
# %           rule{idir}(2,:) the weights)
# %
# % Copyright (C) 2010 Carlo de Falco, Rafael Vazquez
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
# 
# function rule = msh_gauss_nodes (nnodes)
#   
#   for idir = 1:numel (nnodes)
#     [rule{idir}(1,:), rule{idir}(2,:)] = grule (nnodes(idir));
#   end
# 
# end
