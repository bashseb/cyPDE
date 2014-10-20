#!/usr/bin/env python2 
from __future__ import absolute_import  # default since 2.7, exists since 2.5
from __future__ import division
import numpy as np
#import sys
#sys.path.insert(0,'basis/')
# print sys.path
# import module in same directory (bspline.so symlink )

# for test
import sys
sys.path.append('.')
#
from .bspline import bspeval, bspderiv



# function nurbs = nrbmak(coefs,knots)
#
# NRBMAK: Construct the NURBS structure given the control points
#            and the knots.
# 
# Calling Sequence:
# 
#   nurbs   = nrbmak(cntrl,knots);
# 
# INPUT:
# 
#   cntrl       : Control points, these can be either Cartesian or
# 		homogeneous coordinates.
# 
# 		For a curve the control points are represented by a
# 		matrix of size (dim,nu), for a surface a multidimensional
# 		array of size (dim,nu,nv), for a volume a multidimensional array
# 		of size (dim,nu,nv,nw). Where nu is number of points along
# 		the parametric U direction, nv the number of points along
# 		the V direction and nw the number of points along the W direction. 
# 		dim is the dimension. Valid options
# 		are
# 		2 .... (x,y)        2D Cartesian coordinates
# 		3 .... (x,y,z)      3D Cartesian coordinates
# 		4 .... (wx,wy,wz,w) 4D homogeneous coordinates
# 
#   knots	: Non-decreasing knot sequence spanning the interval
#               [0.0,1.0]. It's assumed that the geometric entities
#               are clamped to the start and end control points by knot
#               multiplicities equal to the spline order (open knot vector).
#               For curve knots form a vector and for surfaces (volumes)
#               the knots are stored by two (three) vectors for U and V (and W)
#               in a cell structure {uknots vknots} ({uknots vknots wknots}).
#               
# OUTPUT:
# 
#   nurbs 	: Data structure for representing a NURBS entity
# 
# NURBS Structure:
# 
#   Both curves and surfaces are represented by a structure that is
#   compatible with the Spline Toolbox from Mathworks
# 
# 	nurbs.form   .... Type name 'B-NURBS'
# 	nurbs.dim    .... Dimension of the control points
# 	nurbs.number .... Number of Control points
#       nurbs.coefs  .... Control Points
#       nurbs.order  .... Order of the spline
#       nurbs.knots  .... Knot sequence
# 
#   Note: the control points are always converted and stored within the
#   NURBS structure as 4D homogeneous coordinates. A curve is always stored 
#   along the U direction, and the vknots element is an empty matrix. For
#   a surface the spline order is a vector [du,dv] containing the order
#   along the U and V directions respectively. For a volume the order is
#   a vector [du dv dw]. Recall that order = degree + 1.
# 
# Description:
# 
#   This function is used as a convenient means of constructing the NURBS
#   data structure. Many of the other functions in the toolbox rely on the 
#   NURBS structure been correctly defined as shown above. The nrbmak not
#   only constructs the proper structure, but also checks for consistency.
#   The user is still free to build his own structure, in fact a few
#   functions in the toolbox do this for convenience.
# 
# Examples:
# 
#   Construct a 2D line from (0.0,0.0) to (1.5,3.0).
#   For a straight line a spline of order 2 is required.
#   Note that the knot sequence has a multiplicity of 2 at the
#   start (0.0,0.0) and end (1.0 1.0) in order to clamp the ends.
# 
#   line = nrbmak([0.0 1.5; 0.0 3.0],[0.0 0.0 1.0 1.0]);
#   nrbplot(line, 2);
# 
#   Construct a surface in the x-y plane i.e
#     
#     ^  (0.0,1.0) ------------ (1.0,1.0)
#     |      |                      |
#     | V    |                      |
#     |      |      Surface         |
#     |      |                      |
#     |      |                      |
#     |  (0.0,0.0) ------------ (1.0,0.0)
#     |
#     |------------------------------------>
#                                       U 
#
#   coefs = cat(3,[0 0; 0 1],[1 1; 0 1]);
#   knots = {[0 0 1 1]  [0 0 1 1]}
#   plane = nrbmak(coefs,knots);
#   nrbplot(plane, [2 2]);
#
#    Copyright (C) 2000 Mark Spink, 2010 Rafael Vazquez
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

class Nurb:
	'Nurb'
	nrbCnt = 0
	def __init__(self, coefs, knots):
		Nurb.nrbCnt += 1
		self.index = Nurb.nrbCnt

		self.form = 'B-NURBS' #??
		self.np = coefs.shape #?? (three elems 4, 2, 2)
		self.dim = self.np[0] #?? 4?
		assert self.dim == 4
		self.ndim = coefs.ndim-1 # 
		assert len(knots) == self.ndim
		self.weights = coefs[3]
		assert np.all(coefs[3] >=0 ) # for safety, negative weights are uncommon
		self.cs = coefs[0:3]
		self.coefs = coefs

		self.knots = []


		self.nrbmak(knots)

	def nrbmak(self, knots):

		if self.ndim == 1:  # Crv
			self.nrbmakCrv(knots)
		elif self.ndim == 2:  # Surf
			self.nrbmakSurf(knots)
		elif self.ndim == 3:  # Vol
			self.nrbmakVol(knots)

	def nrbmakCrv(self, knots):
		self.number = self.np[1]
		uknots = np.sort(knots[0])
		self.order = np.array([knots[0].size - self.number])
		self.knots = [uknots] # FIXME should this be a list or not?

	def nrbmakSurf(self, knots):
		if __debug__:
			print "DBG: new Surface Nurb object"
		self.number = self.np[1:3] # 2, 2
		uknots = np.sort(knots[0])
		self.order = np.array([knots[0].size-self.number[0], knots[1].size-self.number[1]])
		vknots = np.sort(knots[1])
		self.knots = [uknots,vknots]
		# TODO normalize if U>1 or u<1

	def nrbmakVol(self, knots):
		# FIXME: untested
		self.number = self.np[1:4]
		uknots = np.sort(knots[0])
		vknots = np.sort(knots[1])
		wknots = np.sort(knots[2])
		self.order = np.array([knots[0].size-self.number[0], knots[1].size-self.number[1]], knots[2].size-self.number[2])
		self.knots = [uknots,vknots,wknots]
	
	def kntrefine(self, nsub, degree, regularity):
		assert nsub.size == degree.size
		assert nsub.size == regularity.size
		if __debug__:
			print "DBG: Knot refinement with {} data".format(nsub)
		# full copy
		aux_knots = self.knots[:]

		if self.ndim > 1:
			assert len(self.knots) == nsub.size 
		else:
			assert nsub.size == 1

		# TODO: currently returns only (rknots and zeta, not newknots)

		rknots = []
		zeta = []
		for idim in np.arange(nsub.size):
			min_mult = degree[idim] - regularity[idim]
			z = np.unique(aux_knots[idim])
			nz = z.size
			# deg holds inferred current degree (=order-1)
			deg = np.sum(aux_knots[idim] == z[0]) - 1
			# add degree[idim]+1 left zeros to new knot vector
			rknots.append(z[np.zeros(degree[idim]+1, dtype=np.uint)])  
			#import pdb;pdb.set_trace()

			# loop over knot size
			for ik in np.arange(1,nz):
				insk = np.linspace (z[ik-1], z[ik], num=nsub[idim]+2)
				# all except knot start/end, repeated min_mult times.
				# insk  = np.tile(insk[1:-1], min_mult)
				insk  = np.tile(insk[1:-1], (min_mult, 1)).flatten(order='F')
				old_mult = np.sum(aux_knots[idim] == z[ik])
				mult = np.max((min_mult, degree[idim] - deg + old_mult))
				rknots[idim] =  np.concatenate((rknots[idim], insk, z[ik*np.ones(mult, dtype=np.int)]))
			
			zeta.append(np.unique(rknots[idim]))

		# overwrite knot vector
		# self.knots = rknots
		# return only zeta
		# FIXME: actually returning zeta doesn't make sense. 
		# One can obtain it any time by using np.unique(<nobj>.knots)
		return rknots, zeta
		
	def plot(self, subd=10):

		if self.ndim == 1:
			self.plotCrv(subd=subd)
		if self.ndim == 2:
			self.plotSurf(subd=subd)
	
	# this is a curve embedded in R3
	# one could check for zero z or zero y components to plot in a plane.
	def plotCrv(self, subd=10):
		import matplotlib as mpl
		import matplotlib.pyplot as plt
		# if ndim == ...
		from mpl_toolkits.mplot3d import axes3d
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		if np.all(self.weights == 1.):
			stri = 'B-spline'
		else:
			stri = 'Nurbs'
		plt.title("{} Curve. n={}, p={}".format(stri, self.number, self.order[0]-1))
		plt.xlabel('x')
		plt.ylabel('y')
		pts = self.nrbeval( np.linspace(self.knots[0][0], self.knots[0][-1], num=subd))
		plt.plot(pts[0],pts[1],pts[2], label = stri+' curve')
		plt.legend(fontsize='x-small',bbox_to_anchor=(0.91, 1), loc=2, borderaxespad=-1.)
		plt.savefig("bspline-crv-R3.png")
		# plt.show() # stops here
		plt.close()

	
	def plotSurf(self, subd=10):
		if __debug__:
			print "DBG: plotting Surface with {} subdivisions.".format(subd)
		from mpl_toolkits.mplot3d import axes3d
		import matplotlib as mpl
		import matplotlib.pyplot as plt
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		if np.all(self.weights == 1.):
			stri = 'B-spline'
		else:
			stri = 'Nurbs'
		plt.title(stri+" Surface. n={}, p={}, m={}, q={}".format(self.number[0], self.order[0]-1, self.number[1], self.order[1]-1))
		plt.xlabel('x')
		plt.ylabel('y')
		pts = self.nrbeval( [np.linspace(self.knots[0][0], self.knots[0][-1], num=subd),np.linspace(self.knots[1][0], self.knots[1][-1], num=subd)])

		ax.plot_wireframe(pts[0,:,:], pts[1,:,:], pts[2,:,:], rstride=1, cstride=1, label=stri+' surface')

		#plt.plot([0,3],[0,1], label = 'test')
		plt.legend(fontsize='x-small',bbox_to_anchor=(0.91, 1), loc=2, borderaxespad=-1.)
		plt.savefig("bspline-surf-R3.png")
		# plt.show() # stops here
		plt.close()

	# cythonize?
	def nrbeval(self,ut,vt=None,wt=None, homog=False, grid=True):

		#import pdb;pdb.set_trace()
		# ut is np 1D or 2D array
		# normally returns xyz + weights
		# optionally returns homogeneous coordinates (homog = True, e.g. for nrbdeval)
		stri = 'Cartesian'
		if homog:
			stri= 'Homogeneous'

		degree = self.order-1
		assert self.ndim <= 2 # else not implemented
		if grid:
			assert isinstance(ut, list)
			assert len(ut) == self.ndim
			assert vt == None
			if __debug__:
				#print 'DBG: {} Nurb evaluation at ut(flat) = {}, vt(flat) = {}'.format(stri, ut.flatten(),vt.flatten())
				print 'nurbs-DBG: {} Grid surf nrbeval over {},{}. upts = {}\t vpts={}'.format(stri, ut[0].size, ut[1].size, ut[0], ut[1])
		else:
			assert not isinstance(ut, list)
			assert ut.ndim == 2
			# ensure vector is correctly orientated:
			assert ut.shape[0] == 2
			assert vt == None
			if __debug__:
				#print 'DBG: {} Nurb evaluation at ut(flat) = {}, vt(flat) = {}'.format(stri, ut.flatten(),vt.flatten())
				# print 'DBG: {} Scatter Surf Nurb evaluation at ut(flat) = {}, ({} pts)'.format(stri, ut.flatten(), ut.shape[0])
				print 'nurbs-DBG: {} Scatter surf nrbeval {}. \t upts = {}, vpts= {}'.format(stri, ut.shape[1], ut[0], ut[1])

		if self.ndim == 1:
			val =  bspeval(p, self.coefs, self.knots[0], ut)
		elif self.ndim == 2:
			num1 = self.number[0]
			num2 = self.number[1]
			if  grid:
				#vt = ut[1,:]
				#ut = ut[0,:]
				assert ut[0].ndim == 1
				assert ut[1].ndim == 1
				# iterate over [u,v] grid
				nt1 = ut[0].size
				nt2 = ut[1].size

				# v direction
				val =  np.reshape(self.coefs, (4*num1, num2),order='F')
				# output same as nurbs TB
				val = bspeval(degree[1], val, self.knots[1], ut[1]) # ut[1] and ut[0] must have ndim=1
				# val shape is [4 * num1, nt2]
				val = np.reshape(val,(4, num1, nt2),order='F')

				val = np.transpose(val, (0,2,1))
				val = np.reshape(val, (4*nt2, num1), order='F')
				val = bspeval(degree[0], val, self.knots[0], ut[0])
				val = np.reshape(val,(4,nt2,nt1), order='F');
				val = np.transpose(val, (0,2,1))

				if not homog:
					# for Cartesian coordinates, division by weights is necessary
					w = val[3,:,:]
					val[0:3,:,:] = val[0:3,:,:]/np.tile(w,[3, 1, 1]) # element-wise divide

			elif grid == False:

				# ut[0,:] u direction
				# ut[1,:] v direction
				
				# ut is ndarray with dim2
				nt = ut.shape[1]

				# v direction
				val = np.reshape(self.coefs, (4*num1, num2),order='F')
				val = bspeval(degree[1], val, self.knots[1], ut[1,:])
				val = np.reshape(val,(4, num1,  nt),order='F')

				# u direction
				pnts = np.zeros((4, nt))
				for v in np.arange(nt):
					#print val[:,:,v] # ok
					coefs = np.reshape( val[:,:,v], (4, num1))
					pnts[:,v] = bspeval(degree[0], coefs, self.knots[0], np.array([ut[0,v]])).flatten()
				#print pnts

				val = pnts
				if not homog:
					w = val[3,:]
					# p = pnts[0:3,:]
					val[0:3,:] = val[0:3,:]/np.tile(w,[3, 1]) # element-wise divide

			else:
				print "FATAL ERROR: grid error" 


			return val

	# cythonize?
	def nrbeval_orig(self,ut,vt=None,wt=None, homog=False, grid=True):

		# ut is np 1D or 2D array
		# normally returns xyz + weights
		# optionally returns homogeneous coordinates (homog = True, e.g. for nrbdeval)
		stri = 'Cartesian'
		if homog:
			stri= 'Homogeneous'

		if __debug__:
			print 'DBG: {} Nurb evaluation at ut(flat) = {}, vt(flat) = {}'.format(stri, ut.flatten(),vt.flatten())

		p = self.order-1

		assert self.ndim <= 2 # else not implemented

		if self.ndim == 1:
			val =  bspeval(p, self.coefs, self.knots[0], ut)
		elif self.ndim == 2:
			n = self.number[0]
			num1 = n
			m = self.number[1]
			num2 = m
			# hack.. see if this works.. required for 'map_'
			# well, this is definitely not good. The next line depend on receiving a row vector....
			if vt is None and ut.ndim == 2 and grid:
			# 	if __debug__:
			# 		print "DBG: nrbeval: ut given shape = {}".format(ut.shape)
				assert ut.shape[0] == 2 # if this fails, column vector was supplied and not row vector. to be changed in the future.
				vt = ut[1,:]
				ut = ut[0,:]
			if vt is not None and grid:
				# iterate over [u,v] grid
				nt1 = ut.size
				nt2 = vt.size

				# v direction
				# val = np.array
				# u direction
				# print vt
				# print self.cs.shape
				# print self.weights.shape
				#val =  np.reshape(self.coefs, (4*self.coefs.shape[1], self.coefs.shape[2]))
				val =  np.reshape(self.coefs, (4*self.coefs.shape[1], self.coefs.shape[2]),order='F')
				# output same as nurbs TB
				val = bspeval(p[1], val, self.knots[1], vt)
				# valid

				val = np.reshape(val,(4, self.coefs.shape[1], vt.shape[0]),order='F')
				val = np.transpose(val, (0,2,1))
				val = np.reshape(val, (4*vt.shape[0], n), order='F')

				# print p[0]
				# print self.knots[0]
				# print ut

				val = bspeval(p[0], val, self.knots[0], ut)
				val = np.reshape(val,(4,nt1,nt2), order='F');
				val = np.transpose(val, (0,2,1))

				if not homog:
					# for Cartesian coordinates, division by weights is necessary
					w = val[3,:,:]
					val[0:3,:,:] = val[0:3,:,:]/np.tile(w,[3, 1, 1]) # element-wise divide

			elif grid == False:

				print "Evaluation at scattered points"
				# ut[0,:] u direction
				# ut[1,:] v direction
				
				# what is ut[1,:] used for?????????

				assert ut.ndim == 2
				nt = ut.shape[1]

				# v direction
				val = np.reshape(self.coefs, (4*num1, num2),order='F')
				val = bspeval(p[1], val, self.knots[1],ut[1,:])
				val = np.reshape(val,(4, num1,  nt),order='F')

				# u direction
				pnts = np.zeros((4, nt))
				print nt
				for v in np.arange(nt):
					print val[:,:,v] # ok
					coefs = np.reshape( val[:,:,v], (4, num1))
					# print ut[0,v]
					# print bspeval(p[0], coefs, self.knots[0], np.array([ut[0,v]])) # arrrrg, single value passed to bspeval not working
					# arrrr
					pnts[:,v] = bspeval(p[0], coefs, self.knots[0], np.array([ut[0,v]])).flatten()
				#print pnts

				val = pnts
				if not homog:
					w = val[3,:]
					# p = pnts[0:3,:]
					val[0:3,:] = val[0:3,:]/np.tile(w,[3, 1]) # element-wise divide

			else:
				print "FATAL ERROR: grid error" 


			return val


	
	def nrbderiv(self, deriv=1):
		''' take the 'deriv'-th derivative of a bspline object'''
		assert deriv==1
		assert self.ndim <= 2 # dim 3 not implemented yet
		if __debug__:
			print "DBG: called nrbderiv."

		if self.ndim == 1: # curve
			dcoefs, dknots = bspderiv(self.order - 1, self.coefs, self.knots) # TESTME
			dnurb = nrb(dcoefs, dknots)

		elif self.ndim == 2:
			dnurb = []
			num1 = self.number[0]
			num2 = self.number[1]
			dknots = list(self.knots)
			# u direction
			# print 'coefs'
			# print self.coefs
			# print self.coefs.shape
			dcoefs = np.transpose(self.coefs, (0, 2 ,1))
			# print 'transposed'
			# print dcoefs
			# print dcoefs.shape
			dcoefs = np.reshape(dcoefs, (4*num2, num1), order='F')
			# print 'reshaped'
			# print dcoefs
			dcoefs, dknots[0] = bspderiv(self.order[0] - 1, dcoefs, self.knots[0])
			# print 'result u'
			# print dcoefs
			# print dknots[0]
			dcoefs = np.transpose( np.reshape(dcoefs, (4, num2, dcoefs.shape[1]),order='F'), (0, 2, 1))
			# print 'tshaped'
			# print dcoefs

			dnurb.append(Nurb(dcoefs, dknots))

			# v direction
			dknots = list(self.knots)
			dcoefs = np.reshape( self.coefs, (4*num1, num2), order='F')
			dcoefs, dknots[1] = bspderiv(self.order[1] - 1, dcoefs, self.knots[1])
			# print 'v result'
			# print dcoefs
			# print dknots[1]
			dcoefs = np.reshape(dcoefs, (4, num1, dcoefs.shape[1]), order ='F')

			dnurb.append( Nurb(dcoefs, dknots))


		return dnurb

		
	# cythonize
	def nrbdeval(self,dnurbs, ut, deriv=1, grid=True):
		# returns [pnt, jacobian] if first deriv or
		# 		  [pnt, jac, hessian] if second deriv]

		# TODO: only surface/curve and first derivative
		assert self.ndim <=2
		assert deriv == 1 # only first order implemented yet

		if __debug__:
			print "DBG: called nrbdeval with ut = {}".format(ut)

		res = self.nrbeval(ut, homog=True, grid=grid) # nrbeval(nurbs, tt)

		# TODO: too many indices?

		if self.ndim == 1:
			print 'dim1'
			#temp = 
			print cw
			print "ERROR: not implememnted"
			return -1

		elif self.ndim == 2:
			# print res
			# -old- cp = res[0:3,:,:] # coefficients
			# -old- cw = res[3,:,:] # weights
			cp = res[0:3]
			cw = res[3]
			# print "dim2"
			# print cw.shape
			#tmp: 
			#cw = np.reshape(np.random.random(cw.shape), (1,cw.shape[0],cw.shape[1]))
			# print cw

			#### TODO: why!@!!?!?!?!?!
			# cw = np.reshape(cw, (1,cw.shape[0],cw.shape[1]))
			# print 'cw - upgraded'
			#print cw[0,:,:]
			idx = np.zeros(3,dtype=np.uint)
			temp =  np.array([cw, cw, cw])

			# print cw.shape
			# print temp.shape
			# print cp.shape
			pnt = cp/temp
			# print 'pnt'
			# print pnt
			# print '---- u eval ---'
			# print dnurbs[0].coefs

			# FIXME: use object not list
			jac = []
			res2 = dnurbs[0].nrbeval(ut, homog=True, grid=grid) # nrbeval(dnurb{1}, tt)
 			cup = res2[0:3]
 			cuw = res2[3]

			#cuw = np.reshape(cuw, (1, cuw.shape[0], cuw.shape[1]))
			#tempu = cuw[idx]
			tempu = np.array([cuw, cuw, cuw])
			# print 'tmpu'
			# print tempu

			jac.append( (cup - tempu*pnt)/temp )
			# print 'jac0'
			# print jac[0]
 
 			res3 = dnurbs[1].nrbeval(ut, homog=True, grid=grid) # nrbeval(dnurb{2}, tt)
 			cvp = res3[0:3]
 			cvw = res3[3]
			#cvw = np.reshape(cvw, (1, cvw.shape[0], cvw.shape[1]))
			#tempv = cvw[idx]
			tempv = np.array([cvw, cvw, cvw])
			jac.append( (cvp - tempv*pnt)/temp )
			# print 'jac1'
			# print jac[1]

			return pnt, jac

		return -1, -1



if __name__ == '__main__':
	''' test '''
	# TODO testing doesnt work
	# laplace example, square
	uknots = np.array([0,0,1,1.])
	knots = [uknots,uknots]
	nu=nv=2
	scoefs = np.zeros([4,nu,nv])
	scoefs[0,1,1] = 1.
	scoefs[1,1,1] = 1.
	scoefs[3] = np.ones([nu,nv])
	scoefs[0,1,0] = 1.
	scoefs[1,0,1] = 1.
	# print scoefs
	
	B = Nurb(scoefs, knots)
