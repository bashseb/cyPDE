#!/usr/bin/env python2
from __future__ import division
import numpy as np
#import sys
#sys.path.insert(0, '../nurbs_toolbox/')
#from nurbs import nurb


# cases for geo construction in geo_load_.:
# - file string
# - cell-array of function handles
# 4x4 matrix representing an affine trafo (?)
# a structure representing a nurbs surf or volume (I'm starting with THIS)
	
class Geo(object):
	"Geometry base class"
	geoCnt = 0
	def __init__(self, inp):
		Geo.geoCnt +=1
		# TODO validate nurbojb
		# if <filename, handles ..>
		# if it's not a nurbs, it will create one itself.
		# here I assume it's a nurb object
		self.nurb = inp

class Geo2d(Geo):
	"2d geometry class"
	if __debug__:
		print "DBG: new geo class 2d"

	def __init__(self, inp):
		super( Geo2d, self).__init__(inp)


	def map_(self, pts, grid=True):
		if __debug__:
			print 'DBG: called map_ with pts = {}'.format(pts)
			#import pprint; pprint.pprint(vars(self.nurb)) # self nurbs or refined nurbs?

		if grid: # for Dirichlet
			assert isinstance(pts, list)
			assert len(pts) < 4
		 	#import pdb; pdb.set_trace()
		else:
			assert not isinstance(pts, list)

		F = self.nurb.nrbeval( pts, grid=grid)
		return np.array([F[0].flatten(order='F'), F[1].flatten(order='F')]) 


	def map_der(self, pts, grid=True):
		if __debug__:
			print 'DBG: called map_der with pts = {}'.format(pts)

		#import pdb; pdb.set_trace()
		deriv = self.nurb.nrbderiv(deriv=1) # returns a list of two nurb surfaces (first is derived wrt u, second wrt v)
		evalpts, jac = self.nurb.nrbdeval(deriv, pts, grid=grid)
		# print 'jac:'
		

		if grid: #  iscell section
			npts = np.prod(pts[0].size * pts[1].size) # FIXME 1d, 3d?
			map_jac = np.zeros((2,2,npts))
			# print 'jacccc'
			# next lines will not work for Dirichlet b.c....
			# WARNING: discrepancy to reference in shape!!
			map_jac[0:2,0,:] = np.reshape(jac[0][0:2,:,:],[2,npts], order = 'F')
			map_jac[0:2,1,:] = np.reshape(jac[1][0:2,:,:],[2,npts], order = 'F')
		else:
			map_jac = np.zeros((2, 2, pts.shape[1]))
			# print pts.shape[1]
			# print jac[0][0:2,:].shape
			# print np.append([2,1],pts.shape[1])
			# print np.reshape(jac[0][0:2,:], (2,1,4), order ='F')
			# print map_jac[0:2,0,:].shape
			# testing:
			map_jac[0:2,0] = jac[0][0:2]
			map_jac[0:2,1] = jac[1][0:2]
			# map_jac[0:2,0,:] = np.reshape(jac[0][0:2,:], (2,1,4), order ='F')
			# map_jac[0:2,1,:] = np.reshape(jac[1][0:2,:], np.append([2,1],pts.shape[1]), order ='F')
			#print map_jac.shape

			#sys.exit(43)
		


		return map_jac


	def map_der2(self, pts, grid=True):
		return 0.

# TODO
# 	def calcMap(self, pts, ders):
# 		if ders == 0:
# 			return map(self, pts)
# 		elif ders == 1:
# 			return map_der(self, pts)
# 		elif ders == 2:
# 			return map_der2(self, pts)
# 		else:
# 			print "ERROR: cannot request {}-th derivative".format(ders)
# 			sys.exit(1)


	



  
if __name__ == '__main__':
	''' test
	'''
	
