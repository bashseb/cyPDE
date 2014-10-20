from __future__ import division
import numpy as np

def calc_JinvT(v):
	assert(v.shape[0] == 2)
	# 2D
	if v.shape[0] == 2:
		det = v[0,0,:,:] * v[1,1,:,:] - v[1,0,:,:] * v[0,1,:,:]
		#print det.shape
		# print v[1,1,:,:].shape
		JinvT = np.empty(v.shape)

		JinvT[0,0,:,:] = v[1,1,:,:]/det
		JinvT[1,1,:,:] = v[0,0,:,:]/det
		JinvT[0,1,:,:] = -v[1,0,:,:]/det
		JinvT[1,0,:,:] = -v[0,1,:,:]/det

		det = np.squeeze(det)
	return JinvT, det

def calc_prod(a, b):
	aux_dim = (a.shape[0], a.shape[2], b.shape[2], b.shape[3])

	a = np.reshape(a, (a.shape[0], a.shape[1], a.shape[2], 1, a.shape[3]), order = 'F')
	b = np.reshape(b, np.insert(b.shape, 0, 1), order ='F')
	# print 'calc_prod'
	# row sum
	return np.reshape(np.sum(a*b,axis=1), newshape = aux_dim, order='F') # broadcasting

def calc_norm(v):
	''' geopdes_norm__.m '''

	assert v.shape[0] == 2 # 3d not implemented
	if v.shape[0] == 2:
		return np.sqrt(np.squeeze( v[0] * v[0] + v[1] * v[1] ))


	return -1

		
# cythonize, basis is geopdes_det__ function
def calc_jacdet(v):
	# print "jacdet"
	# print v

	assert v.shape[0] == 2 # 3d not implemented see geopdes_det__.m

	# for 2d:
	d = v[0,0,:,:] * v[1,1,:,:] - v[1,0,:,:] * v[0,1,:,:]
	

	return d

