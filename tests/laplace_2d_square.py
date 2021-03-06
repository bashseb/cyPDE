#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
import numpy as np
import sys
# Normally, one would include the following dir in the PYTHONPATH environmental variable
sys.path.append('/home/eiser/phd/src/py/cyPDEs-trunk') # load my own modules (give precedence to general modules, e.g. 'operator' from numpy)
from nurbs import nrb

# set line limit
#np.set_printoptions(precision=6,suppress=True, linewidth=300,threshold=100000)
np.set_printoptions(precision=4,suppress=False, linewidth=270,threshold=10000)


vknots = np.array([0,0,0,1,1,1.])
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

B = nrb.Nurb(scoefs, knots)
B.plot(13)

from geo import geo
# aquire maps!
myGeo = geo.Geo2d(B)

#nsub = np.array([9, 9]) 		# no of subdivisions of original knot vector
nsub = np.array([7, 4]) 		# no of subdivisions of original knot vector
# currently only nsub[0] == nsub[1] works
regularity = np.array([2, 2]) 	# spline regularity
degree = np.array([3, 4])		# spline degree
nquad = np.array([4,4]) 			# gauss quadrature pts


# refine knots by inserting `nsub` knots in each original knot interval. degree of inserted knots is `degree -1`
# TODO: regularity???!?!?!? max global regularity
# returns refined knot vector without reps
newknots, zeta = B.kntrefine(nsub-1, degree, regularity)

# import pdb; pdb.set_trace()


from msh import msh
# gauss rule
rule = msh.msh_gauss_nodes(nquad)

qn, qw = msh.set_quad_nodes(zeta, rule)

# create msh object
myMesh = msh.Msh2d(zeta, qn, qw, myGeo)

from space import spBSpline

# create main space
mySpace = spBSpline.BspSpace2d(newknots, degree, myMesh)

from operators.op_gradu_gradv_tp import op_gradu_gradv_tp
lam = 1 # conductivity
c_diff = lambda x, y: lam * np.ones(x.size)
# calculate stiffness matrix using tensor product
stiff_mat = op_gradu_gradv_tp(mySpace, mySpace, myMesh, c_diff)

# rhs, q .. power
q = 1e4
f = lambda x, y:  q * np.ones(x.size)

# initial rhs
rhs = np.zeros(stiff_mat.shape[0]) 

# TODO: no vonNeuman B.C. atm

# dirichlet b.c.
# all four sides have dirichlet b.c. given by
drchlet_sides=np.array([0,1,2,3])
h = lambda x, y, ind: np.exp(x) * np.sin(y) # ind is for id of boundary
#h = lambda x, y, ind: np.exp(x) # * np.sin(y) # ind is for id of boundary
#h = lambda x, y, ind:  np.sin(y*np.pi) # ind is for id of boundary

# find values for Dirichlet nodes
u_drchlt, drchlt_dof = mySpace.sp_drchlt_l2_proj(myMesh, h, drchlet_sides)

u = np.zeros(mySpace.ndof)
u[drchlt_dof] = u_drchlt

int_dof = np.setdiff1d(np.arange(mySpace.ndof), drchlt_dof)

# rhs(int_dofs) = rhs(int_dofs) - stiff_mat(int_dofs, drchlt_dofs)*u_drchlt;
# WARNING: dot product looks differently in reference. will probably fail for non-symmetric problems
rhs[int_dof] = rhs[int_dof] - np.dot(stiff_mat.todense()[drchlt_dof[None,:], int_dof[:,None]], u_drchlt)

from scipy.sparse.linalg import spsolve

#import pdb; pdb.set_trace()

print "Unknowns: {} internal, {} Dirichlet, {} total.".format(int_dof.size, drchlt_dof.size, mySpace.ndof)

###############    SPARSE  SOLVE   #################
u[int_dof] = spsolve(stiff_mat[int_dof[None,:], int_dof[:,None]], rhs[int_dof])

if mySpace.ndof < 40:
	print "Internal stiffness matrix"
	print stiff_mat[int_dof[None,:], int_dof[:,None]].todense()
	print "Right hand side"
	print rhs[int_dof]
	print "Solution vector (including Dirichlet d.o.f):"
	print u
else:
	print "Internal stiffness matrix (summary)"
	print "Sparse {0}x{1} matrix with {2} non-zero elements. Fill = {3:.3g}%".format(stiff_mat.shape[0], stiff_mat.shape[1], stiff_mat.size, stiff_mat.size/(stiff_mat.shape[0]*stiff_mat.shape[1])*100)
	print "Right hand side (summary)"
	print "Max: {0:.3g} min {1:.3g} size {2:g}".format(np.max(rhs[int_dof]), np.min(rhs[int_dof]), rhs[int_dof].size)
	print "Solution vector (summary):"
	print "Max: {0:.3g} min {1:.3g} size {2:g}".format(np.max(u), np.min(u), u.size)

if mySpace.ndof < 500:
	# test if sparse solution is correct
	from numpy.linalg import solve, norm
	u_ = u.copy()
	u_[int_dof] = solve(stiff_mat.todense()[int_dof[None,:],int_dof[None,:].T], rhs[int_dof])
	err = norm(u-u_)
	print "Sparse solution equals dense solution: {}".format(err < 1e-10)

### plotting

print "*************** POST PROCESSING ******************"

uex = lambda x, y: np.exp(x) * np.sin(y) # exact solution
graduex = lambda x, y: np.vstack([np.reshape(np.exp(x)*np.sin(y), (1, x.shape[0]), order='F'), np.reshape(np.exp(x)*np.cos(y), (1, x.shape[0]), order='F')]) 
graduex = lambda x, y: np.vstack([
			np.reshape(np.exp(x)*np.sin(y), np.insert(x.shape, 0, 1), order='F'), 
			np.reshape(np.exp(x)*np.cos(y), np.insert(x.shape, 0, 1), order='F')
					]) 

errh1, errl2 = mySpace.h1_error(myMesh, u, uex, graduex)
print "l2 error : {0:1.4g}".format(errl2)
print "h1 error : {0:1.4g}".format(errh1)

# plot_pts = [np.linspace(0,1,20)[:,None], np.linspace(0,1,20)[:,None]]
plot_pts = [np.linspace(0,1,20), np.linspace(0,1,20)]
#print plot_pts

#import pdb; pdb.set_trace()
eu, F = mySpace.sp_eval(u, myGeo, plot_pts) 
# at least eval_vect is wrong...

x = np.squeeze(F[0])
y = np.squeeze(F[1])

# works
#np.vstack([np.reshape( np.exp(x[0])*np.sin(x[1]), np.insert(x[0].shape,0, 1), order='C'),np.reshape( np.exp(x[0])*np.cos(x[1]), np.insert(x[0].shape,0, 1), order='C')])


from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# import pdb; pdb.set_trace()

ax.plot_surface(x, y, eu) #, rstride=1, cstride=1, )
#ax.plot_wireframe(x, y, eu) #, rstride=1, cstride=1, )
		        #linewidth=0, antialiased=False)
plt.savefig("laplace_2d.png")
plt.show()
