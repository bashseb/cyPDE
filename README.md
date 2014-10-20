cyPDE
=====

Very basic Isogeometric Analysis implementation in python/cython. 

In order to learn about Isogeometric Analysis, Finite Element Analysis and NURBS, I rewrote a simplified version of geoPDEs (on sourceforge) using python as an exercise.
The goal was to get it running quickly and not thinking too much about designing it in a pythonic way (it's not). If I'd do it again, it'd be completely different. 

I'd like to express my gratitude to C. de Falco, A. Reali and R. Vazquezc for providing GeoPDEs.

### running:

e.g. very simple, coarse 2d Poisson problem, with just 8 degrees of freedom

     $ python2 -O tests/laplace_2d_square_f_homDirichlet.py
     Unknowns: 8 internal, 8 Dirichlet, 16 total.
     Internal stiffness matrix
     [[ 46.6667  -5.8333 -11.6667 -11.6667   0.       0.       0.       0.    ]
      [ -5.8333  46.6667 -11.6667 -11.6667   0.       0.       0.       0.    ]
      [-11.6667 -11.6667  93.3333 -11.6667 -11.6667 -11.6667   0.       0.    ]
      [-11.6667 -11.6667 -11.6667  93.3333 -11.6667 -11.6667   0.       0.    ]
      [  0.       0.     -11.6667 -11.6667  93.3333 -11.6667 -11.6667 -11.6667]
      [  0.       0.     -11.6667 -11.6667 -11.6667  93.3333 -11.6667 -11.6667]
      [  0.       0.       0.       0.     -11.6667 -11.6667  46.6667  -5.8333]
      [  0.       0.       0.       0.     -11.6667 -11.6667  -5.8333  46.6667]]
     Right hand side
     [  555.5556   555.5556  1111.1111  1111.1111  1111.1111  1111.1111   555.5556   555.5556]
     Solution vector (including Dirichlet d.o.f):
     [  0.     31.746  31.746   0.      0.     31.746  31.746   0.      0.     31.746  31.746   0.      0.     31.746  31.746   0.   ]
     Sparse solution equals dense solution: True
     *************** POST PROCESSING ******************
     l2 error : 2.646
     h1 error : 27.62
