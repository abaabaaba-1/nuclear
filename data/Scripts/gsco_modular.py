"""
gsco_modular.py

Optimizes a 96x100 wireframe using the GSCO method to develop a solution 
with modular coils as shown in Section 4.2 of the paper. The wireframe is
initialized to have 6 planar coils per half-period whose shapes are 
subsequently modified by the GSCo procedure.

Outputs 2d plot of the current distribution in one half-period, as well as a
VTK model of the wireframe. If desired, 3d plots of the initialized wireframe
and the optimzed solution wireframe may also be rendered (requires the mayavi 
package).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as pl
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.solve import optimize_wireframe

# Set to True generate a 3d rendering with the mayavi package
make_mayavi_plots = False

# Number of wireframe segments per half period in the toroidal dimension
wf_nPhi = 96

# Number of wireframe segments in the poloidal dimension
wf_nTheta = 100

# Maximum number of GSCO iterations
max_iter = 20000

# How often print progress
print_interval = 100

# Resolution of test points on plasma boundary (poloidal and toroidal)
plas_n = 32

# Number of modular coils in the solution per half period
n_mod_coils_hp = 6

# Average magnetic field on axis, in Teslas, to be produced by the wireframe.
# This will be used for initializing the modular coils. The radius
# of the magnetic axis will be estimated from the plasma boundary geometry.
field_on_axis = 1.0

# Weighting factor for the sparsity objective
lambda_S = 10**-6

# File for the desired boundary magnetic surface:
filename_equil = '/home/dataset-assist-0/MOLLM-main/data/Supporting_Files/input.LandremanPaul2021_QA'

# File specifying the geometry of the wireframe surface (made with BNORM)
filename_wf_surf = '/home/dataset-assist-0/MOLLM-main/data/Supporting_Files/nescin.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Load the geometry of the target plasma boundary
plas_nPhi = plas_n
plas_nTheta = plas_n
surf_plas = SurfaceRZFourier.from_vmec_input(filename_equil, 
                nphi=plas_nPhi, ntheta=plas_nTheta, range='half period')

# Construct the wireframe on a toroidal surface
surf_wf = SurfaceRZFourier.from_nescoil_input(filename_wf_surf, 'current')
wf = ToroidalWireframe(surf_wf, wf_nPhi, wf_nTheta)

# Calculate the required net poloidal current
mu0 = 4.0 * np.pi * 1e-7
pol_cur = -2.0*np.pi*surf_plas.get_rc(0,0)*field_on_axis/mu0

# Initialize the wireframe with a set of planar TF coils
coil_current = pol_cur/(2*wf.nfp*n_mod_coils_hp)
wf.add_tfcoil_currents(n_mod_coils_hp, coil_current)

# Set constraint for net poloidal current (note: the constraint is not strictly 
# necessary for GSCO to work properly, but it can be used as a consistency 
# check for the solution)
wf.set_poloidal_current(pol_cur)   

# Generate a 3D plot of the initialized wireframe and plasma if desired
if make_mayavi_plots:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050,800), bgcolor=(1,1,1))
    wf.make_plot_3d(to_show='all', tube_radius=0.006)
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil, 
        nphi=plas_nPhi, ntheta=plas_nTheta, range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True, 
        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=5.5, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'gsco_modular_init_plot3d.png')

# Set the optimization parameters
opt_params = {'lambda_S': lambda_S, 
              'max_iter': max_iter,
              'print_interval': print_interval,
              'no_crossing': True,
              'default_current': np.abs(coil_current),
              'max_current': 1.1 * np.abs(coil_current)
             }

# Run the GSCO optimization
t0 = time.time()
res = optimize_wireframe(wf, 'gsco', opt_params, surf_plas=surf_plas, 
                         verbose=False)
t1 = time.time()
deltaT = t1 - t0

print('')
print('Post-processing')
print('---------------')
print('  opt time [s]   %12.3f' % (deltaT))

# Verify that the solution satisfies all constraints
assert wf.check_constraints()

# Post-processing
res['wframe_field'].set_points(surf_plas.gamma().reshape((-1,3)))
Bfield = res['wframe_field'].B().reshape((plas_nPhi, plas_nTheta, 3))
Bnormal = np.sum(Bfield * surf_plas.unitnormal(), axis=2)
modB = np.sqrt(np.sum(Bfield**2, axis=2))
relBnorm = Bnormal/modB
area = np.sqrt(np.sum(surf_plas.normal()**2, axis=2))/float(modB.size)
meanRelBn = np.sum(np.abs(relBnorm)*area)/np.sum(area)
maxCur = np.max(np.abs(res['x']))

# Print post-processing results
print('  f_B [T^2m^2]   %12.4e' % (res['f_B']))
print('  f_S            %12.4e' % (res['f_S']))
print('  <|Bn|/|B|>     %12.4e' % (meanRelBn))
print('  I_max [MA]     %12.4e' % (maxCur))

# Save plots and visualization data to files
wf.make_plot_2d(coordinates='degrees', quantity='nonzero currents')
pl.savefig(OUT_DIR + 'gsco_modular_curr2d.png')
wf.to_vtk(OUT_DIR + 'gsco_modular')

# Generate a 3D plot of the wireframe and plasma if desired
if make_mayavi_plots:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050,800), bgcolor=(1,1,1))
    wf.make_plot_3d(to_show='active', tube_radius=0.006)
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil, 
        nphi=plas_nPhi, ntheta=plas_nTheta, range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True, 
        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=5.5, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'gsco_modular_plot3d.png')


