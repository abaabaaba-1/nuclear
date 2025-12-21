"""
gsco_multistep.py

Optimizes a 96x100 wireframe using a multi-step GSCO optimization to develop 
a solution with saddle coils as shown in Section 4.4 of the paper. A set of
external, planar coils (3 per half-period) supply the toroidal field. The 
wireframe is initialized to be empty, and segment constraints are applied
to ensure that coils form in sectors between the external coils.

Outputs 2d plot of the current distribution in one half-period, as well as a
VTK model of the wireframe. If desired, a 3d plot of the optimized solution
along with the external TF coils may also be rendered (requires the mayavi 
package).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as pl
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe, \
                        create_equally_spaced_curves
from simsopt.solve import optimize_wireframe
from simsopt.field import WireframeField, BiotSavart, Current, \
                          coils_via_symmetries
from simsopt.util import in_github_actions
from helper_functions import constrain_enclosed_segments, \
                             contiguous_coil_size, find_coil_sizes

# Set to True generate a 3d rendering with the mayavi package
make_mayavi_plots = False

# Number of wireframe segments per half period in the toroidal dimension
wf_nPhi = 96

# Number of wireframe segments in the poloidal dimension
wf_nTheta = 100

# Maximum number of GSCO iterations
max_iter = 2500

# How often to print progress
print_interval = 100

# Resolution of test points on plasma boundary (poloidal and toroidal)
plas_n = 32

# Number of planar TF coils in the solution per half period
n_tf_coils_hp = 3

# Toroidal width, in cells, of the restricted regions (breaks) between sectors
break_width = 4

# GSCO loop current as a fraction of net TF coil current
init_gsco_cur_frac = 0.2 

# Minimum size (in enclosed wireframe cells) for a saddle coil
min_coil_size = 20

# Average magnetic field on axis, in Teslas, to be produced by the wireframe.
# This will be used for initializing the TF coils. The radius of the 
# magnetic axis will be estimated from the plasma boundary geometry.
field_on_axis = 1.0

# Weighting factor for the sparsity objective
lambda_S = 10**-7

# File for the desired boundary magnetic surface:
filename_equil = '/home/dataset-assist-0/MOLLM-main/data/Supporting_Files/input.LandremanPaul2021_QA'

# File specifying the geometry of the wireframe surface (made with BNORM)
filename_wf_surf = '/home/dataset-assist-0/MOLLM-main/data/Supporting_Files/nescin.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

############################################################
# End of input parameters.
############################################################

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

# Initialize the wireframe 
tfcoil_current = pol_cur/(2*wf.nfp*n_tf_coils_hp)

# Constrain toroidal segments around the TF coils to prevent new coils from
# being placed there (and to prevent the TF coils from being reshaped)
wf.set_toroidal_breaks(n_tf_coils_hp, break_width)

# Set constraint for net poloidal current (note: the constraint is not strictly 
# necessary for GSCO to work properly, but it can be used as a consistency 
# check for the solution)
wf.set_poloidal_current(0)   

# Create an external set of TF coils
tf_curves = create_equally_spaced_curves(n_tf_coils_hp, surf_plas.nfp, True, 
                                         R0=1.0, R1=0.85)
tf_curr = [Current(-pol_cur/(2*n_tf_coils_hp*surf_plas.nfp)) 
           for i in range(n_tf_coils_hp)]
tf_coils = coils_via_symmetries(tf_curves, tf_curr, surf_plas.nfp, True)
mf_tf = BiotSavart(tf_coils)

# Initialize loop variables
soln_prev = np.full(wf.currents.shape, np.nan)
soln_current = np.array(wf.currents)
cur_frac = init_gsco_cur_frac
loop_count = None
final_step = False
encl_segs = []
n_step = 0

# Multi-step optimization loop
while not final_step:

    n_step += 1

    if not final_step and np.all(soln_prev == soln_current):
        final_step = True
        wf.set_segments_free(encl_segs)

    step_name = 'step %d' % (n_step) if not final_step else 'final adjustment'
    print('------------------------------------------------------------------')
    print('Performing GSCO for ' + step_name)
    print('------------------------------------------------------------------')
    
    # Set the optimization parameters
    if not final_step:
        opt_params = {'lambda_S': lambda_S, 
                      'max_iter': max_iter,
                      'print_interval': print_interval,
                      'no_crossing': True,
                      'max_loop_count': 1, 
                      'loop_count_init': loop_count, 
                      'default_current': np.abs(cur_frac*pol_cur),
                      'max_current': 1.1 * np.abs(cur_frac*pol_cur)
                     }
    else:
        opt_params = {'lambda_S': lambda_S, 
                      'max_iter': max_iter,
                      'print_interval': print_interval,
                      'no_crossing': True,
                      'max_loop_count': 1, 
                      'loop_count_init': loop_count, 
                      'match_current': True,
                      'no_new_coils': True,
                      'default_current': 0,
                      'max_current': 1.1 * np.abs(init_gsco_cur_frac*pol_cur)
                     }

    # Run the GSCO optimization
    t0 = time.time()
    res = optimize_wireframe(wf, 'gsco', opt_params, surf_plas=surf_plas, 
                             ext_field=mf_tf, verbose=False)
    t1 = time.time()
    deltaT = t1 - t0

    print('')
    print('  Post-processing for ' + step_name)
    print('  ------------------------------------')
    print('    opt time [s]   %12.3f' % (deltaT))

    if not final_step:

        # "Sweep" the solution to remove coils that are too small
        coil_sizes = find_coil_sizes(res['loop_count'], wf.get_cell_neighbors())
        small_inds = np.where(\
            np.logical_and(coil_sizes > 0, coil_sizes < min_coil_size))[0]
        adjoining_segs = wf.get_cell_key()[small_inds,:]
        segs_to_zero = np.unique(adjoining_segs.reshape((-1)))
        
        # Modify the solution by removing the small coils
        loop_count = res['loop_count']
        wf.currents[segs_to_zero] = 0
        loop_count[small_inds] = 0

        # Prevent coils from being placed inside existing coils in subsequent 
        # steps
        encl_segs = constrain_enclosed_segments(wf, loop_count)

    # Verify that the solution satisfies all constraints
    assert wf.check_constraints()

    # Re-calculate field after coil removal
    mf_post = WireframeField(wf) + mf_tf
    mf_post.set_points(surf_plas.gamma().reshape((-1,3)))

    # Post-processing
    x_post = np.array(wf.currents).reshape((-1,1))
    f_B_post = 0.5 * np.sum((res['Amat'] @ x_post - res['bvec'])**2)
    f_S_post = 0.5 * np.linalg.norm(x_post.ravel(), ord=0)
    Bfield = mf_post.B().reshape((plas_nPhi, plas_nTheta, 3))
    Bnormal = np.sum(Bfield * surf_plas.unitnormal(), axis=2)
    modB = np.sqrt(np.sum(Bfield**2, axis=2))
    relBnorm = Bnormal/modB
    area = np.sqrt(np.sum(surf_plas.normal()**2, axis=2))/float(modB.size)
    meanRelBn = np.sum(np.abs(relBnorm)*area)/np.sum(area)
    maxCur = np.max(np.abs(res['x']))

    # Print post-processing results
    print('    f_B [T^2m^2]   %12.4e' % (f_B_post))
    print('    f_S            %12.4e' % (f_S_post))
    print('    <|Bn|/|B|>     %12.4e' % (meanRelBn))
    print('    I_max [MA]     %12.4e' % (maxCur))
    print('')

    cur_frac *= 0.5

    soln_prev = soln_current
    soln_current = np.array(wf.currents)


# Save plots and visualization data to files
wf.make_plot_2d(coordinates='degrees', quantity='nonzero currents')
pl.savefig(OUT_DIR + 'gsco_multistep_curr2d.png')
pl.close(pl.gcf())
wf.to_vtk(OUT_DIR + 'gsco_multistep')

# Generate a 3D plot of the wireframe and plasma if desired
if make_mayavi_plots:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050,800), bgcolor=(1,1,1))
    wf.make_plot_3d(to_show='active')
    for tfc in tf_coils:
        tfc.curve.plot(engine='mayavi', show=False, color=(0.75,0.75,0.75),
                       close=True)
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil, 
        nphi=2*surf_plas.nfp*plas_nPhi, ntheta=plas_nTheta, range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True, 
        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=6, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'gsco_multistep_plot3d.png')


