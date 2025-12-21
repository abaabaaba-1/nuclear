"""
helper_functions.py

Helper functions used in some of the example scripts included in this folder.
"""

import numpy as np
from scipy.linalg import lstsq

def equalize_arc_length(surf):
    """
    Modifies the Fourier coefficients of a SurfaceRZFourier class instance
    such that in each poloidal plane, the quadrature points are equally
    spaced in the poloidal dimension. 

    NOTE: the surface is assumed to have stellarator symmetry.

    Parameters
    ----------
        surf: SurfaceRZFourier class instance
            Surface whose Fourier coeffients are to be modified
    """

    ntheta = len(surf.quadpoints_theta)
    nphi = len(surf.quadpoints_phi)

    delta_xyz = surf.gamma()[:,1:,:] - surf.gamma()[:,:-1,:]
    dists = np.sqrt(np.sum(delta_xyz**2, axis=2))
    arclen = np.cumsum(dists, axis=1)
    Theta = np.zeros((nphi,ntheta,1,1))
    Theta[:,1:,0,0] = 2*np.pi*arclen \
        * surf.quadpoints_theta[-1]/arclen[:,-1].reshape((-1,1))

    Phi = 2*np.pi*surf.quadpoints_phi.reshape((-1,1,1,1))

    Ngrid, Mgrid = np.meshgrid(np.arange(-surf.ntor,surf.ntor+1), 
                             np.arange(surf.mpol+1))
    MR = Mgrid.reshape((-1))[surf.ntor:].reshape((1,1,1,-1))
    NR = Ngrid.reshape((-1))[surf.ntor:].reshape((1,1,1,-1))
    MZ = MR[:,:,:,1:]
    NZ = NR[:,:,:,1:]
    M = np.concatenate((MR, MZ), axis=3)
    N = np.concatenate((NR, NZ), axis=3)

    dgamma_by_dcoeffs = np.zeros((nphi,ntheta,3,M.size))
    dgamma_by_dcoeffs[:,:,0,:MR.size] = \
        (np.cos(MR*Theta - surf.nfp*NR*Phi)*np.cos(Phi))[:,:,0,:]
    dgamma_by_dcoeffs[:,:,1,:MR.size] = \
        (np.cos(MR*Theta - surf.nfp*NR*Phi)*np.sin(Phi))[:,:,0,]
    dgamma_by_dcoeffs[:,:,2,MR.size:] = \
        np.sin(MZ*Theta - surf.nfp*NZ*Phi)[:,:,0,:]

    Amat = dgamma_by_dcoeffs.reshape((ntheta*nphi*3, M.size))
    bvec = surf.gamma().reshape((-1,1))
    soln = lstsq(Amat, bvec)[0]

    surf.set_dofs(soln)

def constrain_enclosed_segments(wframe, loop_count):
    """
    Applies constraints to any segment that is enclosed within a saddle coil.

    Parameters
    ----------
        wframe: ToroidalWireframe class instance
            The wireframe whose segments are to be constrained
        loop_count: integer array
            Array giving the net number of loops of current added to each
            cell in the wireframe (this is output by the GSCO function)

    Returns
    -------
        enclosed_segments: integer array
            Indices of the segments that were constrained
    """

    encl_loops = loop_count != 0
    encl_seg_inds = np.unique(wframe.cell_key[encl_loops,:].reshape((-1)))
    encl_segs = np.full(wframe.n_segments, False)
    encl_segs[encl_seg_inds] = True
    encl_segs[wframe.currents != 0] = False
    wframe.set_segments_constrained(np.where(encl_segs)[0])
    return np.where(encl_segs)[0]

def contiguous_coil_size(ind, count_in, coil_id, coil_ids, loop_count, \
                         neighbors):
    """
    Recursively counts how many cells are enclosed by a saddle coil.

    Parameters
    ----------
        ind: integer
            Index of the first cell to be counted
        count_in: integer
            Running total of cells counted (typically zero; nonzero for
            recursive calls)
        coil_id: integer
            ID number to assign to the coil whose cells are being counted
        coil_ids: integer array
            Array to keep track of which coil ID each cell belongs to
        loop_count: integer array
            Array giving the net number of loops of current added to each
            cell in the wireframe (this is output by the GSCO function)
        neighbors: integer array
            Array giving the indices of the neighboring cells to each cell
            in the wireframe, provided by the `get_cell_neighbors` method

    Returns
    -------
        coil_size: integer
            Number of cells contained within the coil (running total for
            recursive calls)
    """

    # Return without incrementing the count if this cell is already counted
    if loop_count[ind] == 0 or coil_ids[ind] >= 0:
       return count_in

    # Label the current cell with the coil id
    coil_ids[ind] = coil_id

    # Recursively cycle through
    count_out = count_in
    for neighbor_id in neighbors[ind,:]:
        count_out = contiguous_coil_size(neighbor_id, count_out, coil_id, \
                                         coil_ids, loop_count, neighbors)

    # Increment the count for the current cell
    return count_out + 1

def find_coil_sizes(loop_count, neighbors):
    """
    Determines the sizes of the saddle coils in a wireframe GSCO solution

    Parameters
    ----------
        loop_count: integer array
            Array giving the net number of loops of current added to each
            cell in the wireframe (this is output by the GSCO function)
        neighbors: integer array
            Array giving the indices of the neighboring cells to each cell
            in the wireframe, provided by the `get_cell_neighbors` method

    Returns
    -------
        coil_sizes: integer array
            For each wireframe cell, provides the size of the coil to which
            that cell belongs (zero if the cell is not part of a coil)
    """

    unique_coil_ids = []
    coil_ids = np.full(len(loop_count), -1)
    coil_sizes = np.zeros(len(loop_count))
    coil_id = -1

    for i in range(len(loop_count)):
        if loop_count[i] != 0:
            coil_id += 1
            unique_coil_ids.append(coil_id)
            count = contiguous_coil_size(i, 0, coil_id, coil_ids, \
                                         loop_count, neighbors)
            coil_sizes[coil_ids==coil_id] = count

    return coil_sizes


