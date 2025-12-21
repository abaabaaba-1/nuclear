"""
plotting.py

Functions for generating plots of the data in the accompanying subfolders.
"""

import numpy as np

def plot_wireframe(fname, plot_type='2d', to_plot='currents', curr_atol=1e-10):
    """
    Generates a graphical representation of the wireframe based on the data in
    a CSV-formatted file in this archive.

    Parameters
    ----------
        fname: string
            Name of the CSV file containing the wireframe data
        plot_type: string (optional)
            If '2d', will produce a 2d plot of the wireframe; if '3d' will 
            produce a 3d rendering (note: the '3d' option requires the
            mayavi package to be installed)
        to_plot: string (optional)
            If 'currents', will show the currents in each segment;
            if 'nonzero currents', will show the currents in each segment but
                will hide segments carrying zero current;
            if 'constrained segments', will highlight the segments that are
                constrained to carry zero current
        curr_atol: double (optional)
            Tolerance for determining whether a segment carries zero current
    """

    wfdata = np.loadtxt(fname, delimiter=',', skiprows=1)

    segments = np.zeros((wfdata.shape[0], 2, 2))
    segments[:,0,0] = wfdata[:,3]
    segments[:,0,1] = wfdata[:,4]
    segments[:,1,0] = wfdata[:,8]
    segments[:,1,1] = wfdata[:,9]

    if to_plot == 'nonzero currents':
        inds = np.where(np.abs(wfdata[:,10]) > curr_atol)[0]
    else:
        inds = np.arange(segments.shape[0])

    if plot_type.lower() == '2d':

        import matplotlib.pyplot as pl
        from matplotlib.collections import LineCollection
    
        lc = LineCollection(segments[inds])
    
        if to_plot == 'currents' or to_plot == 'nonzero currents':
            lc.set_array(wfdata[inds,10]*wfdata[inds,11]*1e-6)
            maxcurr = np.max(np.abs(wfdata[inds,10]))*1e-6
            lc.set_clim((-maxcurr, maxcurr))
        elif to_plot == 'constrained segments':
            lc.set_array(wfdata[inds,12])
            lc.set_clim([-1,1])
        else:
            raise ValueError('Unrecognized input for `to_plot`')
    
        lc.set_cmap('coolwarm')
    
        pl.figure()
        pl.gca().add_collection(lc)
        pl.gca().set_xlim([np.min(segments[:,0,0])-10, 
                           np.max(segments[:,1,0])+10])
        pl.gca().set_ylim([np.min(segments[:,0,1])-10, 
                           np.max(segments[:,1,1])+10])
        pl.gca().set_xlabel('Phi [deg]')
        pl.gca().set_ylabel('Theta [deg]')
        cb = pl.colorbar(lc)
        if to_plot == 'currents' or to_plot == 'nonzero currents':
            cb.set_label('Current [MA]')
        elif to_plot == 'constrained segments':
            cb.set_label('1 = constrained; 0 = unconstrained')
    
        pl.show()

    elif plot_type.lower() == '3d':

        from mayavi import mlab
    
        x = (wfdata[inds,:])[:,[0,5]].reshape((-1))
        y = (wfdata[inds,:])[:,[1,6]].reshape((-1))
        z = (wfdata[inds,:])[:,[2,7]].reshape((-1))
        s = np.ones((len(inds),2))
        if to_plot == 'currents' or to_plot == 'nonzero currents':
            s[:,0] = wfdata[inds,10]*wfdata[inds,11]
            s[:,1] = wfdata[inds,10]*wfdata[inds,11]
        elif to_plot == 'constrained segments':
            s[:,0] = wfdata[inds,12]
            s[:,1] = wfdata[inds,12]
        s = s.reshape((-1))
    
        pts = mlab.pipeline.scalar_scatter(x, y, z, s)
        connections = np.arange(2*wfdata[inds,:].shape[0]).reshape((-1,2))
        pts.mlab_source.dataset.lines = connections
    
        tube = mlab.pipeline.tube(pts, tube_radius=0.01)
        tube.filter.radius_factor = 1.
        surf = mlab.pipeline.surface(tube, colormap='bwr')
    
        if to_plot == 'currents' or to_plot == 'nonzero currents':
            curr_lim = np.max(np.abs(s))
            surf.module_manager.scalar_lut_manager.data_range = \
                        (-curr_lim, curr_lim)
        elif to_plot == 'constrained segments':
            surf.module_manager.scalar_lut_manager.data_range = (-1, 1)
    
        mlab.show()

def plot_relBnorm(fname, scale='linear'):
    """
    Make a 2D plot of the relative normal component of the magnetic field
    on the target plasma boundary based on the CSV-formatted files in this
    archive.

    Parameters
    ----------
        fname: string
            Name of the CSV file containing the relBnorm data
        scale: string (optional)
            If 'linear', will plot the signed quantity on an linear scale;
            if 'log', will plot the absolute value on a log scale
    """

    import matplotlib.pyplot as pl

    rBndata = np.loadtxt(fname, delimiter=',', skiprows=1)

    nPhi = len(np.unique(rBndata[:,0]))
    nTheta = len(np.unique(rBndata[:,1]))
    Phi = rBndata[:,0].reshape((nPhi,nTheta)).T
    Theta = rBndata[:,1].reshape((nPhi,nTheta)).T
    relBn = rBndata[:,2].reshape((nPhi,nTheta)).T

    if scale=='linear':
        relBnPlt = relBn
        vmin = -np.max(np.abs(relBn))
        vmax = np.max(np.abs(relBn))
        relBn_label = r'$\mathbf{B}\cdot\mathbf{\hat{n}}/|\mathbf{B}|$'
        cmap = 'coolwarm'
    elif scale=='log':
        relBnPlt = np.abs(relBn)
        vmin = 1e-4
        vmax = 0.5
        relBn_label = r'$|\mathbf{B}\cdot\mathbf{\hat{n}}|/|\mathbf{B}|$'
        cmap = 'viridis'
    else:
        raise ValueError('Unrecognized value for `scale`')

    pl.figure()
    pl.imshow(relBnPlt, origin='lower', interpolation='bicubic', aspect='auto',
              extent=[np.min(Phi), np.max(Phi), np.min(Theta), np.max(Theta)],
              norm=scale, vmin=vmin, vmax=vmax, cmap=cmap) 
    pl.gca().set_xlabel('Phi [deg]')
    pl.gca().set_ylabel('Theta [deg]')
    cb = pl.colorbar()
    cb.set_label(relBn_label)

    pl.show()

def plot_fieldlines(fname_flines, fname_bdry=None):
    """
    Generates a Poincare plot according to CSV-formatted data stored in this
    archive

    Parameters
    ----------
        fname_flines: string
            Prefix for the files containing the r and z coordinates of the 
            Poincare points, omitting '_r.csv' or '_z.csv'
        fname_bdry: string (optional)
            Name of the CSV file containing the geometry of the cross-section
            of the target plasma boundary
    """

    import matplotlib.pyplot as pl

    r_lines = np.loadtxt(fname_flines + '_r.csv', delimiter=',', skiprows=1)
    z_lines = np.loadtxt(fname_flines + '_z.csv', delimiter=',', skiprows=1)

    pl.figure()
    pl.plot(r_lines, z_lines, '.', markersize=0.4)

    if fname_bdry is not None:
        bdry_data = np.loadtxt(fname_bdry, delimiter=',', skiprows=1)  
        r_bdry = bdry_data[:,0]
        z_bdry = bdry_data[:,1]
        pl.plot(r_bdry, z_bdry, 'k-', linewidth=2)

    pl.gca().set_xlabel('r [m]')
    pl.gca().set_ylabel('z [m]')

    pl.show()
