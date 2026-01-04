          ------------------------------------------------------------
          |                    Data and code for                     |
          |                                                          |
          |           "A framework for discrete optimization         |
          |                   of stellarator coils"                  |
          |                                                          |
          |                       K. C. Hammond                      |
          ------------------------------------------------------------

Author information:

  Kenneth C. Hammond
  Princeton Plasma Physics Laboratory
  100 Stellarator Road, Princeton, NJ 08540, USA
  Email: khammond@pppl.gov
  ORCID: 0000-0002-1104-4434

This study was carried out from approx. November 2023 through February 2025.

Methodology for data collection are described in the associated journal 
publication: https://doi.org/10.1088/1741-4326/adbb02

Data are stored at Princeton University, Princeton, NJ, USA.

This work was conducted under the Laboratory Directed Research and Development
(LDRD) Program at the Princeton Plasma Physics Laboratory, a national laboratory
operated by Princeton University for the U.S. Department of Energy under 
contract number DE-AC02-09CH11466. The U.S. Government retains a non-exclusive, 
paid-up, irrevocable, world-wide license to publish or reproduce the published 
form of this manuscript, or alow others to do so, for U.S. Government purposes.


Software used
-------------

This work employed the following software:

  Simsopt: stellarator optimization code 
    Documentation: https://simsopt.readthedocs.io/en/latest/index.html#
    Source code: https://github.com/hiddenSymmetries/simsopt
  
  STELLOPT: suite of stellarator optimization and modeling codes
    This study specifically utilized BNORM.
    Documentation: https://princetonuniversity.github.io/STELLOPT/STELLOPT.html
    Source code: https://github.com/PrincetonUniversity/STELLOPT


Description of files
--------------------

The data in this directory are organized in subfolders, most of which 
correspond to figures in the associated publication. In addition, the subfolder
"Scripts" contains Python scripts for reproducing the optimizations described
in the paper. In order to run these, Simsopt must be installed. Installation
instructions may be found on the Simsopt documentation site:
https://simsopt.readthedocs.io/en/latest/installation.html

Also in the "Scripts" subfolder the file plotting.py contains functions that
generate plots of the data contained in the CSV-formatted files associated 
with each figure.

Details on the contents of each folder are described below.

 
Figure_03
---------

  wf_data_halfperiod.csv 

    CSV-formatted data file containing data about the wireframe and its current 
    distribution. Each row corresponds to one segment. The data represent one 
    half-period. The columns are as follows:

      p1x: Cartesian x component [m] of the start point of the segment
      p1y: Cartesian y component [m] of the start point of the segment
      p1z: Cartesian z component [m] of the start point of the segment
      p1phi: Toroidal phi angle [deg] of the start point of the segment
      p1theta: Poloidal theta angle [deg] of the start point of the segment
      p1x: Cartesian x component [m] of the end point of the segment
      p1y: Cartesian y component [m] of the end point of the segment
      p1z: Cartesian z component [m] of the end point of the segment
      p1phi: Toroidal phi angle [deg] of the end point of the segment
      p1theta: Poloidal theta angle [deg] of the end point of the segment
      current: Current [A] carried by the segment (positive current flows from 
        the start point to the end point)
      orientation: for toroidal segments, +1 if positive current flows in the
        positive toroidal direction and -1 if positive current flows in the 
        negative toroidal direction; for poloidal segments, +1 if positive
        current flows in the positive poloidal direction and -1 if positive 
        current flows in the negative poloidal direction (note: in general, 
        consistent with stellarator symmetry, segments in even half-periods 
        have positive orientations and segments in odd half-periods have 
        negative orientations)
      constrained: 1 if the segment is constrained to carry no current; 
        0 otherwise

    To generate a plot of this data, use the function "plot_wireframe" available
    in plotting.py in the Scripts subfolder (more information on usage is 
    available in the docstring).
    
  wf_data_fulltorus.csv

    Like wf_data_halfperiod.csv as described above, but including data for all 
    of the segments in the torus.

  wf_model_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe, which 
    can be loaded into software such as ParaView for visualization. The 
    properties of the wireframe in this model are as follows:

      current: the current [A] in each segment
      constrained: 1 if the segment is constrained to carry no current; 
        0 otherwise
      constrained_exp: 1 if the segment is "explicitly" constrained, i.e. 
        directly set by the user to be constrained; 0 otherwise
      constrained_imp: 1 if the segment is "implicitly" constrained, i.e.
        not set as constrained by the user but effectively constrained to 
        carry zero current due to other constraints

  wf_model_fulltorus.vtu

    Like wf_model_halfperiod.vtu as described above, but including data for
    the full torus.


Figure_04
---------

  relBnorm.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary, relative to the
    total magnetic field strength. Each row represents one test point. The 
    columns are as follows:

      phi: toroidal angle [deg] of the test point
      theta: poloidal angle [deg] of the test point
      relBnorm: relative normal field at the test point

    To generate a plot of this data, use the function "plot_relBnorm" available
    in plotting.py in the Scripts subfolder (more information on usage is
    available in the docstring).

  target_boundary.csv

    CSV-formatted file containing the radial (r) and vertical (z) coordinates
    of points on a cross-section of the target plasma boundary. The columns
    are as follows:

      r: radial (r) coordinate [m]
      z: vertical (z) coordinate [m]

  fieldlines_poincare_r.csv

    CSV-formatted file containing the radial (r) components [m] of Poincare 
    "puncture points" where field lines intersect the cross-section. Each
    column corresponds to a different field line; each row represents a 
    different puncture point for each field line.

    To generate a plot of this data, use the function "plot_fieldlines" 
    available in plotting.py in the Scripts subfolder (more information on 
    usage is available in the docstring).

  fieldlines_poincare_z.csv

    Like fieldlines_poincare_r.csv, but with the vertical (z) components [m] of 
    the puncture points.


Figure_05
---------

  wf_data_halfperiod.csv

    CSV-formatted data file containing data about the wireframe and its current 
    distribution for one half-period. See notes for `wf_data_halfperiod.csv` 
    under Figure_03 for details on formatting. 

  wf_data_fulltorus.csv

    Like wf_data_halfperiod.csv as described above, but including data for all 
    of the segments in the torus.

  wf_model_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe, which 
    can be loaded into software such as ParaView for visualization. See notes
    for `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_fulltorus.vtu

    Like wf_model_halfperiod.vtu as described above, but including data for
    the full torus.

  ports.vtu

    VTK-formatted file with models for each of the ports used to constrain
    the wireframe current distribution. 


Figure_09
---------

  wf_data_9a_halfperiod.csv

    CSV-formatted file containing data about the wireframe in Fig. 9a 
    (initialization) and its current distribution for one half-period. See 
    notes for `wf_data_halfperiod.csv` under Figure_03 or details on formatting.

  wf_data_9a_fulltorus.csv

    Like wf_data_9a_halfperiod.csv as described above, but including data for 
    all of the segments in the torus.

  wf_model_9a_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe in 
    Fig. 9a (initialization), which can be loaded into software such as 
    ParaView for visualization. See notes for `wf_model_halfperiod.vtu` under 
    Figure_03 for details on formatting. 

  wf_model_9a_fulltorus.vtu

    Like wf_model_9a_halfperiod.vtu as described above, but including data for
    the full torus.

  relBnorm_9a.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary for Fig. 9a
    (initialization), relative to the total magnetic field strength. See
    notes for `relBnorm.csv` under Figure_04 for details on formatting.

  wf_data_9b_halfperiod.csv

    CSV-formatted file containing data about the wireframe in Fig. 9b 
    (solution with lambda_S = 10^-9 T^2 m^2) and its current distribution for 
    one half-period. See notes for `wf_data_halfperiod.csv` under Figure_03 or 
    details on formatting.

  wf_data_9b_fulltorus.csv

    Like wf_data_9b_halfperiod.csv as described above, but including data for 
    all of the segments in the torus.

  wf_model_9b_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe in 
    Fig. 9b (solution with lambda_S = 10^-9 T^2 m^2), which can be loaded into 
    software such as ParaView for visualization. See notes for 
    `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_9b_fulltorus.vtu

    Like wf_model_9b_halfperiod.vtu as described above, but including data for
    the full torus.

  relBnorm_9b.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary for Fig. 9b
    (solution with lambda_S = 10^-9 T^2 m^2), relative to the total magnetic 
    field strength. See notes for `relBnorm.csv` under Figure_04 for details on 
    formatting.

  wf_data_9c_halfperiod.csv

    CSV-formatted file containing data about the wireframe in Fig. 9c 
    (solution with lambda_S = 10^-6 T^2 m^2) and its current distribution for 
    one half-period. See notes for `wf_data_halfperiod.csv` under Figure_03 or 
    details on formatting.

  wf_data_9c_fulltorus.csv

    Like wf_data_9c_halfperiod.csv as described above, but including data for 
    all of the segments in the torus.

  wf_model_9c_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe in 
    Fig. 9c (solution with lambda_S = 10^-6 T^2 m^2), which can be loaded into 
    software such as ParaView for visualization. See notes for 
    `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_9c_fulltorus.vtu

    Like wf_model_9c_halfperiod.vtu as described above, but including data for
    the full torus.

  relBnorm_9c.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary for Fig. 9c
    (solution with lambda_S = 10^-6 T^2 m^2), relative to the total magnetic 
    field strength. See notes for `relBnorm.csv` under Figure_04 for details on 
    formatting.

  wf_data_9d_halfperiod.csv

    CSV-formatted file containing data about the wireframe in Fig. 9d 
    (solution with lambda_S = 10^-5 T^2 m^2) and its current distribution for 
    one half-period. See notes for `wf_data_halfperiod.csv` under Figure_03 or 
    details on formatting.

  wf_data_9d_fulltorus.csv

    Like wf_data_9d_halfperiod.csv as described above, but including data for 
    all of the segments in the torus.

  wf_model_9d_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe in 
    Fig. 9d (solution with lambda_S = 10^-5 T^2 m^2), which can be loaded into 
    software such as ParaView for visualization. See notes for 
    `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_9d_fulltorus.vtu

    Like wf_model_9d_halfperiod.vtu as described above, but including data for
    the full torus.

  relBnorm_9d.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary for Fig. 9d
    (solution with lambda_S = 10^-5 T^2 m^2), relative to the total magnetic 
    field strength. See notes for `relBnorm.csv` under Figure_04 for details on 
    formatting.


Figure_10
---------

  objectives_m9.csv

    CSV-formatted file containing data on the objective function values at 
    selected GSCO iterations for the solution with lambda_S = 10^-9 T^2 m^2.
    The columns are as follows:

      iter: iteration number
      f_B: value of the field accuracy objective f_B [T^2 m^2]
      f_S: value of the sparsity objective f_S

  objectives_m6.csv

    Like the above, but for the solution with lambda_S = 10^-9 T^2 m^2.

  objectives_m5.csv

    Like the above, but for the solution with lambda_S = 10^-5 T^2 m^2.


Figure_11
---------

  wf_data_halfperiod.csv

    CSV-formatted data file containing data about the wireframe and its 
    constrained segments for one half-period. See notes for 
    `wf_data_halfperiod.csv` under Figure_03 for details on formatting. 

  wf_data_fulltorus.csv

    Like wf_data_halfperiod.csv as described above, but including data for all 
    of the segments in the torus.

  wf_model_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe, which 
    can be loaded into software such as ParaView for visualization. See notes
    for `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_fulltorus.vtu

    Like wf_model_halfperiod.vtu as described above, but including data for
    the full torus.


Figure_12
---------

  wf_data_halfperiod.csv

    CSV-formatted data file containing data about the wireframe and its current 
    distribution for one half-period. See notes for `wf_data_halfperiod.csv` 
    under Figure_03 for details on formatting. 

  wf_data_fulltorus.csv

    Like wf_data_halfperiod.csv as described above, but including data for all 
    of the segments in the torus.

  wf_model_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe, which 
    can be loaded into software such as ParaView for visualization. See notes
    for `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_fulltorus.vtu

    Like wf_model_halfperiod.vtu as described above, but including data for
    the full torus.

  relBnorm.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary, relative to the
    total magnetic field strength. See notes for `relBnorm.csv` under Figure_04 
    for details on formatting.

  target_boundary.csv

    CSV-formatted file containing the radial (r) and vertical (z) coordinates
    of points on a cross-section of the target plasma boundary. The columns
    are as follows:

      r: radial (r) coordinate [m]
      z: vertical (z) coordinate [m]

  fieldlines_poincare_r.csv

    CSV-formatted file containing the radial (r) components [m] of Poincare 
    "puncture points" where field lines intersect the cross-section. Each
    column corresponds to a different field line; each row represents a 
    different puncture point for each field line.

  fieldlines_poincare_z.csv

    Like fieldlines_poincare_r.csv, but with the vertical (z) components [m] of 
    the puncture points.

Figure_13
---------

  wf_data_halfperiod.csv

    CSV-formatted data file containing data about the wireframe and its current 
    distribution for one half-period. See notes for `wf_data_halfperiod.csv` 
    under Figure_03 for details on formatting. 

  wf_data_fulltorus.csv

    Like wf_data_halfperiod.csv as described above, but including data for all 
    of the segments in the torus.

  wf_model_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe, which 
    can be loaded into software such as ParaView for visualization. See notes
    for `wf_model_halfperiod.vtu` under Figure_03 for details on formatting. 

  wf_model_fulltorus.vtu

    Like wf_model_halfperiod.vtu as described above, but including data for
    the full torus.

  relBnorm.csv

    CSV-formatted file containing data on the normal component of the magnetic
    field at selected points on the target plasma boundary, relative to the
    total magnetic field strength. See notes for `relBnorm.csv` under Figure_04 
    for details on formatting.

  target_boundary.csv

    CSV-formatted file containing the radial (r) and vertical (z) coordinates
    of points on a cross-section of the target plasma boundary. The columns
    are as follows:

      r: radial (r) coordinate [m]
      z: vertical (z) coordinate [m]

  fieldlines_poincare_r.csv

    CSV-formatted file containing the radial (r) components [m] of Poincare 
    "puncture points" where field lines intersect the cross-section. Each
    column corresponds to a different field line; each row represents a 
    different puncture point for each field line.

  fieldlines_poincare_z.csv

    Like fieldlines_poincare_r.csv, but with the vertical (z) components [m] of 
    the puncture points.


Figure_15
---------

  wf_orig_data_halfperiod.csv

    CSV-formatted data file containing data about the wireframe (with the
    original structure) and its current distribution for one half-period. See 
    notes for `wf_data_halfperiod.csv` under Figure_03 for details on 
    formatting. 

  wf_eqlg_data_halfperiod.csv

    Like the above, but for the wireframe with equalized poloidal segment
    lengths.

  wf_orig_data_fulltorus.csv

    Like wf_orig_data_halfperiod.csv as described above, but including data for 
    all of the segments in the torus.

  wf_eqlg_data_fulltorus.csv

    Like wf_eqlg_data_halfperiod.csv as described above, but including data for 
    all of the segments in the torus.

  wf_orig_model_halfperiod.vtu

    VTK-formatted file with a model of one half-period of the wireframe (with
    the original structure, which can be loaded into software such as ParaView 
    for visualization. See notes for `wf_model_halfperiod.vtu` under Figure_03 
    for details on formatting. 

  wf_eqlg_model_halfperiod.vtu

    Like the above, but for the wireframe with equalized poloidal segment 
    lengths.

  wf_orig_model_fulltorus.vtu

    Like wf_orig_model_halfperiod.vtu as described above, but including data for
    the full torus.

  wf_eqlg_model_fulltorus.vtu

    Like wf_eqlg_model_halfperiod.vtu as described above, but including data for
    the full torus.

  orig_fieldlines_poincare_r.csv

    CSV-formatted file containing the radial (r) components [m] of Poincare 
    "puncture points" where field lines intersect the cross-section in the 
    magnetic field from the wireframe with the original structure. Each
    column corresponds to a different field line; each row represents a 
    different puncture point for each field line.

  orig_fieldlines_poincare_z.csv

    Like orig_fieldlines_poincare_r.csv, but with the vertical (z) 
    components [m] of the puncture points.

  eqlg_fieldlines_poincare_r.csv

    Like orig_fieldlines_poincare_r.csv, but for the field from the wireframe
    with equalized poloidal segment lengths.

  eqlg_fieldlines_poincare_z.csv

    Like eqlg_fieldlines_poincare_z.csv, but for the field from the wireframe
    with equalized poloidal segment lengths.


Scripts
-------

  rcls_basic.py

    Produces the basic RCLS optimization in Section 3.2

  rcls_ports.py

    Produces the RCLS optimization on a wireframe with constraints to prevent
    current flow through a set of ports as described in Section 3.3

  gsco_modular.py

    Produces the GSCO solution with modular coils from Section 4.2

  gsco_sector.py

    Produces the GSCO solution with sector-confined saddle coils from 
    Section 4.3

  gsco_multistep.py

    Produces the multi-step GSCO solution with sector-confined saddle coils
    with different current values as described in Section 4.4

  helper_functions.py

    Helper functions utilized in the above scripts

  plotting.py

    Functions that generate plots based on the data contained in the files
    in the "Figure" subfolders


Supporting_Files
----------------

  input.LandremanPaul2021_QA

    VMEC input file specifying the "Precise QA" equilibrium described in
    Landreman and Paul, Phys. Rev. Lett. 128, 035001 (2022). Retrieved
    from the Simsopt repository (https://github.com/hiddenSymmetries/simsopt).

  nescin.LandremanPaul2021_QA

    NESCIN/REGCOIL input specification file, generated by the BNORM code, that
    includes Fourier coeffients for a winding surface spaced about 0.3 m from 
    the boundary of the Precise QA equilibrium. The winding surface is used
    to construct the wireframes in some of the examples in the Scripts 
    subfolder.
