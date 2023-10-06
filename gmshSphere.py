# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

import sys
import gmsh
from gmshToolkit import *
import shutil 

x_center = 0.0
y_center = 0.0
z_center = 0.0
radius = 1.0
elemSize = 0.2

elemOrder = 8 # 8 is max order supported my navier_mfem: github.com/mfem/mfem/issues/3759
highOrderBLoptim = 4 # 0: none,
                     # 1: optimization, 
                     # 2: elastic+optimization, 
                     # 3: elastic, 
                     # 4: fast curving
                     # by default choose 4. If for small "gridPts_alongNACA", LE curvature fails, try other values. 

gmsh.initialize()

pointTag = 0
lineTag = 0
surfaceTag = 0
volumeTag = 0

pb_2Dim = 2

# [sphereSurfTri, pointTag, lineTag, surfaceTag] = gmeshed_sphereTri_surf(x_center-3*radius, y_center, z_center, radius, elemSize, pointTag, lineTag, surfaceTag)
[sphereSurfQuad, pointTag, lineTag, surfaceTag] = gmeshed_sphereQuad_surf(x_center, y_center, z_center, radius, elemSize, pointTag, lineTag, surfaceTag)


gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", elemOrder) # gmsh.model.mesh.setOrder(elemOrder)
gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
gmsh.option.setNumber("Mesh.HighOrderOptimize", highOrderBLoptim) # NB: Where straight layers in BL are satisfactory, use addPlaneSurface() instead of addSurfaceFilling() and remove this high-order optimisation.
gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder) # just visualisation ??

gmsh.model.mesh.generate(2)

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()
[nodePerEntity, elemPerEntity] = countDOF()

gmsh.model.addPhysicalGroup(pb_2Dim, [*sphereSurfQuad], 1, "Spherical Grid Quad")

# gmsh.model.addPhysicalGroup(pb_2Dim, [*sphereSurfTri], 2, "Spherical Grid Tri")

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

# export mesh with all tags for computation:
gmsh.write("sphere_"+str(sum(elemPerEntity))+"elems_o"+str(elemOrder)+".msh")

# delete the "__pycache__" folder:
try:
    shutil.rmtree("__pycache__")
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

# Creates  graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()