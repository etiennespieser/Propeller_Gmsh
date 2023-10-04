# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------


# gmsh script based on the geo kernel to enable higher order elements
# http://web4.cs.ucl.ac.uk/research/vis/toast/demo_meshgen1/demo_meshgen1.html
# https://github.com/toastpp/toastpp/blob/master/examples/matlab/gmsh/sphere.geo


import sys
import gmsh
from gmshToolkit import *
import shutil 

x_center = 0.0
y_center = 0.0
z_center = 0.0
radius = 1.0
elemSize = 0.2

elemOrder = 2 # 8 is max order supported my navier_mfem: github.com/mfem/mfem/issues/3759
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

pb_1Dim = 1
pb_2Dim = 2
pb_3Dim = 3

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Points # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$

gmsh.model.geo.addPoint(x_center, y_center, z_center, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_center = pointTag

gmsh.model.geo.addPoint(x_center+radius, y_center, z_center, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_right = pointTag
gmsh.model.geo.addPoint(x_center-radius, y_center, z_center, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_left = pointTag

gmsh.model.geo.addPoint(x_center, y_center+radius, z_center, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_top = pointTag
gmsh.model.geo.addPoint(x_center, y_center-radius, z_center, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_bottom = pointTag

gmsh.model.geo.addPoint(x_center, y_center, z_center+radius, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_front = pointTag
gmsh.model.geo.addPoint(x_center, y_center, z_center-radius, radius/10 ,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_back = pointTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Lines # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$

print(np.round((np.pi*radius/2)/elemSize)+1)

gmsh.model.geo.addCircleArc(point_back, point_center, point_right, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_backRight = lineTag
gmsh.model.geo.addCircleArc(point_right, point_center, point_front, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_frontRight = lineTag
gmsh.model.geo.addCircleArc(point_front, point_center, point_left, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_frontLeft = lineTag
gmsh.model.geo.addCircleArc(point_left, point_center, point_back, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_backLeft = lineTag

gmsh.model.geo.addCircleArc(point_top, point_center, point_right, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_topRight = lineTag
gmsh.model.geo.addCircleArc(point_right, point_center, point_bottom, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_bottomRight = lineTag
gmsh.model.geo.addCircleArc(point_bottom, point_center, point_left, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_bottomLeft = lineTag
gmsh.model.geo.addCircleArc(point_left, point_center, point_top, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_topLeft = lineTag

gmsh.model.geo.addCircleArc(point_back, point_center, point_top, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_backTop = lineTag
gmsh.model.geo.addCircleArc(point_top, point_center, point_front, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_frontTop = lineTag
gmsh.model.geo.addCircleArc(point_front, point_center, point_bottom, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_frontBottom = lineTag
gmsh.model.geo.addCircleArc(point_bottom, point_center, point_back, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, np.int(np.round((np.pi*radius/2)/elemSize)+1))
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_backBottom = lineTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Surfaces # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

gmsh.model.geo.add_curve_loop([-line_backRight, line_topRight, line_backTop], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
# gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1 
surf_cylSide1 = surfaceTag


# [ cylSurf1, pointTag, lineTag, surfaceTag] = gmeshed_cylinder_surf(y_min, y_max, r_cyl, elemSize, pointTag, lineTag, surfaceTag)

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

# gmsh.model.addPhysicalGroup(pb_2Dim, [sphereSurf1], 1, "Cylinder Grid")


# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

# export mesh with all tags for computation:
gmsh.write("Cylinder_"+str(sum(elemPerEntity))+"elems.msh")

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