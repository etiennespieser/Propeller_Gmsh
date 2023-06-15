# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

import sys
import gmsh
import shutil 

elemOrder = 10 # (1=linear elements, N (<6) = elements of higher order)

# https://gitlab.onelab.info/gmsh/gmsh/-/issues/1128
# https://github.com/mfem/mfem/issues/2032
# https://www.sciencedirect.com/science/article/pii/S0021999113004956?ref=pdf_download&fr=RR-2&rr=7d66d5048c5e10ac

# https://gitlab.onelab.info/gmsh/gmsh/-/issues/527

# https://wiki.freecad.org/FEM_MeshGmshFromShape

# https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_11_1/tutorials/python/t5.py

# Initialize gmsh:
gmsh.initialize()

pointTag = 0
lineTag = 0
surfaceTag = 0

gmsh.model.geo.addPoint(0.0, 1.0, 0.0, 0.1, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
gmsh.model.geo.addPoint(1.0, 0.5, 0.0, 0.1, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
gmsh.model.geo.addPoint(1.5, 2.0, 0.0, 0.1, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
gmsh.model.geo.addPoint(1.5, 3.0, 0.0, 0.1, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
gmsh.model.geo.addPoint(-0.5, 1.5, 0.0, 0.1, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop

# gmsh.model.geo.add_line(1, 2, lineTag+1)
# lineTag = lineTag+1
# gmsh.model.geo.add_line(2, 3, lineTag+1)
# lineTag = lineTag+1
# https://gitlab.onelab.info/gmsh/gmsh/blob/master/examples/api/spline.py
gmsh.model.geo.addSpline([1,2,3], lineTag+1) 
lineTag = lineTag+1
# gmsh.model.geo.addBSpline([1,2,3], lineTag+1)
# lineTag = lineTag+1

gmsh.model.geo.add_line(3, 4, lineTag+1)
lineTag = lineTag+1
gmsh.model.geo.add_line(4, 5, lineTag+1)
lineTag = lineTag+1
gmsh.model.geo.add_line(5, 1, lineTag+1)
lineTag = lineTag+1

for i in range(1,5):
    gmsh.model.geo.mesh.setTransfiniteCurve(i, 5)

gmsh.model.geo.add_curve_loop([1, 2, 3, 4], surfaceTag+1)
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
surfaceTag = surfaceTag+1

gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.RecombineAll", 1)

gmsh.model.mesh.generate(2)


gmsh.model.mesh.setOrder(elemOrder) 
gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder) # just visualisation ??
gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
gmsh.option.setNumber("Mesh.HighOrderOptimize", 2) # (0: none, 1: optimization, 2: elastic+optimization, 3: elastic, 4: fast curving)

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

gmsh.write("high-order_test.msh")
gmsh.write("high-order_test.vtk")

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
