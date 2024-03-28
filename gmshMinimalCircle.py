
import sys
import gmsh
from gmshToolkit import *
import shutil

elemOrder = 8 # discretisation of the line
highOrderBLoptim = 4 # 0: none,
                     # 1: optimization, 
                     # 2: elastic+optimization, 
                     # 3: elastic, 
                     # 4: fast curving
                     # by default choose 4. If for small "gridPts_alongNACA", LE curvature fails, try other values.  
                     
R = 1
dl = R/100
center = [0.0, 0.0, 0.0]
arcPts = int(np.pi/(2*dl))

# Initialize gmsh:
gmsh.initialize()

pointTag = 0
lineTag = 0
surfaceTag = 0

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Points # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$

# circle center:
gmsh.model.geo.addPoint(center[0], center[1], center[2], dl, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodCenter = pointTag
# circle control points:
gmsh.model.geo.addPoint(center[0]+R, center[1], center[2], dl, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt1 = pointTag
gmsh.model.geo.addPoint(center[0], center[1]+R, center[2], dl, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt2 = pointTag
gmsh.model.geo.addPoint(center[0]-R, center[1], center[2], dl, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt3 = pointTag
gmsh.model.geo.addPoint(center[0], center[1]-R, center[2], dl, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt4 = pointTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Lines # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$

gmsh.model.geo.addCircleArc(point_rodPt1, point_rodCenter, point_rodPt2, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, arcPts)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc1 = lineTag
gmsh.model.geo.addCircleArc(point_rodPt2, point_rodCenter, point_rodPt3, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, arcPts)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc2 = lineTag
gmsh.model.geo.addCircleArc(point_rodPt3, point_rodCenter, point_rodPt4, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, arcPts)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc3 = lineTag
gmsh.model.geo.addCircleArc(point_rodPt4, point_rodCenter, point_rodPt1, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, arcPts)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc4 = lineTag

lineLoop = [line_rodArc1, line_rodArc2, line_rodArc3, line_rodArc4]

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the surface (not exported, for visualisation only) # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

gmsh.model.geo.add_curve_loop(lineLoop, surfaceTag+1)
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# high-order params
gmsh.option.setNumber("Mesh.ElementOrder", elemOrder) # gmsh.model.mesh.setOrder(elemOrder)
gmsh.option.setNumber("Mesh.HighOrderOptimize", highOrderBLoptim)
gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder)

gmsh.model.mesh.generate()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the physical group # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()
# gmsh.model.setColor([(2, 3)], 255, 0, 0)  # Red

gmsh.model.addPhysicalGroup(pb_1Dim, [*lineLoop], 1, "Circle")

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

# export mesh with all tags for computation:
gmsh.write("circle_"+str(np.max([4, 4*(arcPts-1)]))+"elems_mo"+str(elemOrder)+".msh")

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
