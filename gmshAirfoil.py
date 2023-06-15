# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

import sys
import gmsh
from gmshToolkit import *
import shutil 

NACA_type = '0012'

bluntTrailingEdge = False

gridPtsRichness = 0.5
elemOrder = 5
highOrderBLoptim = 0 # (0: none, 1: optimization, 2: elastic+optimization, 3: elastic, 4: fast curving). alternative: Where straight layers in BL are satisfactory, use addPlaneSurface() instead of addSurfaceFilling() and remove this high-order optimisation.

gridPts_alongNACA = int(75*gridPtsRichness)

gridPts_inBL = int(15*gridPtsRichness) # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.05

TEpatchGridFlaringAngle = 30 # deg
gridPts_alongTEpatch = int(8*gridPtsRichness) # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.05

wakeGridFlaringAngle = 10 # deg
gridPts_alongWake = int(25*gridPtsRichness) # > 2 for split into fully hex mesh
gridGeomProg_alongWake = 1.0

pitch = 12.0 # deg
chord = 0.2 # m 

# Initialize gmsh:
gmsh.initialize()

pTS1 = [] # pointTag_struct -- blade 1
lTS1 = [] # lineTag_struct -- blade 1
sTS1 = [] # surfaceTag_struct -- blade 1

pTS2 = [] # pointTag_struct -- blade 2
lTS2 = [] # lineTag_struct -- blade 2
sTS2 = [] # surfaceTag_struct -- blade 2

pointTag = 0
lineTag = 0
surfaceTag = 0
volumeTag = 0

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the airfoil mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

airfoilReferenceAlongChord = 0.5*chord
TEpatchLength = 0.1*chord*np.cos(pitch*np.pi/180) # length of the TEpatch in along the x-axis
wakeLength = 0.5*chord*np.cos(pitch*np.pi/180) # length of the wake in along the x-axis
height_LE = 0.05*chord # Structured Grid offset layer gap at the leading edge
height_TE = 0.1*chord # Structured Grid offset layer gap at the trailing edge
gridPts_inTE = int(gridPts_inBL/7) # if the TE is blunt, number of cells in the TE half height. NB: for the Blossom algorithm to work an even number of faces must be given.

airfoilReferenceAlongChord = 0.5*chord
airfoilReferenceCoordinate = [0.0, 0.0, 0.0]

rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree
shiftVec = [0.0, 0.0, 0.0] # shift of the airfoil origin

structTag = [pointTag, lineTag, surfaceTag]
GeomSpec = [NACA_type, bluntTrailingEdge, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
# [pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)
[pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil_HO(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)

airfoilLine = returnAirfoilContour(lineTag_list, bluntTrailingEdge)
structBLouterLine = returnStructGridOuterContour(lineTag_list, bluntTrailingEdge)
structGridSurf = returnStructGridSide(surfaceTag_list, bluntTrailingEdge)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the exterior region # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

x_min = - 1.5*chord
x_max = 3*chord
y_min = - 1.5*chord
y_max = 1.5*chord
elemSize_rect = chord/20/gridPtsRichness

x_minBUFF = - 1.75*chord
x_maxBUFF = 7*chord
y_minBUFF = - 1.75*chord
y_maxBUFF = 1.75*chord
elemSize_rectBUFF = elemSize_rect

x_minINF = - 10.0*chord
x_maxINF = 20.0*chord
y_minINF = - 10.0*chord
y_maxINF = 10.0*chord
elemSize_rectINF = 20*elemSize_rect


rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree around [axisZ, axisY, axisX]
shiftVec = np.array([0.0, 0.0, 0.0]) # shift of the origin

[ rectLine, pointTag, lineTag] = gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag, rotMat, shiftVec)
[ rectLineBUFF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minBUFF, x_maxBUFF, y_minBUFF, y_maxBUFF, elemSize_rectBUFF, pointTag, lineTag, rotMat, shiftVec)
[ rectLineINF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minINF, x_maxINF, y_minINF, y_maxINF, elemSize_rectINF, pointTag, lineTag, rotMat, shiftVec)

gmsh.model.geo.add_curve_loop( [*rectLine, *structBLouterLine], surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstr = surfaceTag

gmsh.model.geo.add_curve_loop( [*rectLine, *rectLineBUFF], surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstrBUFF = surfaceTag

gmsh.model.geo.add_curve_loop( [*rectLineBUFF, *rectLineINF], surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstrINF = surfaceTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# 2D pavement
# gmsh.option.setNumber("Mesh.Smoothing", 3)
# gmsh.option.setNumber("Mesh.Algorithm", 11) # mesh 2D
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", elemOrder) # gmsh.model.mesh.setOrder(elemOrder)
gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
gmsh.option.setNumber("Mesh.HighOrderOptimize", highOrderBLoptim) # (0: none, 1: optimization, 2: elastic+optimization, 3: elastic, 4: fast curving)
gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder) # just visualisation ??

gmsh.model.mesh.generate(2)

# generating a high quality fully hex mesh is a tall order: 
# https://gitlab.onelab.info/gmsh/gmsh/-/issues/784

# # gmsh.option.setNumber('Mesh.Recombine3DLevel', 0)
# # gmsh.option.setNumber("Mesh.NbTetrahedra", 0)
# # gmsh.option.setNumber("Mesh.Algorithm3D", 4) # mesh 3D

# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2) # most robust way to obtain pure hex mesh: subdivise it
# # gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 3) # perhaps better but conflict with transfinite mesh... to dig further

# gmsh.model.mesh.generate()

# gmsh.model.mesh.recombine()
# gmsh.model.mesh.refine()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the physical group # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(pb_2Dim, [*structGridSurf, surf_unstr], 1, "CFD")
gmsh.model.addPhysicalGroup(pb_2Dim, [surf_unstrBUFF, surf_unstrINF], 2, "Buff")

gmsh.model.addPhysicalGroup(pb_1Dim, [*airfoilLine], 5, "airfoil skin")
gmsh.model.addPhysicalGroup(pb_1Dim, [*rectLine], 6, "regular CAA frontier")

# gmsh.model.addPhysicalGroup(pb_1Dim, [*rectLineBUFF], 5, "BUFF outer contour")

ExtrudUnstruct_bottom = rectLineINF[0]
ExtrudUnstruct_outlet = rectLineINF[1]
ExtrudUnstruct_top = rectLineINF[2]
ExtrudUnstruct_inlet = rectLineINF[3]

gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_inlet], 7, "Inlet BC")
gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_outlet], 8, "Outlet BC")

gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_bottom], 9, "Bottom BC")
gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_top], 10, "Top BC")

# gmsh.model.setColor([(2, 3)], 255, 0, 0)  # Red

[nodePerEntity, elemPerEntity] = countDOF()

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

gmsh.write("NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.msh")
gmsh.write("NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.vtk")
# paraview support for High-order meshes: https://www.kitware.com/high-order-using-gmsh-reader-plugin-in-paraview/


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
