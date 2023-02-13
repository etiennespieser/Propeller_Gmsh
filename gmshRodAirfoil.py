# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

# aims at reproducing the rod-airfoil benchmark, Casalino, Jacob and Roger aiaaj03 DOI: 10.2514/2.1959

import sys
import gmsh
from gmshToolkit import *
import shutil 

NACA_type = '4412'

bluntTrailingEdge = False
optimisedGridSpacing = True

gridPts_alongNACA = 5

gridPts_inBL = 3 # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.1

TEpatchGridFlaringAngle = 0 # deg
gridPts_alongTEpatch = 5 # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.05

wakeGridFlaringAngle = 0 # deg
gridPts_alongWake = 3 # > 2 for split into fully hex mesh
gridGeomProg_alongWake = 1.0

pitch = 20.0 # deg
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
wakeLength = 0.3*chord*np.cos(pitch*np.pi/180) # length of the wake in along the x-axis
height_LE = 0.05*chord # Structured Grid offset layer gap at the leading edge
height_TE = 0.1*chord # Structured Grid offset layer gap at the trailing edge
gridPts_inTE = int(gridPts_inBL/4) # if the TE is blunt, number of cells in the TE half height. NB: for the Blossom algorithm to work an even number of faces must be given.

airfoilReferenceAlongChord = 0.5*chord
airfoilReferenceCoordinate = [0.0, 0.0, 0.0]

rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree
shiftVec = [0.0, 0.0, 0.0] # shift of the airfoil origin

structTag = [pointTag, lineTag, surfaceTag]
GeomSpec = [NACA_type, bluntTrailingEdge, optimisedGridSpacing, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
[pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)

bladeLine = returnStructGridOuterContour(lineTag_list, bluntTrailingEdge)
structGridSurf = returnStructGridSide(surfaceTag_list, bluntTrailingEdge)


# $$$$$$$$$$$$$$$$$$$$$
# # Creation of rod # #
# $$$$$$$$$$$$$$$$$$$$$

rotMat = rotationMatrix([20.0, 30.0, 40.0]) # angles in degree
shiftVec = [0.0, 0.0, 0.0] # shift of the airfoil origin

rodPos = [-2.0*chord, 0.0, 0.0]
rodR = 0.1*chord
rodElemSize = 0.02*chord
rodBLwidth = 0.05*chord


gridPts_alongRod = int(2*np.pi*rodR/rodElemSize/4)
gridPts_inRodBL = 5
gridGeomProg_inRodBL = 1.1

#### gmeshed_cylinder_line



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Points # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$

rotVec = np.matmul(rotMat, np.array([rodPos[0], rodPos[1], rodPos[2]])) - shiftVec
gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodCenter = pointTag

gmsh.model.geo.addPoint(rodPos[0]+rodR, rodPos[1], rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt1 = pointTag
gmsh.model.geo.addPoint(rodPos[0], rodPos[1]+rodR, rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt2 = pointTag
gmsh.model.geo.addPoint(rodPos[0]-rodR, rodPos[1], rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt3 = pointTag
gmsh.model.geo.addPoint(rodPos[0], rodPos[1]-rodR, rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodPt4 = pointTag

gmsh.model.geo.addPoint(rodPos[0]+rodR+rodBLwidth, rodPos[1], rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodBLpt1 = pointTag
gmsh.model.geo.addPoint(rodPos[0], rodPos[1]+rodR+rodBLwidth, rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodBLpt2 = pointTag
gmsh.model.geo.addPoint(rodPos[0]-rodR-rodBLwidth, rodPos[1], rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodBLpt3 = pointTag
gmsh.model.geo.addPoint(rodPos[0], rodPos[1]-rodR-rodBLwidth, rodPos[2], rodR/10,pointTag+1)
pointTag = pointTag+1 # store the last 'pointTag' from previous loop
point_rodBLpt4 = pointTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Lines # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$

gmsh.model.geo.add_line(point_rodPt1, point_rodBLpt1, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_inRodBL, "Progression", gridGeomProg_inRodBL)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodConnect1 = lineTag
gmsh.model.geo.add_line(point_rodPt2, point_rodBLpt2, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_inRodBL, "Progression", gridGeomProg_inRodBL)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodConnect2 = lineTag
gmsh.model.geo.add_line(point_rodPt3, point_rodBLpt3, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_inRodBL, "Progression", gridGeomProg_inRodBL)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodConnect3 = lineTag
gmsh.model.geo.add_line(point_rodPt4, point_rodBLpt4, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_inRodBL, "Progression", gridGeomProg_inRodBL)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodConnect4 = lineTag

gmsh.model.geo.addCircleArc(point_rodPt1, point_rodCenter, point_rodPt2, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc1 = lineTag
gmsh.model.geo.addCircleArc(point_rodPt2, point_rodCenter, point_rodPt3, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc2 = lineTag
gmsh.model.geo.addCircleArc(point_rodPt3, point_rodCenter, point_rodPt4, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc3 = lineTag
gmsh.model.geo.addCircleArc(point_rodPt4, point_rodCenter, point_rodPt1, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodArc4 = lineTag

gmsh.model.geo.addCircleArc(point_rodBLpt1, point_rodCenter, point_rodBLpt2, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodBLarc1 = lineTag
gmsh.model.geo.addCircleArc(point_rodBLpt2, point_rodCenter, point_rodBLpt3, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodBLarc2 = lineTag
gmsh.model.geo.addCircleArc(point_rodBLpt3, point_rodCenter, point_rodBLpt4, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodBLarc3 = lineTag
gmsh.model.geo.addCircleArc(point_rodBLpt4, point_rodCenter, point_rodBLpt1, lineTag+1)
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, gridPts_alongRod)
lineTag = lineTag+1 # store the last 'lineTag' from previous loop
line_rodBLarc4 = lineTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the Surfaces # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

gmsh.model.geo.add_curve_loop([-line_rodConnect1, line_rodArc1, line_rodConnect2, -line_rodBLarc1], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1 
surf_rodStruct1 = surfaceTag

gmsh.model.geo.add_curve_loop([-line_rodConnect2, line_rodArc2, line_rodConnect3, -line_rodBLarc2], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1 
surf_rodStruct2 = surfaceTag

gmsh.model.geo.add_curve_loop([-line_rodConnect3, line_rodArc3, line_rodConnect4, -line_rodBLarc3], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1 
surf_rodStruct3 = surfaceTag

gmsh.model.geo.add_curve_loop([-line_rodConnect4, line_rodArc4, line_rodConnect1, -line_rodBLarc4], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1 
surf_rodStruct4 = surfaceTag


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the exterior region # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

x_min = - 3*chord
x_max = 1.5*chord
y_min = - chord
y_max = chord
elemSize_rect = chord/10

x_minBUFF = - 3.5*chord
x_maxBUFF = 3*chord
y_minBUFF = - 1.5*chord
y_maxBUFF = 1.5*chord
elemSize_rectBUFF = chord/5

[ rectLine, pointTag, lineTag] = gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag)
[ rectLineBUFF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minBUFF, x_maxBUFF, y_minBUFF, y_maxBUFF, elemSize_rectBUFF, pointTag, lineTag)

# gmsh.model.geo.add_curve_loop( [*rectLine, *bladeLine], surfaceTag+1) 
# gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
# gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
# surfaceTag = surfaceTag+1
# surf_unstr = surfaceTag

gmsh.model.geo.add_curve_loop( [*rectLine, *rectLineBUFF], surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstrBUFF = surfaceTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# 2D pavement
# gmsh.option.setNumber("Mesh.Smoothing", 3)
# gmsh.option.setNumber("Mesh.Algorithm", 11) # mesh 2D
gmsh.option.setNumber("Mesh.RecombineAll", 1)
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

gmsh.model.addPhysicalGroup(pb_2Dim, [*structGridSurf], 1, "CFD") # physical surface
gmsh.model.addPhysicalGroup(pb_2Dim, [surf_unstrBUFF], 2, "Buff") # physical surface

[nodePerEntity, elemPerEntity] = countDOF()

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

gmsh.write("NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.msh")

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
