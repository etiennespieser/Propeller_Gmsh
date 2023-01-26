# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser
# ------------------------------------------------------------------------------------

import sys
import gmsh
from gmshToolkit import *
import shutil

NACA_type = '0012'

bluntTrailingEdge = False
optimisedGridSpacing = True

gridPts_alongNACA = 30

gridPts_inBL = 4 # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.1

TEpatchGridFlaringAngle = 10 # deg
gridPts_alongTEpatch = 5 # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.05

wakeGridFlaringAngle = 0 # deg
gridPts_alongWake = 5 # > 2 for split into fully hex mesh
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
height_TE = 0.08*chord # Structured Grid offset layer gap at the trailing edge
gridPts_inTE = int(gridPts_inBL/4) # if the TE is blunt, number of cells in the TE half height. NB: for the Blossom algorithm to work an even number of faces must be given.

airfoilReferenceAlongChord = 0.5*chord
airfoilReferenceCoordinate = [0.0, 0.0, 0.0]

rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree

structTag = [pointTag, lineTag, surfaceTag]
GeomSpec = [NACA_type, bluntTrailingEdge, optimisedGridSpacing, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
[pTL_slice, lTL_slice, sTL_slice, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat)

bladeLine = returnStructGridOuterContour(lTL_slice, bluntTrailingEdge)
structGridSurf = returnStructGridSide(sTL_slice, bluntTrailingEdge)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Definition of the tip termination # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# MODIFY LAST SLICE TO BE A NACA 0012. Create a tip with a half NACA 0012 profile

rotMat = rotationMatrix([-pitch, -pitch, 90.0]) # angles in degree

structTag = [pointTag, lineTag, surfaceTag]
GeomSpec = ['0012', bluntTrailingEdge, optimisedGridSpacing, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
[pTL_tip, lTL_tip, sTL_tip, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat)




# Tags for easily accessing the list elements

pLE = 0
pTE = 1
pTEu = 2
pTEl = 3
pTEwake = 4
pTEfarWake = 5
pleft = 6
pup = 7
pupRight = 8
pupFarRight = 9
plow = 10
plowRight = 11
plowFarRight = 12
pupMidRight = 13
plowMidRight = 14
pupMidFarRight = 15
plowMidFarRight = 16

lairfoilUp = 0
lairfoilLow = 1
lBLup = 2
lBLlow = 3
lBLrad = 4
lA = 5
lB = 6
lC = 7
lD = 8
lEu = 9
lEl = 10
lFu = 11
lFl = 12
lG = 13
lH = 14
lI = 15
lJu = 16
lJl = 17
lK = 18
lL = 19
lM = 20
lN = 21
lO = 22
lP = 23
lAr = 24
lBr = 25
lAer = 26
lBer = 27


# convention for the tip line directions. From the side to the tip !!

gridPts_tipSide = 10

# $$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the lines # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$

# connecting together the skeleton of the new generting airfoil and the last propeller slice
gmsh.model.geo.add_line(pTL_tip[pLE]-1, pTL_slice[pLE], lineTag+1)
lineTag = lineTag+1
line_tipConnectionToLE = lineTag
gmsh.model.geo.add_line(pTL_tip[pleft]-1, pTL_slice[pleft], lineTag+1)
lineTag = lineTag+1
line_tipConnectionToLeft = lineTag

if ~bluntTrailingEdge:
    gmsh.model.geo.add_line(pTL_slice[pTE], pTL_tip[pTEu]+1, lineTag+1)
    lineTag = lineTag+1
    line_tipConnectionToTE = lineTag

# creating the oblique lines
line_TEuTipU = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pTEu]+i, pTL_tip[pTEu]+i, lineTag+1)
    lineTag = lineTag+1
line_LEtipU = lineTag
line_LEtipL = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pLE]+1+i, pTL_tip[pLE]-1-i, lineTag+1)
    lineTag = lineTag+1
line_TElTipL = lineTag

line_upTipU = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pup]+i, pTL_tip[pup]+i, lineTag+1)
    lineTag = lineTag+1
line_leftTipU = lineTag
line_leftTipL = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pleft]+1+i, pTL_tip[pleft]-1-i, lineTag+1)
    lineTag = lineTag+1
line_lowTipL = lineTag

gmsh.model.geo.add_line(pTL_slice[pupRight], pTL_tip[pupRight], lineTag+1)
lineTag = lineTag+1
line_upMidRightTipU = lineTag
gmsh.model.geo.add_line(pTL_slice[plowRight], pTL_tip[pupRight], lineTag+1)
lineTag = lineTag+1
line_upMidRightTipU = lineTag

if bluntTrailingEdge:
    gmsh.model.geo.add_line(pTL_slice[pupMidRight], pTL_tip[pupMidRight], lineTag+1)
    lineTag = lineTag+1
    line_upMidRightTipU = lineTag

    gmsh.model.geo.add_line(pTL_slice[plowMidRight], pTL_tip[pupMidRight], lineTag+1)
    lineTag = lineTag+1
    line_upMidRightTipU = lineTag




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the surfaces # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# generate transfinite curves
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.mesh.setTransfiniteCurve(line_TEuTipU+i, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_LEtipL+i, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_upTipU+i, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_leftTipL+i, gridPts_tipSide)
gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToLE, 2)
gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToLeft, 2)
gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToTE, 2)

# connecting together the skeleton of the new generting airfoil and the last propeller slice
# LE:
gmsh.model.geo.add_curve_loop([-line_tipConnectionToLE, -lTL_slice[lG], line_tipConnectionToLeft, (lTL_tip[lG]-1)], surfaceTag+1)
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_tipLEconnectionStructGridUp = surfaceTag
# TE:
print("here TE connection pannel to be coded..")

# airfoil tipSkin uper side
BLstructStartSurfTag_tipU = surfaceTag+1
if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][0], line_TEuTipU, -lTL_slice[lairfoilUp][0], -(line_TEuTipU+1)], surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
else:
    gmsh.model.geo.add_curve_loop([-lTL_slice[lairfoilUp][0], -(line_TEuTipU+1), line_tipConnectionToTE], surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1

for i in range(1,gridPts_alongNACA-2):
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][i], line_TEuTipU+i, -lTL_slice[lairfoilUp][i], -(line_TEuTipU+i+1)], surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
gmsh.model.geo.add_curve_loop([line_tipConnectionToLE, (line_TEuTipU+gridPts_alongNACA-2), -lTL_slice[lairfoilUp][gridPts_alongNACA-2]], surfaceTag+1)
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
BLstructEndSurfTag_tipU = surfaceTag

# airfoil tipSkin lower side

# print([line_LEtipL, line_tipConnectionToLE, lTL_slice[lairfoilLow][0]])

BLstructStartSurfTag_tipL = surfaceTag+1
gmsh.model.geo.add_curve_loop([-line_tipConnectionToLE, -lTL_slice[lairfoilLow][0], -line_LEtipL], surfaceTag+1)
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
for i in range(1,gridPts_alongNACA-2):
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][-i-1], -(line_LEtipL+i-1), lTL_slice[lairfoilLow][i], (line_LEtipL+i)], surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1

if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][-gridPts_alongNACA+1], -(line_LEtipL+gridPts_alongNACA-3), lTL_slice[lairfoilLow][gridPts_alongNACA-2], (line_LEtipL+gridPts_alongNACA-2)], surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
else:
    gmsh.model.geo.add_curve_loop([-line_tipConnectionToTE, -lTL_slice[lairfoilLow][-1], (line_LEtipL+gridPts_alongNACA-3)], surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    

BLstructEndSurfTag_tipL = surfaceTag








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