# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser
# ------------------------------------------------------------------------------------

import sys
import gmsh
from gmshToolkit import *
import shutil

import traceback # used for try catch statement
import logging # used for try catch statement


NACA_type = '0012'

bluntTrailingEdge = False
optimisedGridSpacing = True

gridPts_alongNACA = 10

gridPts_inBL = 5 # > 2 for split into fully hex mesh
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

sairfoil = 0
sBLstructGrid = 1
sBLstructGridUp = 2
sBLstructGridLow = 3
sTEpatchUp = 4
sTEpatchLow = 5
sTEpatchMidUp = 6
sTEpatchMidLow = 7
swakeUp = 8
swakeLow = 9
swakeMidUp = 10
swakeMidLow = 11


# convention for the tip line directions. From the side to the tip !!

gridPts_tipSide = max(gridPts_inTE,3) # enforce this to be able to connect with the propeller geom

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

if not bluntTrailingEdge:
    gmsh.model.geo.add_line(pTL_slice[pTE], pTL_tip[pTEu]+1, lineTag+1)
    lineTag = lineTag+1
    line_tipConnectionToTEalongAirfoil = lineTag

    gmsh.model.geo.add_line(pTL_slice[pTEwake], pTL_tip[pupRight], lineTag+1)
    lineTag = lineTag+1
    line_tipConnectionToUpRight = lineTag

    gmsh.model.geo.add_line(pTL_slice[pTE], pTL_tip[pup], lineTag+1)
    lineTag = lineTag+1
    line_tipConnectionToUp = lineTag

### creating the oblique/transverse lines ###
# airfoil skin
line_TEuTipU = lineTag+1
for i in range(gridPts_alongNACA-1):
    if not(bluntTrailingEdge is False and i==0): # to avoid creating a 0 distance line when TE is sharp
        gmsh.model.geo.add_line(pTL_slice[pTEu]+i, pTL_tip[pTEu]+i, lineTag+1) 
    lineTag = lineTag+1
line_LEtipU = lineTag
line_LEtipL = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pLE]+1+i, pTL_tip[pLE]-1-i, lineTag+1)
    lineTag = lineTag+1
line_TEuTipL = lineTag

# BL skin
line_upTipU = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pup]+i, pTL_tip[pup]+i, lineTag+1)
    lineTag = lineTag+1
line_leftTipU = lineTag
line_leftTipL = lineTag+1
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.add_line(pTL_slice[pleft]+1+i, pTL_tip[pleft]-1-i, lineTag+1)
    lineTag = lineTag+1
line_upTipL = lineTag

gmsh.model.geo.add_line(pTL_slice[pupRight], pTL_tip[pupRight], lineTag+1)
lineTag = lineTag+1
line_upRightTipU = lineTag
gmsh.model.geo.add_line(pTL_slice[plowRight], pTL_tip[pupRight], lineTag+1)
lineTag = lineTag+1
line_upRightTipL = lineTag

if bluntTrailingEdge:
    gmsh.model.geo.add_line(pTL_slice[pupMidRight], pTL_tip[pupMidRight], lineTag+1)
    lineTag = lineTag+1
    line_upMidRightTipU = lineTag

    gmsh.model.geo.add_line(pTL_slice[plowMidRight], pTL_tip[pupMidRight], lineTag+1)
    lineTag = lineTag+1
    line_upMidRightTipL = lineTag

    gmsh.model.geo.add_line(pTL_slice[pTE], pTL_tip[pTEu], lineTag+1)
    lineTag = lineTag+1
    line_TEuTipM = lineTag

    gmsh.model.geo.add_line(pTL_slice[pTEwake], pTL_tip[pupMidRight], lineTag+1)
    lineTag = lineTag+1
    line_upMidRightTipM = lineTag


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the surfaces # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# generate transfinite curves
for i in range(gridPts_alongNACA-1):
    gmsh.model.geo.mesh.setTransfiniteCurve(line_TEuTipU+i, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_LEtipL+i, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_upTipU+i, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_leftTipL+i, gridPts_tipSide)
gmsh.model.geo.mesh.setTransfiniteCurve(line_upRightTipU, gridPts_tipSide)
gmsh.model.geo.mesh.setTransfiniteCurve(line_upRightTipL, gridPts_tipSide)
gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToLE, 2)
gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToLeft, 2)

if bluntTrailingEdge:
    gmsh.model.geo.mesh.setTransfiniteCurve(line_upMidRightTipU, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_upMidRightTipL, gridPts_tipSide)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_TEuTipM, gridPts_inTE)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_upMidRightTipM, gridPts_inTE)

else:
    gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToTEalongAirfoil, 2)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToUp, gridPts_inBL, "Progression", gridGeomProg_inBL)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_tipConnectionToUpRight, gridPts_inBL)

# connecting together the new generting airfoil skeleton to the last propeller slice
# LE:
gmsh.model.geo.add_curve_loop([-line_tipConnectionToLE, -lTL_slice[lG], line_tipConnectionToLeft, (lTL_tip[lG]-1)], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_tipLEconnectionStructGridUp = surfaceTag
# TE:
if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([line_TEuTipM, lTL_tip[lM], -line_upMidRightTipM, -lTL_slice[lK]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEconnectionTEpatchMidUp = surfaceTag
else:
    gmsh.model.geo.add_curve_loop([line_tipConnectionToTEalongAirfoil, lTL_tip[lBLrad][1], -lTL_tip[lBLup][0], -line_tipConnectionToUp], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEconnectionStructGridUp = surfaceTag

    gmsh.model.geo.add_curve_loop([-line_tipConnectionToUp, lTL_slice[lK], line_tipConnectionToUpRight, lTL_tip[lD]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEpatchconnectionStructGridUp = surfaceTag

### airfoil skin ###
# airfoil tipSkin uper side
airfoilStructStartSurfTag_tipU = surfaceTag+1
if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][0], line_TEuTipU, -lTL_slice[lairfoilUp][0], -(line_TEuTipU+1)], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
else:
    gmsh.model.geo.add_curve_loop([-lTL_slice[lairfoilUp][0], -(line_TEuTipU+1), line_tipConnectionToTEalongAirfoil], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1

for i in range(1,gridPts_alongNACA-2):
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][i], line_TEuTipU+i, -lTL_slice[lairfoilUp][i], -(line_TEuTipU+i+1)], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
gmsh.model.geo.add_curve_loop([line_tipConnectionToLE, (line_TEuTipU+gridPts_alongNACA-2), -lTL_slice[lairfoilUp][gridPts_alongNACA-2]], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
airfoilStructEndSurfTag_tipU = surfaceTag

# airfoil tipSkin lower side
airfoilStructStartSurfTag_tipL = surfaceTag+1
gmsh.model.geo.add_curve_loop([-line_tipConnectionToLE, -lTL_slice[lairfoilLow][0], -line_LEtipL], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
for i in range(1,gridPts_alongNACA-2):
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][-i-1], -(line_LEtipL+i-1), lTL_slice[lairfoilLow][i], (line_LEtipL+i)], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][-gridPts_alongNACA+1], -(line_LEtipL+gridPts_alongNACA-3), lTL_slice[lairfoilLow][gridPts_alongNACA-2], (line_LEtipL+gridPts_alongNACA-2)], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
else:
    gmsh.model.geo.add_curve_loop([-line_tipConnectionToTEalongAirfoil, -lTL_slice[lairfoilLow][-1], (line_LEtipL+gridPts_alongNACA-3)], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
airfoilStructEndSurfTag_tipL = surfaceTag

### BL skin ###
# airfoil tipSkin uper side
BLstructStartSurfTag_tipU = surfaceTag+1
for i in range(gridPts_alongNACA-2):
    gmsh.model.geo.add_curve_loop([line_upTipU+i, lTL_tip[lBLup][i], -(line_upTipU+1+i), -lTL_slice[lBLup][i]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
# handling the two triangular surfaces appearing at the LE   
gmsh.model.geo.add_curve_loop([lTL_slice[lBLup][gridPts_alongNACA-2], -line_leftTipU, -line_tipConnectionToLeft], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
BLstructEndSurfTag_tipU = surfaceTag
BLstructStartSurfTag_tipL = surfaceTag +1
gmsh.model.geo.add_curve_loop([lTL_slice[lBLlow][0], line_leftTipL, line_tipConnectionToLeft], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
# airfoil tipSkin lower side
for i in range(1,gridPts_alongNACA-1):
    gmsh.model.geo.add_curve_loop([line_leftTipL+i-1, -lTL_tip[lBLup][-i-1], -(line_leftTipL+i), -lTL_slice[lBLlow][i]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
BLstructEndSurfTag_tipL = surfaceTag

### TE patch ###
gmsh.model.geo.add_curve_loop([-lTL_slice[lD], line_upRightTipU, lTL_tip[lD], -line_upTipU], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
lDsurf_tipU = surfaceTag

gmsh.model.geo.add_curve_loop([lTL_slice[lC], line_upRightTipL, lTL_tip[lD], -line_upTipL], surfaceTag+1)
gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
lDsurf_tipL = surfaceTag

if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([line_TEuTipU, lTL_tip[lM], -line_upMidRightTipU, -lTL_slice[lM]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    lMsurf_tipU = surfaceTag

    gmsh.model.geo.add_curve_loop([line_TEuTipL, lTL_tip[lM], -line_upMidRightTipL, -lTL_slice[lL]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    lMsurf_tipL = surfaceTag

### Generating transverse surfaces ###
BLstructStartTransverseSurfTag_tipU = surfaceTag+1
if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([line_TEuTipU, lTL_tip[lBLrad][0], -line_upTipU, -lTL_slice[lBLrad][0]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
else:
    gmsh.model.geo.add_curve_loop([line_tipConnectionToUp, -line_upTipU, -lTL_slice[lBLrad][0]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
for i in range(1,gridPts_alongNACA-1):
    gmsh.model.geo.add_curve_loop([line_TEuTipU+i, lTL_tip[lBLrad][i], -(line_upTipU+i), -lTL_slice[lBLrad][i]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
BLstructEndTransverseSurfTag_tipU = surfaceTag
BLstructStartTransverseSurfTag_tipL = surfaceTag +1
for i in range(1,gridPts_alongNACA-1):
    gmsh.model.geo.add_curve_loop([line_LEtipL+i-1, lTL_tip[lBLrad][gridPts_alongNACA-1-i], -(line_leftTipL+i-1), -lTL_slice[lBLrad][gridPts_alongNACA-1+i]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
if bluntTrailingEdge:
    gmsh.model.geo.add_curve_loop([line_LEtipL+gridPts_alongNACA-2, lTL_tip[lBLrad][0], -(line_leftTipL+gridPts_alongNACA-2), -lTL_slice[lBLrad][2*gridPts_alongNACA-2]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
else:
    gmsh.model.geo.add_curve_loop([line_tipConnectionToUp, -line_upTipL, -lTL_slice[lBLrad][-1]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
BLstructEndTransverseSurfTag_tipL = surfaceTag

### TE patch ###
if bluntTrailingEdge:
    # TE patch Mid Up
    gmsh.model.geo.add_curve_loop([-line_TEuTipM, lTL_slice[lEu], line_TEuTipU], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEtransverseConnectionU = surfaceTag
    gmsh.model.geo.add_curve_loop([-line_TEuTipM, -lTL_slice[lEl], line_TEuTipL], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEtransverseConnectionL = surfaceTag
    gmsh.model.geo.add_curve_loop([-line_upMidRightTipM, lTL_slice[lFu], line_upMidRightTipU], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEwakeTransverseConnectionU = surfaceTag
    gmsh.model.geo.add_curve_loop([-line_upMidRightTipM, -lTL_slice[lFl], line_upMidRightTipL], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEwakeTransverseConnectionL = surfaceTag

    # TE patch Up
    gmsh.model.geo.add_curve_loop([line_upMidRightTipU, lTL_tip[lAr], -line_upRightTipU, -lTL_slice[lAr]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEpatchUpTransverseConnectionU = surfaceTag
    gmsh.model.geo.add_curve_loop([line_upMidRightTipL, lTL_tip[lAr], -line_upRightTipL, lTL_slice[lBr]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEpatchUpTransverseConnectionL = surfaceTag
else:
    gmsh.model.geo.add_curve_loop([-line_tipConnectionToUpRight, line_upRightTipU, lTL_slice[lAr]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEpatchUpTransverseConnectionU = surfaceTag
    gmsh.model.geo.add_curve_loop([-line_tipConnectionToUpRight, line_upRightTipL, -lTL_slice[lBr]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipTEpatchUpTransverseConnectionL = surfaceTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the volumes # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$

### tip Struct BL ###

tipStructBLstartVolumeTag = volumeTag+1
if bluntTrailingEdge:
    gmsh.model.geo.addSurfaceLoop([BLstructStartTransverseSurfTag_tipU, BLstructStartTransverseSurfTag_tipU+1, BLstructStartSurfTag_tipU, airfoilStructStartSurfTag_tipU, sTL_slice[sBLstructGrid][0], sTL_tip[sBLstructGrid][0]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
else:
    gmsh.model.geo.addSurfaceLoop([BLstructStartTransverseSurfTag_tipU, BLstructStartTransverseSurfTag_tipU+1, BLstructStartSurfTag_tipU, airfoilStructStartSurfTag_tipU, sTL_slice[sBLstructGrid][0], surf_tipTEconnectionStructGridUp], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    # gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) ## not working... By spliting the volume into a pyramid with square basis and a prism with triangular basis, still not working!
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1

for i in range(1,gridPts_alongNACA-2):
    gmsh.model.geo.addSurfaceLoop([BLstructStartTransverseSurfTag_tipU+i, BLstructStartTransverseSurfTag_tipU+1+i, BLstructStartSurfTag_tipU+i, airfoilStructStartSurfTag_tipU+i, sTL_slice[sBLstructGrid][i], sTL_tip[sBLstructGrid][i]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1

gmsh.model.geo.addSurfaceLoop([sTL_slice[sBLstructGrid][gridPts_alongNACA-2], surf_tipLEconnectionStructGridUp, BLstructStartTransverseSurfTag_tipU+gridPts_alongNACA-2, airfoilStructEndSurfTag_tipU, BLstructEndSurfTag_tipU], volumeTag+1)
gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) # gridPts_tipSide needs to be > 2 otherwise this transfinite operation fails !
gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
volumeTag = volumeTag+1
gmsh.model.geo.addSurfaceLoop([sTL_slice[sBLstructGrid][gridPts_alongNACA-1], surf_tipLEconnectionStructGridUp, BLstructStartTransverseSurfTag_tipU+gridPts_alongNACA-1, airfoilStructStartSurfTag_tipL, BLstructStartSurfTag_tipL], volumeTag+1)
gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) # gridPts_tipSide needs to be > 2 otherwise this transfinite operation fails !
gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
volumeTag = volumeTag+1

for i in range(1,gridPts_alongNACA-2):
    gmsh.model.geo.addSurfaceLoop([BLstructStartSurfTag_tipL+i, BLstructStartTransverseSurfTag_tipL+i-1, BLstructStartTransverseSurfTag_tipL+i, airfoilStructStartSurfTag_tipL+i, sTL_slice[sBLstructGrid][gridPts_alongNACA+i-1], sTL_tip[sBLstructGrid][gridPts_alongNACA-i-2]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1

if bluntTrailingEdge:
    gmsh.model.geo.addSurfaceLoop([BLstructEndTransverseSurfTag_tipL, BLstructEndTransverseSurfTag_tipL-1, BLstructEndSurfTag_tipL, airfoilStructEndSurfTag_tipL, sTL_slice[sBLstructGrid][-1], sTL_tip[sBLstructGrid][0]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
else:
    gmsh.model.geo.addSurfaceLoop([BLstructEndTransverseSurfTag_tipL, BLstructEndTransverseSurfTag_tipL-1, BLstructEndSurfTag_tipL, airfoilStructEndSurfTag_tipL, sTL_slice[sBLstructGrid][-1], surf_tipTEconnectionStructGridUp], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    # gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) ## not working... By spliting the volume into a pyramid with square basis and a prism with triangular basis, still not working!
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
tipStructBLendVolumeTag = volumeTag

tipStructBL = list(range(tipStructBLstartVolumeTag, tipStructBLendVolumeTag+1))

### tip TE patch ###

tipTEpatchStartVolumeTag = volumeTag+1
if bluntTrailingEdge:
    gmsh.model.geo.addSurfaceLoop([surf_tipTEconnectionTEpatchMidUp, surf_tipTEtransverseConnectionU, surf_tipTEwakeTransverseConnectionU, lMsurf_tipU, sTL_slice[sTEpatchMidUp]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
    gmsh.model.geo.addSurfaceLoop([surf_tipTEconnectionTEpatchMidUp, surf_tipTEtransverseConnectionL, surf_tipTEwakeTransverseConnectionL, lMsurf_tipL, sTL_slice[sTEpatchMidLow]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
    gmsh.model.geo.addSurfaceLoop([sTL_tip[sTEpatchUp], sTL_slice[sTEpatchUp], BLstructStartTransverseSurfTag_tipU, surf_tipTEpatchUpTransverseConnectionU, lDsurf_tipU, lMsurf_tipU], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
    gmsh.model.geo.addSurfaceLoop([sTL_tip[sTEpatchUp], sTL_slice[sTEpatchLow], BLstructEndTransverseSurfTag_tipL, surf_tipTEpatchUpTransverseConnectionL, lDsurf_tipL, lMsurf_tipL], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
else:
    gmsh.model.geo.addSurfaceLoop([lDsurf_tipU, sTL_slice[sTEpatchUp], surf_tipTEpatchUpTransverseConnectionU, surf_tipTEpatchconnectionStructGridUp, BLstructStartTransverseSurfTag_tipU], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
    gmsh.model.geo.addSurfaceLoop([lDsurf_tipL, sTL_slice[sTEpatchLow], surf_tipTEpatchUpTransverseConnectionL, surf_tipTEpatchconnectionStructGridUp, BLstructEndTransverseSurfTag_tipL], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
tipTEpatchEndVolumeTag = volumeTag

tipTEpatch = list(range(tipTEpatchStartVolumeTag, tipTEpatchEndVolumeTag+1))
    
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# 2D pavement
# gmsh.option.setNumber("Mesh.Smoothing", 3)
# gmsh.option.setNumber("Mesh.Algorithm", 11) # mesh 2D
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.generate()

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

# gmsh.model.addPhysicalGroup(pb_2Dim, [*structGridSurf], 1, "CFD") # physical surface
gmsh.model.addPhysicalGroup(pb_3Dim, [*tipStructBL, *tipTEpatch], 1, "CFD") # physical surface


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