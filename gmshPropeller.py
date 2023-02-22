# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

import sys
import gmsh
from gmshToolkit import *
import shutil 

NACA_type = '4412'

geometry_file = "SP2_geom" # "VP1304_geom" , "SP2_geom" from https://doi.org/10.1063/5.0098891

bluntTrailingEdge = False
optimisedGridSpacing = False

gridPts_alongNACA = 4

gridPts_inBL = 5 # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.1

TEpatchGridFlaringAngle = 0 # deg
gridPts_alongTEpatch = 4 # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.05

wakeGridFlaringAngle = 0 # deg
gridPts_alongWake = 5 # > 2 for split into fully hex mesh
gridGeomProg_alongWake = 1.0

[radii_vec, chord_vec, pitch_vecAngle, rake_vec, skew_vecAngle] = read_geometry(geometry_file+".dat") # reads geometry and defines tip truncation

skew_vec = np.sin(skew_vecAngle*np.pi/180)*radii_vec
# "SP2_geom.dat" considers the total rake. rake_vec = generatorRake_vec + skew_vec*np.tan(pitch_vecAngle*np.pi/180)

# # conversion from (m) to (mm)
# radii_vec = 1000*radii_vec
# chord_vec = 1000*chord_vec
# rake_vec = 1000*rake_vec
# skew_vec = 1000*skew_vec

airfoilReferenceCoordinate = np.array([skew_vec, -rake_vec, -radii_vec]).transpose()

# # for dummy geom, uncomment below:
# radii_vec = [0.1, 0.8, 1.5]
# pitch_vecAngle = [20.0, 30.0, 35.0]

radii_step = [1] * len(radii_vec) # number of radial elements between to radial slices. 

airfoilReferenceAlongChord_c = 0.5
TEpatchLength_c = 0.1 # length of the TEpatch in along the x-axis
wakeLength_c = 0.3 # length of the wake in along the x-axis
height_LE_c = 0.1 # Structured Grid offset layer gap at the leading edge
height_TE_c = 0.2 # Structured Grid offset layer gap at the trailing edge
gridPts_inTE = int(gridPts_inBL/4) # if the TE is blunt, number of cells in the TE half height. NB: for the Blossom algorithm to work an even number of faces must be given.
airfoilReferenceAlongChord_c = 0.5

# Initialize gmsh:
gmsh.initialize()

##  to dig to enable multi-threading:
# gmsh.option.setNumber("General.NumThreads",4) 
# print("general nt: ", gmsh.option.getNumber("General.NumThreads"))
# # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1436 
# # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1093 
# # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1422


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

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the propeller mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Creation of blade number 1 ***   ***   ***
rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree around [axisZ, axisY, axisX]
shiftVec = [0.0, 0.0, 0.0]
for i in range(len(radii_vec)):

    structTag = [pointTag, lineTag, surfaceTag]
    GeomSpec = [NACA_type, bluntTrailingEdge, optimisedGridSpacing, pitch_vecAngle[i], chord_vec[i], airfoilReferenceAlongChord_c*chord_vec[i], airfoilReferenceCoordinate[i], height_LE_c*chord_vec[i], height_TE_c*chord_vec[i], TEpatchLength_c*chord_vec[i]*np.cos(pitch_vecAngle[i]*np.pi/180), TEpatchGridFlaringAngle, wakeLength_c*chord_vec[i]*np.cos(pitch_vecAngle[i]*np.pi/180), wakeGridFlaringAngle]
    GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]

    if i == len(radii_vec)-1:
        GeomSpec[0] = '0012' # force 'NACA_type' to be 0012 for the last slice
    [pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)
    
    pTS1.append(pointTag_list)
    lTS1.append(lineTag_list)
    sTS1.append(surfaceTag_list)

[tlTS1, lineTag] = gmeshed_blade_tl(pTS1, gridPts_alongNACA, radii_step, bluntTrailingEdge, lineTag)

[tsTS1, surfaceTag] = gmeshed_blade_ts(lTS1, tlTS1, gridPts_alongNACA, radii_step, bluntTrailingEdge, surfaceTag)

[tsTS_tip1, pointTag, lineTag, surfaceTag] = gmeshed_bladeTip_ts(pTS1[-1], lTS1[-1], GeomSpec, GridPtsSpec, rotMat, shiftVec, pointTag, lineTag, surfaceTag)

[sStructGridSkin1, sairfoilSkin1] = returnStructGridOuterShell(sTS1, tsTS1, tsTS_tip1, radii_step, bluntTrailingEdge)


# Creation of blade number 2 ***   ***   ***
rotMat = rotationMatrix([180.0, 0.0, 180.0]) # angles in degree around [axisZ, axisY, axisX]

for i in range(len(radii_vec)):

    structTag = [pointTag, lineTag, surfaceTag]
    GeomSpec = [NACA_type, bluntTrailingEdge, optimisedGridSpacing, pitch_vecAngle[i], chord_vec[i], airfoilReferenceAlongChord_c*chord_vec[i], airfoilReferenceCoordinate[i], height_LE_c*chord_vec[i], height_TE_c*chord_vec[i], TEpatchLength_c*chord_vec[i]*np.cos(pitch_vecAngle[i]*np.pi/180), TEpatchGridFlaringAngle, wakeLength_c*chord_vec[i]*np.cos(pitch_vecAngle[i]*np.pi/180), wakeGridFlaringAngle]
    GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]

    if i == len(radii_vec)-1:
        GeomSpec[0] = '0012' # force 'NACA_type' to be 0012 for the last slice
    [pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)
    
    pTS2.append(pointTag_list)
    lTS2.append(lineTag_list)
    sTS2.append(surfaceTag_list)

[tlTS2, lineTag] = gmeshed_blade_tl(pTS2, gridPts_alongNACA, radii_step, bluntTrailingEdge, lineTag)

[tsTS2, surfaceTag] = gmeshed_blade_ts(lTS2, tlTS2, gridPts_alongNACA, radii_step, bluntTrailingEdge, surfaceTag)

[tsTS_tip2, pointTag, lineTag, surfaceTag] = gmeshed_bladeTip_ts(pTS2[-1], lTS2[-1], GeomSpec, GridPtsSpec, rotMat, shiftVec, pointTag, lineTag, surfaceTag)

[sStructGridSkin2, sairfoilSkin2] = returnStructGridOuterShell(sTS2, tsTS2, tsTS_tip2, radii_step, bluntTrailingEdge)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the transfinite blade volumes # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

[vTS1, volumeTag] = gmeshed_blade_vol(sTS1, tsTS1, gridPts_alongNACA, radii_step, bluntTrailingEdge, volumeTag)
[vTS_tip1, volumeTag] = gmeshed_bladeTip_vol(sTS1[-1], tsTS_tip1, gridPts_alongNACA, bluntTrailingEdge, volumeTag)

[vTS2, volumeTag] = gmeshed_blade_vol(sTS2, tsTS2, gridPts_alongNACA, radii_step, bluntTrailingEdge, volumeTag)
[vTS_tip2, volumeTag] = gmeshed_bladeTip_vol(sTS2[-1], tsTS_tip2, gridPts_alongNACA, bluntTrailingEdge, volumeTag)

vol_blade_1 = returnStructGridVol(vTS1, vTS_tip1, bluntTrailingEdge)
vol_blade_2 = returnStructGridVol(vTS2,vTS_tip2,  bluntTrailingEdge)

## below to generate a blade propeller without tip correction: 
# [sStructGridSkin1, sairfoilSkin1] = returnStructGridOuterShell_withoutTip(sTS1, tsTS1, radii_step, bluntTrailingEdge)
# vol_blade_1 = returnStructGridVol_withoutTip(vTS1, bluntTrailingEdge)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the cylindrical volumes # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

if geometry_file == "VP1304_geom" :
    y_max_cyl1 = 0.15
    y_min_cyl1 = -0.15
    r_cyl1 = 0.3
    elemSize_cyl1 = 0.01

    y_max_cyl2 = 0.3
    y_min_cyl2 = -0.3
    r_cyl2 = 0.5
    elemSize_cyl2 = 0.02

elif geometry_file == "SP2_geom":
    y_max_cyl1 = 0.02
    y_min_cyl1 = -0.03
    r_cyl1 = 0.125
    elemSize_cyl1 = 0.0025

    y_max_cyl2 = 0.075
    y_min_cyl2 = -0.15
    r_cyl2 = 0.2
    elemSize_cyl2 = 0.01

[ cylSurf1, pointTag, lineTag, surfaceTag] = gmeshed_cylinder_surf(y_min_cyl1, y_max_cyl1, r_cyl1, elemSize_cyl1, pointTag, lineTag, surfaceTag)
[ cylSurf2, pointTag, lineTag, surfaceTag] = gmeshed_cylinder_surf(y_min_cyl2, y_max_cyl2, r_cyl2, elemSize_cyl2, pointTag, lineTag, surfaceTag)

gmsh.model.geo.addSurfaceLoop([*sStructGridSkin1, *sStructGridSkin2, *cylSurf1], volumeTag+1)
gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
volumeTag = volumeTag+1
vol_unstructCFD = volumeTag

gmsh.model.geo.addSurfaceLoop([*cylSurf1, *cylSurf2], volumeTag+1)
gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
volumeTag = volumeTag+1
vol_unstructBUFF = volumeTag


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# 2D pavement
# gmsh.option.setNumber("Mesh.Smoothing", 3)
# gmsh.option.setNumber("Mesh.Algorithm", 11) # mesh 2D
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
# gmsh.model.mesh.generate(2)

# generating a high quality fully hex mesh is a tall order: 
# https://gitlab.onelab.info/gmsh/gmsh/-/issues/784

# gmsh.option.setNumber('Mesh.Recombine3DLevel', 0)
# gmsh.option.setNumber("Mesh.NbTetrahedra", 0)
# gmsh.option.setNumber("Mesh.Algorithm3D", 4) # mesh 3D

gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2) # most robust way to obtain pure hex mesh: subdivise it
# gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 3) # perhaps better but conflict with transfinite mesh... to dig further

gmsh.model.mesh.generate(3)

# gmsh.model.mesh.recombine()
# gmsh.model.mesh.refine()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the physical group # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()
[nodePerEntity, elemPerEntity] = countDOF()

gmsh.model.addPhysicalGroup(pb_3Dim, [*vol_blade_1, *vol_blade_2, vol_unstructCFD], 1, "Propeller Grid")
gmsh.model.addPhysicalGroup(pb_3Dim, [vol_unstructBUFF], 2, "Outer Grid")

# export volume mesh only for visualisation:
gmsh.write(geometry_file+"_NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.vtk")

gmsh.model.addPhysicalGroup(pb_2Dim, [*sairfoilSkin1], 1, "Blade 1 Hard Wall")
gmsh.model.addPhysicalGroup(pb_2Dim, [*sairfoilSkin2], 2, "Blade 2 Hard Wall")

gmsh.model.addPhysicalGroup(pb_2Dim, [cylSurf1[0], cylSurf1[1], cylSurf1[2], cylSurf1[3]], 3, "Inner Cylinder Side")
gmsh.model.addPhysicalGroup(pb_2Dim, [cylSurf1[4]], 4, "Inner Cylinder Top")
gmsh.model.addPhysicalGroup(pb_2Dim, [cylSurf1[5]], 5, "Inner Cylinder Bottom")

gmsh.model.addPhysicalGroup(pb_2Dim, [cylSurf2[0], cylSurf2[1], cylSurf2[2], cylSurf2[3]], 6, "Outer Cylinder Side")
gmsh.model.addPhysicalGroup(pb_2Dim, [cylSurf2[4]], 7, "Inner Cylinder Top")
gmsh.model.addPhysicalGroup(pb_2Dim, [cylSurf2[5]], 8, "Inner Cylinder Bottom")


# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

# export mesh with all tags for computation:
gmsh.write(geometry_file+"_NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.msh")


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