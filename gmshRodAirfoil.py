# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

# aims at reproducing the rod-airfoil benchmark, Casalino, Jacob and Roger aiaaj03 DOI: 10.2514/2.1959

import sys
import gmsh
from gmshToolkit import *
import shutil

NACA_type = '0012'
CONF = 'airfoil' # airfoil, rod, rodAirfoil

bluntTrailingEdge = True

gridPtsRichness = 2.0

gridPts_alongNACA = int(75*gridPtsRichness)

gridPts_alongSpan = int(20*gridPtsRichness)

gridPts_inBL = int(15*gridPtsRichness) # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.1

TEpatchGridFlaringAngle = 30 # deg
gridPts_alongTEpatch = int(8*gridPtsRichness) # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.05

wakeGridFlaringAngle = 10 # deg
gridPts_alongWake = int(25*gridPtsRichness) # > 2 for split into fully hex mesh
gridGeomProg_alongWake = 1.0

pitch = 12.0 # deg
chord = 0.2 # m 
span = 0.2*chord # m

# Initialize gmsh:
gmsh.initialize()

pointTag = 0
lineTag = 0
surfaceTag = 0
volumeTag = 0

rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree around [axisZ, axisY, axisX]
shiftVec = np.array([0.0, 0.0, 0.0]) # shift of the origin

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the airfoil mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

if not (CONF == 'rod'):

    airfoilReferenceAlongChord = 0.5*chord
    TEpatchLength = 0.1*chord*np.cos(pitch*np.pi/180) # length of the TEpatch in along the x-axis
    wakeLength = 0.5*chord*np.cos(pitch*np.pi/180) # length of the wake in along the x-axis
    height_LE = 0.05*chord # Structured Grid offset layer gap at the leading edge
    height_TE = 0.1*chord # Structured Grid offset layer gap at the trailing edge
    gridPts_inTE = int(gridPts_inBL/8) # if the TE is blunt, number of cells in the TE half height. NB: for the Blossom algorithm to work an even number of faces must be given.

    airfoilReferenceAlongChord = 0.5*chord
    airfoilReferenceCoordinate = [0.0, 0.0, 0.0]

    structTag = [pointTag, lineTag, surfaceTag]
    GeomSpec = [NACA_type, bluntTrailingEdge, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
    GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
    [pTL_airfoil, lTL_airfoil, sTL_airfoil, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec) 

    bladeLine = returnStructGridOuterContour(lTL_airfoil, bluntTrailingEdge)
    structGridSurf = returnStructGridSide(sTL_airfoil, bluntTrailingEdge)

# $$$$$$$$$$$$$$$$$$$$$
# # Creation of rod # #
# $$$$$$$$$$$$$$$$$$$$$

if not (CONF == 'airfoil'):

    rodPos = [-2.0*chord, 0.0, 0.0]
    rodR = 0.1*chord
    rodElemSize = 0.01*chord/gridPtsRichness
    rodBLwidth = 0.05*chord

    gridPts_alongRod = int(2*np.pi*rodR/rodElemSize/4)
    gridPts_inRodBL = int(10*gridPtsRichness)
    gridGeomProg_inRodBL = 1.1

    structTag = [pointTag, lineTag, surfaceTag]
    RodGeomSpec = [rodPos, rodR, rodBLwidth]
    RodGridPtsSpec = [gridPts_alongRod, gridPts_inRodBL, gridGeomProg_inRodBL]
    [pTL_rod, lTL_rod, sTL_rod, pointTag, lineTag, surfaceTag] = gmeshed_disk(structTag, RodGeomSpec, RodGridPtsSpec, rotMat, shiftVec)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the exterior region # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

x_min = - 1.5*chord
x_max = 3*chord
y_min = - 1.5*chord
y_max = 1.5*chord
elemSize_rect = chord/10/gridPtsRichness

x_minBUFF = - 2*chord
x_maxBUFF = 5*chord
y_minBUFF = - 2*chord
y_maxBUFF = 2*chord
elemSize_rectBUFF = chord/5/gridPtsRichness

[rectLine, pointTag, lineTag] = gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag, rotMat, shiftVec)
[rectLineBUFF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minBUFF, x_maxBUFF, y_minBUFF, y_maxBUFF, elemSize_rectBUFF, pointTag, lineTag, rotMat, shiftVec)

lRodConn = 0
lRodArc = 1
lRodBL = 2

if CONF == 'rodAirfoil':
    unstructCFD_curve = [*rectLine, *bladeLine, *lTL_rod[lRodBL]]
elif CONF == 'airfoil':
    unstructCFD_curve = [*rectLine, *bladeLine]
elif CONF == 'rod':
    unstructCFD_curve = [*rectLine, *lTL_rod[lRodBL]]

gmsh.model.geo.add_curve_loop(unstructCFD_curve, surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstructCFD = surfaceTag

gmsh.model.geo.add_curve_loop( [*rectLine, *rectLineBUFF], surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstructBUFF = surfaceTag

# $$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Extrusion of the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$

if not (CONF == 'airfoil'):
    [ExtrudRodBL_vol, ExtrudRodBL_symFace, ExtrudRodBL_skin] = extrude_rodBL(sTL_rod, span, gridPts_alongSpan)
    surfMesh_rodHardWall = [*ExtrudRodBL_skin]
    
if not (CONF == 'rod'):
    [ExtrudAirfoildStruct_vol, ExtrudStructAirfoil_symFace, ExtrudStructAirfoil_skin] = extrude_airfoilStruct(sTL_airfoil, bluntTrailingEdge, gridPts_alongNACA, span, gridPts_alongSpan)
    surfMesh_airfoilHardWall = [*ExtrudStructAirfoil_skin]

[ExtrudUnstructCFD_vol, ExtrudUnstructCFD_symFace] = extrude_unstructCFD(surf_unstructCFD, span, gridPts_alongSpan)
[ExtrudUnstructBUFF_vol, ExtrudUnstructBUFF_symFace, ExtrudUnstructBUFF_innerSkin, ExtrudUnstructBUFF_outerSkin] = extrude_unstructBUFF(surf_unstructBUFF, span, gridPts_alongSpan)

if CONF == 'rodAirfoil':
    volMesh = [*ExtrudRodBL_vol, *ExtrudAirfoildStruct_vol, *ExtrudUnstructCFD_vol]
    surfMesh_original = [*sTL_rod, *structGridSurf, surf_unstructCFD, surf_unstructBUFF ]
    surfMesh_symFace = [*ExtrudRodBL_symFace, *ExtrudStructAirfoil_symFace, *ExtrudUnstructCFD_symFace, *ExtrudUnstructBUFF_symFace]
elif CONF == 'airfoil':
    volMesh = [*ExtrudAirfoildStruct_vol, *ExtrudUnstructCFD_vol]
    surfMesh_original = [*structGridSurf, surf_unstructCFD, surf_unstructBUFF ]
    surfMesh_symFace = [*ExtrudStructAirfoil_symFace, *ExtrudUnstructCFD_symFace, *ExtrudUnstructBUFF_symFace]
elif CONF == 'rod':
    volMesh = [*ExtrudRodBL_vol, *ExtrudUnstructCFD_vol]
    surfMesh_original = [*sTL_rod, surf_unstructCFD, surf_unstructBUFF ]
    surfMesh_symFace = [*ExtrudRodBL_symFace, *ExtrudUnstructCFD_symFace, *ExtrudUnstructBUFF_symFace]

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Set periodic bounday condition # # 
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# periodicity along z axis at separation of span
gmsh.model.geo.synchronize()
gmsh.model.mesh.setPeriodic(pb_2Dim, [*surfMesh_symFace], [*surfMesh_original], [1,0,0,0, 0,1,0,0, 0,0,1,span, 0,0,0,1])
# from here on, "surfMesh_symFace" and "surfMesh_original" refer to the same elements.

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# 2D pavement
# gmsh.option.setNumber("Mesh.Smoothing", 3)
# gmsh.option.setNumber("Mesh.Algorithm", 11) # mesh 2D
# gmsh.option.setNumber("Mesh.RecombineAll", 1)

gmsh.model.mesh.generate()

# generating a high quality fully hex mesh is a tall order: 
# https://gitlab.onelab.info/gmsh/gmsh/-/issues/784

# gmsh.option.setNumber('Mesh.Recombine3DAll', 0)
# gmsh.option.setNumber('Mesh.Recombine3DLevel', 0)
# gmsh.option.setNumber("Mesh.NbTetrahedra", 0)
# gmsh.option.setNumber("Mesh.Algorithm3D", 4) # mesh 3D

# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2) # most robust way to obtain pure hex mesh: subdivise it
# # gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 3) # perhaps better but conflict with transfinite mesh... to dig further

# gmsh.model.mesh.generate()

# gmsh.model.mesh.refine()
# gmsh.model.mesh.recombine()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the physical group # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()
[nodePerEntity, elemPerEntity] = countDOF()

gmsh.model.addPhysicalGroup(pb_3Dim, [*volMesh], 1, "CFD Grid")
gmsh.model.addPhysicalGroup(pb_3Dim, [*ExtrudUnstructBUFF_vol], 2, "BUFF Grid")

# export volume mesh only for visualisation:
if CONF == 'rod':
    gmsh.write("rod_"+str(sum(elemPerEntity))+"elems.vtk")
else:
    gmsh.write(CONF+"_NACA"+NACA_type+"_"+str(sum(elemPerEntity))+"elems.vtk")

gmsh.model.addPhysicalGroup(pb_2Dim, [*surfMesh_original], 3, "Periodic BC")

if not (CONF == 'airfoil'):
    gmsh.model.addPhysicalGroup(pb_2Dim, [*surfMesh_rodHardWall], 4, "Rod Hard Wall")
if not (CONF == 'rod'):
    gmsh.model.addPhysicalGroup(pb_2Dim, [*surfMesh_airfoilHardWall], 5, "Airfoil Hard Wall")

gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_innerSkin], 6, "BUFF inner Wrap")

ExtrudUnstructBUFF_inlet = ExtrudUnstructBUFF_outerSkin[0]
ExtrudUnstructBUFF_bottom = ExtrudUnstructBUFF_outerSkin[1]
ExtrudUnstructBUFF_outlet = ExtrudUnstructBUFF_outerSkin[2]
ExtrudUnstructBUFF_top = ExtrudUnstructBUFF_outerSkin[3]

gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_inlet], 7, "Inlet BC")
gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_outlet], 8, "Outlet BC")

gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_bottom], 9, "Bottom BC")
gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_top], 10, "Top BC")


# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

# export mesh with all tags for computation:

if CONF == 'rod':
    gmsh.write("rod_"+str(sum(elemPerEntity))+"elems.msh")
else:
    gmsh.write(CONF+"_NACA"+NACA_type+"_"+str(sum(elemPerEntity))+"elems.msh")

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
