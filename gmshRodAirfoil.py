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
optimisedGridSpacing = False

gridPts_alongNACA = 80

gridPts_alongSpan = 10

gridPts_inBL = 25 # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.1

TEpatchGridFlaringAngle = 0 # deg
gridPts_alongTEpatch = 10 # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.05

wakeGridFlaringAngle = 0 # deg
gridPts_alongWake = 20 # > 2 for split into fully hex mesh
gridGeomProg_alongWake = 1.0

pitch = 20.0 # deg
chord = 0.2 # m 
span = 0.75*chord # m


# Initialize gmsh:
gmsh.initialize()

pTS_airfoil = [] # pointTag_struct -- airfoil
lTS_airfoil = [] # lineTag_struct -- airfoil
sTS_airfoil = [] # surfaceTag_struct -- airfoil

pTS_rod = [] # pointTag_struct -- rod
lTS_rod = [] # lineTag_struct -- rod
sTS_rod = [] # surfaceTag_struct -- rod

pTS_frame = [] # pointTag_struct -- CFD+BUFF
lTS_frame = [] # lineTag_struct -- CFD+BUFF
sTS_frame = [] # surfaceTag_struct -- CFD+BUFF

pointTag = 0
lineTag = 0
surfaceTag = 0
volumeTag = 0

rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree around [axisZ, axisY, axisX]
shiftVec = np.array([0.0, 0.0, 0.0]) # shift of the origin

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

structTag = [pointTag, lineTag, surfaceTag]
GeomSpec = [NACA_type, bluntTrailingEdge, optimisedGridSpacing, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
[pTL_airfoil, lTL_airfoil, sTL_airfoil, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec) 

bladeLine = returnStructGridOuterContour(lTL_airfoil, bluntTrailingEdge)
structGridSurf = returnStructGridSide(sTL_airfoil, bluntTrailingEdge)

# $$$$$$$$$$$$$$$$$$$$$
# # Creation of rod # #
# $$$$$$$$$$$$$$$$$$$$$

rodPos = [-2.0*chord, 0.0, 0.0]
rodR = 0.1*chord
rodElemSize = 0.02*chord
rodBLwidth = 0.05*chord

gridPts_alongRod = int(2*np.pi*rodR/rodElemSize/4)
gridPts_inRodBL = 25
gridGeomProg_inRodBL = 1.1

structTag = [pointTag, lineTag, surfaceTag]
RodGeomSpec = [rodPos, rodR, rodBLwidth]
RodGridPtsSpec = [gridPts_alongRod, gridPts_inRodBL, gridGeomProg_inRodBL]
[pTL_rod, lTL_rod, sTL_rod, pointTag, lineTag, surfaceTag] = gmeshed_disk(structTag, RodGeomSpec, RodGridPtsSpec, rotMat, shiftVec)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the exterior region # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

pRodCenter = 0
pRod = 1
pRodBL = 2
lRodConn = 0
lRodArc = 1
lRodBL = 2

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

[rectLine, pointTag, lineTag] = gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag, rotMat, shiftVec)
[rectLineBUFF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minBUFF, x_maxBUFF, y_minBUFF, y_maxBUFF, elemSize_rectBUFF, pointTag, lineTag, rotMat, shiftVec)

gmsh.model.geo.add_curve_loop( [*rectLine, *bladeLine, *lTL_rod[lRodBL]], surfaceTag+1) 
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

### Extrude rod BL
rodBLdoublet = []
for elem in sTL_rod:
    rodBLdoublet.append((pb_2Dim,elem))
ExtrudRodBL = gmsh.model.geo.extrude(rodBLdoublet, 0, 0, span, [gridPts_alongSpan], recombine=True)

# Extract volume tags of rod BL
ExtrudRodBL_vol = []
for i in range(len(ExtrudRodBL)):
    if ExtrudRodBL[i][0] == 3:  
        ExtrudRodBL_vol.append(ExtrudRodBL[i][1])

# Extract surface tags of rod BL
ExtrudRodBL_skin = []
ExtrudRodBL_symFace = []
# ExtrudRodBL_rodBLouterSkin = []
for i in range(len(ExtrudRodBL)):
    if ExtrudRodBL[i][0] == 3:  
        ExtrudRodBL_symFace.append(ExtrudRodBL[i-1][1]) # Rod BL extruded periodic face
        ExtrudRodBL_skin.append(ExtrudRodBL[i+2][1]) # Rod cylinder skin
        # ExtrudRodBL_rodBLouterSkin.append(ExtrudRodBL[i+4][1]) # Rod BL connection to unstruct CFD mesh

### Extrude struct Airfoil
airfoilStructDoublet = []
for elem in sTL_airfoil:
    if isinstance(elem, int): 
        if not(elem==sTL_airfoil[sairfoil]) and not(elem==-1): # in order not to extrude the airfoil interior and the empty tags
            airfoilStructDoublet.append((pb_2Dim,elem))
    else:
        if not(elem==sTL_airfoil[sBLstructGrid]): # in order not to generate twice the elemenst of the unstruct BL grid 
            for subElem in elem:
                airfoilStructDoublet.append((pb_2Dim,subElem))
ExtrudAirfoildStruct = gmsh.model.geo.extrude(airfoilStructDoublet, 0, 0, span, [gridPts_alongSpan], recombine=True)

# Extract volume tags of struct Airfoil
ExtrudAirfoildStruct_vol = []
for i in range(len(ExtrudAirfoildStruct)):
    if ExtrudAirfoildStruct[i][0] == 3:  
        ExtrudAirfoildStruct_vol.append(ExtrudAirfoildStruct[i][1])

# Extract surface tags of struct Airfoil
ExtrudStructAirfoil_skin = []
ExtrudStructAirfoil_symFace = []
for i in range(len(ExtrudAirfoildStruct)):
    if ExtrudAirfoildStruct[i][0] == 3:  
        ExtrudStructAirfoil_symFace.append(ExtrudAirfoildStruct[i-1][1]) # Struct Airfoil extruded periodic face
for i in range((2*gridPts_alongNACA-2)*6):
    if ExtrudAirfoildStruct[i][0] == 3:  
        ExtrudStructAirfoil_skin.append(ExtrudAirfoildStruct[i+3][1]) # Airfoil skin
if bluntTrailingEdge:
    ExtrudStructAirfoil_skin.append(ExtrudAirfoildStruct[(2*gridPts_alongNACA-2)*6+17][1]) # Addition of the trailing edge surfaces
    ExtrudStructAirfoil_skin.append(ExtrudAirfoildStruct[(2*gridPts_alongNACA-2)*6+23][1]) # Addition of the trailing edge surfaces

### Extrude unstructCFD domain
ExtrudUnstructCFD = gmsh.model.geo.extrude([(pb_2Dim, surf_unstructCFD)], 0, 0, span, [gridPts_alongSpan], recombine=True)

# Extract volume tags of unstruct CFD
ExtrudUnstructCFD_vol = []
for i in range(len(ExtrudUnstructCFD)):
    if ExtrudUnstructCFD[i][0] == 3:  
        ExtrudUnstructCFD_vol.append(ExtrudUnstructCFD[i][1])

# Extract surface tags of unstruct CFD
ExtrudUnstructCFD_symFace = []
ExtrudUnstructCFD_symFace.append(ExtrudUnstructCFD[0][1]) # unstruct CFD extruded periodic face
# ExtrudUnstructCFD_outerSkin = []
# ExtrudUnstructCFD_outerSkin.append(ExtrudUnstructCFD[2][1]) # untruct CFD junction with BUFF
# ExtrudUnstructCFD_outerSkin.append(ExtrudUnstructCFD[3][1]) # untruct CFD junction with BUFF
# ExtrudUnstructCFD_outerSkin.append(ExtrudUnstructCFD[4][1]) # untruct CFD junction with BUFF
# ExtrudUnstructCFD_outerSkin.append(ExtrudUnstructCFD[5][1]) # untruct CFD junction with BUFF

### Extrude unstructBUFF domain
ExtrudUnstructBUFF = gmsh.model.geo.extrude([(pb_2Dim, surf_unstructBUFF)], 0, 0, span, [gridPts_alongSpan], recombine=True)

# Extract volume tags of unstruct BUFF
ExtrudUnstructBUFF_vol = []
for i in range(len(ExtrudUnstructBUFF)):
    if ExtrudUnstructBUFF[i][0] == 3:  
        ExtrudUnstructBUFF_vol.append(ExtrudUnstructBUFF[i][1])

# Extract surface tags of unstruct BUFF
ExtrudUnstructBUFF_symFace = []
ExtrudUnstructBUFF_symFace.append(ExtrudUnstructBUFF[0][1]) # unstruct CFD extruded periodic face
ExtrudUnstructBUFF_innerSkin = []
ExtrudUnstructBUFF_innerSkin.append(ExtrudUnstructBUFF[2][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_innerSkin.append(ExtrudUnstructBUFF[3][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_innerSkin.append(ExtrudUnstructBUFF[4][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_innerSkin.append(ExtrudUnstructBUFF[5][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_bottom = []
ExtrudUnstructBUFF_bottom.append(ExtrudUnstructBUFF[6][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_outlet = []
ExtrudUnstructBUFF_outlet.append(ExtrudUnstructBUFF[7][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_top = []
ExtrudUnstructBUFF_top.append(ExtrudUnstructBUFF[8][1]) # untruct CFD junction with BUFF
ExtrudUnstructBUFF_inlet = []
ExtrudUnstructBUFF_inlet.append(ExtrudUnstructBUFF[9][1]) # untruct CFD junction with BUFF


volMesh = [*ExtrudRodBL_vol, *ExtrudAirfoildStruct_vol, *ExtrudUnstructCFD_vol, *ExtrudUnstructBUFF_vol]

surfMesh_original = [*sTL_rod, *structGridSurf, surf_unstructCFD, surf_unstructBUFF ]
surfMesh_symFace = [*ExtrudRodBL_symFace, *ExtrudStructAirfoil_symFace, *ExtrudUnstructCFD_symFace, *ExtrudUnstructBUFF_symFace]
surfMesh_rodHardWall = [*ExtrudRodBL_skin]
surfMesh_airfoilHardWall = [*ExtrudStructAirfoil_skin]

# [*ExtrudUnstructBUFF_innerSkin, *ExtrudUnstructBUFF_outerSkin]


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


gmsh.model.addPhysicalGroup(pb_3Dim, [*volMesh], 1, "CFD Grid") # physical volume

gmsh.model.addPhysicalGroup(pb_2Dim, [*surfMesh_original], 1, "Periodic BC") # physical volume

gmsh.model.addPhysicalGroup(pb_2Dim, [*surfMesh_rodHardWall], 2, "Rod Hard Wall BC") # physical volume

gmsh.model.addPhysicalGroup(pb_2Dim, [*surfMesh_airfoilHardWall], 3, "Airfoil Hard Wall BC") # physical volume

gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_innerSkin], 4, "BUFF inner BC") # physical volume

gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_inlet], 5, "Inlet BC") # physical volume
gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_outlet], 6, "Outlet BC") # physical volume

gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_bottom], 7, "Bottom Frontier BC") # physical volume
gmsh.model.addPhysicalGroup(pb_2Dim, [*ExtrudUnstructBUFF_top], 8, "Top Frontier BC") # physical volume




[nodePerEntity, elemPerEntity] = countDOF()

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

gmsh.write("Rod_NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.msh")
gmsh.write("Rod_NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.vtk")

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
