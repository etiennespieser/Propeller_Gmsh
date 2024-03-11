# Copyright (c) 2022-2023 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------
# aims at reproducing the rod-airfoil benchmark, Casalino, Jacob and Roger aiaaj03 DOI: 10.2514/2.1959

import sys
import gmsh
from gmshToolkit import *
import shutil 

NACA_type = '0012'
CONF = 'rodAirfoil' # airfoil, rod, rodAirfoil

bluntTrailingEdge = False

gridPts_alongNACA = 30 # "gridPts_alongNACA" pts makes "gridPts_alongNACA-1" elements
                       # Other parameters scale with this one. 
elemOrder = 2 # 8 is max order supported my navier_mfem: github.com/mfem/mfem/issues/3759, 10 is the max order supported by Gmsh
highOrderBLoptim = 4 # 0: none,
                     # 1: optimization, 
                     # 2: elastic+optimization, 
                     # 3: elastic, 
                     # 4: fast curving
                     # by default choose 4. If for small "gridPts_alongNACA", LE curvature fails, try other values.  
                     
gridPts_inBL = int(0.3*gridPts_alongNACA) # > 2 for split into fully hex mesh
gridGeomProg_inBL = 1.15

TEpatchGridFlaringAngle = 30 # deg
gridPts_alongTEpatch = int(13*gridPts_alongNACA/75.0) # > 2 for split into fully hex mesh
gridGeomProg_alongTEpatch = 1.10

wakeGridFlaringAngle = 10 # deg
gridPts_alongWake = int(30*gridPts_alongNACA/75.0) # > 2 for split into fully hex mesh
gridGeomProg_alongWake = 1.0

pitch = 12.0 # deg
chord = 20.0 # m 

# Initialize gmsh:
gmsh.initialize()

pointTag = 0
lineTag = 0
surfaceTag = 0
volumeTag = 0

rotMat = rotationMatrix([0.0, 0.0, 0.0]) # angles in degree
shiftVec = [0.0, 0.0, 0.0] # shift of the airfoil origin

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the airfoil mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

if not (CONF == 'rod'):

    airfoilReferenceAlongChord = 0.5*chord
    TEpatchLength = 0.1*chord*np.cos(pitch*np.pi/180) # length of the TEpatch in along the x-axis
    wakeLength = 0.5*chord*np.cos(pitch*np.pi/180) # length of the wake in along the x-axis
    height_LE = 0.05*chord # Structured Grid offset layer gap at the leading edge
    height_TE = 0.1*chord # Structured Grid offset layer gap at the trailing edge
    gridPts_inTE = int(gridPts_inBL/7) # if the TE is blunt, number of cells in the TE half height. NB: for the Blossom algorithm to work an even number of faces must be given.

    airfoilReferenceAlongChord = 0.5*chord
    airfoilReferenceCoordinate = [0.0, 0.0, 0.0]

    structTag = [pointTag, lineTag, surfaceTag]
    GeomSpec = [NACA_type, bluntTrailingEdge, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
    GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]
    # [pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)
    [pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil_HO(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)

    airfoilLine, airfoilLineSuction, airfoilLinePressure = returnAirfoilContour(lineTag_list, bluntTrailingEdge)
    bladeLine = returnStructGridOuterContour(lineTag_list, bluntTrailingEdge)
    structGridSurf = returnStructGridSide(surfaceTag_list, bluntTrailingEdge)

# $$$$$$$$$$$$$$$$$$$$$
# # Creation of rod # #
# $$$$$$$$$$$$$$$$$$$$$

if not (CONF == 'airfoil'):

    rodPos = [-1.5*chord, 0.0, 0.0]
    rodR = 0.05*chord
    rodElemSize = 0.01*chord/(gridPts_alongNACA/75.0)
    rodBLwidth = 4*rodR

    gridPts_alongRod = int(np.pi*rodR/rodElemSize)
    gridPts_inRodBL = int(gridPts_alongNACA/2.0)
    gridGeomProg_inRodBL = 1.1

    structTag = [pointTag, lineTag, surfaceTag]
    RodGeomSpec = [rodPos, rodR, rodBLwidth]
    RodGridPtsSpec = [gridPts_alongRod, gridPts_inRodBL, gridGeomProg_inRodBL]
    [pTL_rod, lTL_rod, sTL_rod, pointTag, lineTag, surfaceTag] = gmeshed_disk(structTag, RodGeomSpec, RodGridPtsSpec, rotMat, shiftVec) # works for high-order

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Creation of the exterior region # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

x_min = - 2.5*chord
x_max = 4*chord
y_min = - 1.5*chord
y_max = 1.5*chord
elemSize_rect = chord/20/(gridPts_alongNACA/75.0)

x_minBUFF = - 3.0*chord
x_maxBUFF = 8.0*chord
y_minBUFF = - 2.0*chord
y_maxBUFF = 2.0*chord
elemSize_rectBUFF = elemSize_rect

x_minINF = - 10.0*chord
x_maxINF = 15.0*chord
y_minINF = - 10.0*chord
y_maxINF = 10.0*chord
elemSize_rectINF = np.min([50*elemSize_rect, (y_maxINF-y_minINF)/gridPts_alongNACA])


[ rectLine, pointTag, lineTag] = gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag, rotMat, shiftVec)
[ rectLineBUFF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minBUFF, x_maxBUFF, y_minBUFF, y_maxBUFF, elemSize_rectBUFF, pointTag, lineTag, rotMat, shiftVec)
[ rectLineINF, pointTag, lineTag] = gmeshed_rectangle_contour(x_minINF, x_maxINF, y_minINF, y_maxINF, elemSize_rectINF, pointTag, lineTag, rotMat, shiftVec)

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

gmsh.model.geo.add_curve_loop( [*rectLineBUFF, *rectLineINF], surfaceTag+1) 
gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
surfaceTag = surfaceTag+1
surf_unstructINF = surfaceTag

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
gmsh.option.setNumber("Mesh.HighOrderOptimize", highOrderBLoptim) # NB: Where straight layers in BL are satisfactory, use addPlaneSurface() instead of addSurfaceFilling() and remove this high-order optimisation.
gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder) # just visualisation ??

gmsh.model.mesh.generate(2)

# generating a high quality fully hex mesh is a tall order: 
# https://gitlab.onelab.info/gmsh/gmsh/-/issues/784

# gmsh.option.setNumber('Mesh.Recombine3DLevel', 0)
# gmsh.option.setNumber("Mesh.NbTetrahedra", 0)
# gmsh.option.setNumber("Mesh.Algorithm3D", 4) # mesh 3D

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
[nodePerEntity, elemPerEntity] = countDOF()
# gmsh.model.setColor([(2, 3)], 255, 0, 0)  # Red

if CONF == 'rodAirfoil':
    surf_unstructConf = [*structGridSurf, *sTL_rod, surf_unstructCFD]
elif CONF == 'airfoil':
    surf_unstructConf = [*structGridSurf, surf_unstructCFD]
elif CONF == 'rod':
    surf_unstructConf = [*sTL_rod, surf_unstructCFD]

gmsh.model.addPhysicalGroup(pb_2Dim, [*surf_unstructConf], 1, "CFD")
gmsh.model.addPhysicalGroup(pb_2Dim, [surf_unstructBUFF, surf_unstructINF], 2, "Buff")

# export volume mesh only for visualisation:
if CONF == 'rod':
    gmsh.write("2D_rod_"+str(sum(elemPerEntity))+"elems_chordPts"+str(gridPts_alongNACA)+"_mo"+str(elemOrder)+".vtk")
else:
    gmsh.write("2D_"+CONF+"_NACA"+NACA_type+"_"+str(sum(elemPerEntity))+"elems_"+str(int(pitch))+"degAoA_chordPts"+str(gridPts_alongNACA)+"_mo"+str(elemOrder)+".vtk")

if not (CONF == 'airfoil'):
    gmsh.model.addPhysicalGroup(pb_1Dim, [*lTL_rod[lRodArc]], 4, "Rod Hard Wall")
if not (CONF == 'rod'):
    gmsh.model.addPhysicalGroup(pb_1Dim, [*airfoilLine], 5, "Airfoil Hard Wall")

gmsh.model.addPhysicalGroup(pb_1Dim, [*rectLine], 6, "BUFF inner Wrap")

ExtrudUnstruct_bottom = rectLineINF[0]
ExtrudUnstruct_outlet = rectLineINF[1]
ExtrudUnstruct_top = rectLineINF[2]
ExtrudUnstruct_inlet = rectLineINF[3]

gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_inlet], 7, "Inlet BC")
gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_outlet], 8, "Outlet BC")

gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_bottom], 9, "Bottom BC")
gmsh.model.addPhysicalGroup(pb_1Dim, [ExtrudUnstruct_top], 10, "Top BC")

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).

# export mesh with all tags for computation:
if CONF == 'rod':
    gmsh.write("2D_rod_"+str(sum(elemPerEntity))+"elems_chordPts"+str(gridPts_alongNACA)+"_mo"+str(elemOrder)+".msh")
else:
    gmsh.write("2D_"+CONF+"_NACA"+NACA_type+"_"+str(sum(elemPerEntity))+"elems_"+str(int(pitch))+"degAoA_chordPts"+str(gridPts_alongNACA)+"_mo"+str(elemOrder)+".msh")

# export surfaces where the solution will be later interpolated.
gmsh.model.removePhysicalGroups()
if not (CONF == 'airfoil'):
    gmsh.model.addPhysicalGroup(pb_1Dim, [*lTL_rod[lRodArc]], 1, "Rod Hard Wall")
    gmsh.write("2D_rod_"+str(sum(elemPerEntity))+"elems_chordPts"+str(gridPts_alongNACA)+"_mo"+str(elemOrder)+"_rodSurf.msh")

gmsh.model.removePhysicalGroups()
if not (CONF == 'rod'):
    gmsh.model.addPhysicalGroup(pb_1Dim, [*airfoilLine], 1, "Airfoil Hard Wall")
    gmsh.write("2D_"+CONF+"_NACA"+NACA_type+"_"+str(sum(elemPerEntity))+"elems_"+str(int(pitch))+"degAoA_chordPts"+str(gridPts_alongNACA)+"_mo"+str(elemOrder)+"_airfoilSurf.msh")

# paraview support for High-order meshes: https://www.kitware.com/high-order-using-gmsh-reader-plugin-in-paraview/

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Calculate the first cell size # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

if not (CONF == 'rod'):
    ### Computation of the 1st cell height
    if (gridGeomProg_inBL==1):
        height_firstCell_LE = height_LE/gridPts_inBL
        height_firstCell_TE = height_TE/gridPts_inBL
    else:
        #  h_{i+1} = h_i*gridGeomProg_inBL
        #  H_tot = Sum(h_i)_{i=1..gridPts_inBL} = firstCell * (1-gridGeomProg_inBL^gridPts_inBL)/(1-gridGeomProg_inBL)
        height_firstCell_LE = height_LE*(1-gridGeomProg_inBL)/(1-gridGeomProg_inBL**gridPts_inBL)
        height_firstCell_TE = height_TE*(1-gridGeomProg_inBL)/(1-gridGeomProg_inBL**gridPts_inBL)
    print("Quality : 1st cell size @LE = "+ '{:.2e}'.format(height_firstCell_LE/chord)+" * chord")
    print("Quality : 1st cell size @TE = "+ '{:.2e}'.format(height_firstCell_TE/chord)+" * chord")

    ### Computation of the NACA profile length (suction side)
    # arc length L of a function y=f(x) for x=a..b is L = int_{x=a..b} sqrt(1+ (df(x)/dx)^2) dx 
    # the coordinates are accessed through the api intsead: https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/api/detail/
    suctionLine_PG_tag = 99
    gmsh.model.addPhysicalGroup(pb_1Dim, [*airfoilLineSuction], suctionLine_PG_tag, "airfoil suction line")
    line_airfoilUp_coord = gmsh.model.mesh.getNodesForPhysicalGroup(1, suctionLine_PG_tag)[1]
    line_airfoilUp_coord = line_airfoilUp_coord.reshape(elemOrder*(gridPts_alongNACA-1)+1,3)
    line_airfoilUp_coord = line_airfoilUp_coord[line_airfoilUp_coord[:,0].argsort()] # sorting by chordwise coordinates https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    suctionSideArcLength = 0
    for i in range(0, np.shape(line_airfoilUp_coord)[0]-1):
        suctionSideArcLength = suctionSideArcLength + np.sqrt( (line_airfoilUp_coord[i+1,0]-line_airfoilUp_coord[i,0])**2 + (line_airfoilUp_coord[i+1,1]-line_airfoilUp_coord[i,1])**2)
    # from matplotlib import pyplot as plt
    # plt.plot(line_airfoilUp_coord[:,0],line_airfoilUp_coord[:,1],'-+')
    # plt.show()
    print("Quality : cell size along chord = "+ '{:.2e}'.format(suctionSideArcLength/((gridPts_alongNACA-1)*chord))+ " * chord")


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
