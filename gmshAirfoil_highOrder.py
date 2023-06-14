# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

import sys
import gmsh
from gmshToolkit import *
import shutil 

def gmeshed_airfoil_HO(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec):

    pointTag = structTag[0]
    lineTag = structTag[1]
    surfaceTag = structTag[2]

    NACA_type = GeomSpec[0]
    bluntTrailingEdge = GeomSpec[1]
    AoA = GeomSpec[2]
    chord = GeomSpec[3]
    airfoilReferenceAlongChord = GeomSpec[4]
    airfoilReferenceCoordinate = GeomSpec[5]
    height_LE = GeomSpec[6]
    height_TE = GeomSpec[7]
    TEpatchLength = GeomSpec[8]
    TEpatchGridFlaringAngle = GeomSpec[9]
    wakeLength = GeomSpec[10]
    wakeGridFlaringAngle = GeomSpec[11]

    gridPts_alongNACA = GridPtsSpec[0]
    gridPts_inBL = GridPtsSpec[1]
    gridPts_inTE = GridPtsSpec[2]
    gridPts_alongTEpatch = GridPtsSpec[3]
    gridPts_alongWake = GridPtsSpec[4]
    gridGeomProg_inBL = GridPtsSpec[5]
    gridGeomProg_alongTEpatch = GridPtsSpec[6]
    gridGeomProg_alongWake = GridPtsSpec[7]

    shiftVec = np.array(shiftVec)
    airfoilReferenceCoordinate = np.array(airfoilReferenceCoordinate)

    optimisedGridSpacing = True
    spline_controlPts = 500

    [upper_NACAfoil, lower_NACAfoil, camberLine, upper_offset, lower_offset, theta_TE] = NACAxxx(NACA_type, bluntTrailingEdge, AoA, chord, airfoilReferenceAlongChord, height_LE, height_TE, optimisedGridSpacing, spline_controlPts)
    dyc_dx_TE = np.tan(theta_TE*np.pi/180)

    # translate the airfoil to the airfoil reference center
    for i in range(2):
        upper_NACAfoil[i,:] = upper_NACAfoil[i,:] - airfoilReferenceCoordinate[i]
        lower_NACAfoil[i,:] = lower_NACAfoil[i,:] - airfoilReferenceCoordinate[i]
        upper_offset[i,:] = upper_offset[i,:] - airfoilReferenceCoordinate[i]
        lower_offset[i,:] = lower_offset[i,:] - airfoilReferenceCoordinate[i]

    # simply compute the corners coordinate to create the TEpatch region
    deltaTEpatch_flaringAngle = TEpatchLength*np.tan(TEpatchGridFlaringAngle*np.pi/180)/np.cos(theta_TE*np.pi/180)

    x_TEpatch = np.array([lower_offset[0,-1] + TEpatchLength + deltaTEpatch_flaringAngle*np.cos(np.pi/2 - theta_TE*np.pi/180) , upper_offset[0,0] + TEpatchLength - deltaTEpatch_flaringAngle*np.cos(np.pi/2 - theta_TE*np.pi/180)])
    y_TEpatch = np.array([lower_offset[1,-1] + TEpatchLength*dyc_dx_TE - deltaTEpatch_flaringAngle*np.sin(np.pi/2 - theta_TE*np.pi/180), upper_offset[1,0] + TEpatchLength*dyc_dx_TE + deltaTEpatch_flaringAngle*np.sin(np.pi/2 - theta_TE*np.pi/180)])

    # simply compute the corners coordinate to stretch the wake region
    deltaWake_flaringAngle = wakeLength*np.tan(wakeGridFlaringAngle*np.pi/180)/np.cos(theta_TE*np.pi/180)

    x_wake = np.array([x_TEpatch[0] + wakeLength + deltaWake_flaringAngle*np.cos(np.pi/2 - theta_TE*np.pi/180), x_TEpatch[1] + wakeLength - deltaWake_flaringAngle*np.cos(np.pi/2 - theta_TE*np.pi/180)])
    y_wake = np.array([y_TEpatch[0] + wakeLength*dyc_dx_TE - deltaWake_flaringAngle*np.sin(np.pi/2 - theta_TE*np.pi/180), y_TEpatch[1] + wakeLength*dyc_dx_TE + deltaWake_flaringAngle*np.sin(np.pi/2 - theta_TE*np.pi/180)])

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Points # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # creation of the NACA profile
    point_TEu = pointTag+1
    for i in range(spline_controlPts):
        rotVec = np.matmul(rotMat, np.array([upper_NACAfoil[0,i], upper_NACAfoil[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_LE = pointTag
    if bluntTrailingEdge:
        for i in range(spline_controlPts-1):
            rotVec = np.matmul(rotMat, np.array([lower_NACAfoil[0,i], lower_NACAfoil[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
            gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
        pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
        point_TEl = pointTag
        rotVec = np.matmul(rotMat, np.array([0.5*(upper_NACAfoil[0,0]+lower_NACAfoil[0,-1]), 0.5*(upper_NACAfoil[1,0]+lower_NACAfoil[1,-1]), airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_TE = pointTag
    else:
        for i in range(spline_controlPts-2):
            rotVec = np.matmul(rotMat, np.array([lower_NACAfoil[0,i], lower_NACAfoil[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
            gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
        pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
        point_TEl = point_TEu
        point_TE = point_TEu

    # creation of the offset layer
    point_up = pointTag+1
    for i in range(spline_controlPts):
        rotVec = np.matmul(rotMat, np.array([upper_offset[0,i], upper_offset[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_left = pointTag
    for i in range(spline_controlPts-1):
        rotVec = np.matmul(rotMat, np.array([lower_offset[0,i], lower_offset[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_low = pointTag

    # creation of the TEpatch
    point_lowRight = pointTag+1
    for i in range(np.size(x_TEpatch)):
        rotVec = np.matmul(rotMat, np.array([x_TEpatch[i], y_TEpatch[i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_upRight = pointTag

    point_lowFarRight = pointTag+1
    for i in range(np.size(x_wake)):
        rotVec = np.matmul(rotMat, np.array([x_wake[i], y_wake[i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_upFarRight = pointTag

    rotVec = np.matmul(rotMat, np.array([0.5*(x_TEpatch[0]+x_TEpatch[-1]), 0.5*(y_TEpatch[0]+y_TEpatch[-1]), airfoilReferenceCoordinate[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_TEwake = pointTag

    rotVec = np.matmul(rotMat, np.array([0.5*(x_wake[0]+x_wake[-1]), 0.5*(y_wake[0]+y_wake[-1]), airfoilReferenceCoordinate[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_TEfarWake = pointTag

    # to define transfinite region, number of gridPoint must be >=2.
    gridPts_inBL = max(gridPts_inBL,2)
    gridPts_inTE = max(gridPts_inTE,2)
    # for the Blossom algorithm to work (setRecombine when applied for the regular CAA domain), an odd
    # number of points in the contour needs to be specified. All edges have a up/low twin appart from
    #  the line F. The discretisation of the TE is rounded so to ensure that an odd number of points in 
    # the periphery of the profile is used. 
    if ~(gridPts_inTE % 2):
        gridPts_inTE = gridPts_inTE+1

    if bluntTrailingEdge:
        alphaStretch = (max(1,gridPts_inTE-2))/(gridPts_inTE+gridPts_inBL-3)

        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_TEpatch[-1]+0.5*(1+alphaStretch)*x_TEpatch[0], 
                                0.5*(1-alphaStretch)*y_TEpatch[-1]+0.5*(1+alphaStretch)*y_TEpatch[0],
                                airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_lowMidRight = pointTag

        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_TEpatch[0]+0.5*(1+alphaStretch)*x_TEpatch[-1], 
                                0.5*(1-alphaStretch)*y_TEpatch[0]+0.5*(1+alphaStretch)*y_TEpatch[-1],
                                airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_upMidRight = pointTag

        # ~~~
        
        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_wake[-1]+0.5*(1+alphaStretch)*x_wake[0], 
                                0.5*(1-alphaStretch)*y_wake[-1]+0.5*(1+alphaStretch)*y_wake[0],
                                airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_lowMidFarRight = pointTag

        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_wake[0]+0.5*(1+alphaStretch)*x_wake[-1], 
                                0.5*(1-alphaStretch)*y_wake[0]+0.5*(1+alphaStretch)*y_wake[-1],
                                airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_upMidFarRight = pointTag
    else:
        point_lowMidRight = point_TEwake
        point_upMidRight = point_TEwake
        point_lowMidFarRight = point_TEfarWake
        point_upMidFarRight = point_TEfarWake







    # $$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Lines # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$

    # creation of the NACA profile
    gmsh.model.geo.addSpline(range(point_TEu, point_LE+1), lineTag+1) 
    lineTag = lineTag+1 # store the last 'pointTag' from previous loop
    airfoilExtrado_lineTag = lineTag
    gmsh.model.geo.addSpline(range(point_LE, point_TEl+1), lineTag+1) 
    lineTag = lineTag+1 # store the last 'pointTag' from previous loop
    airfoilIntrado_lineTag = lineTag
    
    # creation of the offset layer
    gmsh.model.geo.addSpline(range(point_up, point_left+1), lineTag+1) 
    lineTag = lineTag+1 # store the last 'pointTag' from previous loop
    airfoilExtradoOffset_lineTag = lineTag
    gmsh.model.geo.addSpline(range(point_left, point_low+1), lineTag+1) 
    lineTag = lineTag+1 # store the last 'pointTag' from previous loop
    airfoilIntradoOffset_lineTag = lineTag


    gmsh.model.geo.add_line(point_TEu, point_up,lineTag+1)
    lineTag = lineTag+1
    line_A = lineTag

    gmsh.model.geo.add_line(point_LE, point_left,lineTag+1)
    lineTag = lineTag+1
    line_G = lineTag

    gmsh.model.geo.add_line(point_TEl, point_low,lineTag+1)
    lineTag = lineTag+1
    line_B = lineTag



    # # creation of the NACA profile
    # airfoil_startLineTag = lineTag + 1
    # for i in range(spline_controlPts-1):
    #     gmsh.model.geo.add_line(point_TEu+i, point_TEu+i+1,lineTag+i+1)
    # lineTag = lineTag+i+1 # store the last 'lineTag' from previous loop
    # airfoil_LE_lineTag = lineTag
    # if bluntTrailingEdge:
    #     for i in range(spline_controlPts-1):
    #         gmsh.model.geo.add_line(point_LE+i, point_LE+i+1,lineTag+i+1)
    # else:
    #     for i in range(spline_controlPts-2):
    #         gmsh.model.geo.add_line(point_LE+i, point_LE+i+1,lineTag+i+1)
    #     # if the TE is sharp, the last point corresponds to the first one
    #     i = i+1
    #     gmsh.model.geo.add_line(point_LE+i, point_TEl, lineTag+i+1)
    # lineTag = lineTag+i+1 # store the last 'lineTag' from previous loop
    # airfoil_endLineTag = lineTag
    #
    # # creation of the offset layer
    # structGrid_startLineTag = lineTag + 1
    # for i in range(spline_controlPts-1):
    #     gmsh.model.geo.add_line(point_up+i, point_up+i+1,lineTag+i+1)
    # lineTag = lineTag+i+1
    # structGrid_LE_lineTag = lineTag
    # for i in range(spline_controlPts-1):
    #     gmsh.model.geo.add_line(point_left+i, point_left+i+1,lineTag+i+1)
    # lineTag = lineTag+i+1 # store the last 'lineTag' from previous loop
    # structGrid_endLineTag = lineTag

    # line_A = lineTag+1
    # for i in range(spline_controlPts-1):
    #     gmsh.model.geo.add_line(point_TEu+i, point_up+i,lineTag+i+1)
    # lineTag = lineTag+i+1
    # line_G = lineTag+1
    # for i in range(spline_controlPts-1):
    #     gmsh.model.geo.add_line(point_LE+i, point_left+i,lineTag+i+1)
    # i = i+1
    # gmsh.model.geo.add_line(point_TEl, point_low,lineTag+i+1)
    # lineTag = lineTag+i+1
    # line_B = lineTag

    gmsh.model.geo.add_line(point_lowRight, point_lowMidRight,lineTag+1)
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_Br = lineTag
    gmsh.model.geo.add_line(point_upMidRight, point_upRight,lineTag+1)
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_Ar = lineTag

    gmsh.model.geo.add_line(point_lowFarRight, point_lowMidFarRight,lineTag+1)
    lineTag = lineTag + 1
    line_Ber = lineTag
    gmsh.model.geo.add_line(point_upMidFarRight, point_upFarRight,lineTag+1)
    lineTag = lineTag + 1
    line_Aer = lineTag

    gmsh.model.geo.add_line(point_low, point_lowRight,lineTag+1)
    lineTag = lineTag + 1
    line_C = lineTag
    
    gmsh.model.geo.add_line(point_upRight,point_up,lineTag+1)
    lineTag = lineTag + 1
    line_D = lineTag

    gmsh.model.geo.add_line(point_lowRight,point_lowFarRight,lineTag+1)
    lineTag = lineTag + 1
    line_H = lineTag

    gmsh.model.geo.add_line(point_upFarRight, point_upRight,lineTag+1)
    lineTag = lineTag + 1
    line_I = lineTag

    gmsh.model.geo.add_line(point_TE,point_TEwake,lineTag+1)
    lineTag = lineTag + 1
    line_K = lineTag

    gmsh.model.geo.add_line(point_TEwake, point_TEfarWake, lineTag+1)
    lineTag = lineTag + 1
    line_N = lineTag

    if bluntTrailingEdge: # create a line for the TE
        gmsh.model.geo.add_line(point_TEl,point_TE,lineTag+1)
        lineTag = lineTag + 1
        line_El = lineTag
        gmsh.model.geo.add_line(point_TE, point_TEu,lineTag+1)
        lineTag = lineTag + 1
        line_Eu = lineTag
        gmsh.model.geo.add_line(point_lowMidRight, point_TEwake,lineTag+1)
        lineTag = lineTag+1 # store the last 'lineTag' from previous loop
        line_Fl = lineTag
        gmsh.model.geo.add_line(point_TEwake, point_upMidRight,lineTag+1)
        lineTag = lineTag+1 # store the last 'lineTag' from previous loop
        line_Fu = lineTag
        gmsh.model.geo.add_line(point_lowMidFarRight, point_TEfarWake,lineTag+1)
        lineTag = lineTag + 1
        line_Jl = lineTag
        gmsh.model.geo.add_line(point_TEfarWake, point_upMidFarRight,lineTag+1)
        lineTag = lineTag + 1
        line_Ju = lineTag

        gmsh.model.geo.add_line(point_TEu,point_upMidRight,lineTag+1)
        lineTag = lineTag + 1
        line_M = lineTag 
        gmsh.model.geo.add_line(point_TEl, point_lowMidRight,lineTag+1)
        lineTag = lineTag + 1
        line_L = lineTag
        gmsh.model.geo.add_line(point_lowMidRight, point_lowMidFarRight,lineTag+1)
        lineTag = lineTag + 1
        line_O = lineTag
        gmsh.model.geo.add_line(point_upMidRight, point_upMidFarRight,lineTag+1)
        lineTag = lineTag + 1
        line_P = lineTag

    else:
        line_M = line_K
        line_L = line_K
        line_O = line_N
        line_P = line_N
        line_El = -1
        line_Eu = -1
        line_Fl = -1
        line_Fu = -1
        line_Jl = -1
        line_Ju = -1


    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of some general CurveLoops # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    line_airfoilUp = [airfoilExtrado_lineTag]
    line_airfoilLow = [airfoilIntrado_lineTag]

    line_BLup = [airfoilExtradoOffset_lineTag]
    line_BLlow = [airfoilIntradoOffset_lineTag]
    line_BLradii = list(range(line_A, line_B+1))


    # line_airfoilUp = list(range(airfoil_startLineTag, airfoil_LE_lineTag+1))
    # line_airfoilLow = list(range(airfoil_LE_lineTag+1, airfoil_endLineTag+1))

    # line_BLup = list(range(structGrid_startLineTag, structGrid_LE_lineTag+1))
    # line_BLlow = list(range(structGrid_LE_lineTag+1, structGrid_endLineTag+1))
    # line_BLradii = list(range(line_A, line_B+1))

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the surfaces # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # # mesh of the airfoil interior (test --> OK)
    #
    # line_airfoil = [*line_airfoilUp, *line_airfoilLow]
    # # line_airfoil = list(range(airfoil_startLineTag, airfoil_endLineTag+1))
    # if bluntTrailingEdge: # taking into account of the TE to close the contour
    #     airfoil_boundaries = [*line_airfoil, line_El, line_Eu]
    # else:
    #     airfoil_boundaries = [*line_airfoil]
    #
    # gmsh.model.geo.add_curve_loop( airfoil_boundaries, surfaceTag+1) 
    # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
    # if bluntTrailingEdge:
    #     if (spline_controlPts <= gridPts_inTE):
    #         print("Warning. A struct mesh cannot be created with this ration of points in the blunt TE and along the airfoil. Choose gridPts_alongNACA/gridPts_inTE > 1")
    #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [point_TEl, point_TEu, point_LE-gridPts_inTE+2, point_LE+gridPts_inTE-2])
    # else:
    #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [point_TE, point_TE+int(spline_controlPts/2), point_LE, point_LE+int(spline_controlPts/2)])
    # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # surfaceTag = surfaceTag+1
    # surf_airfoil = surfaceTag

    # # (structured / transfinite) boundary layer grid

    gmsh.model.geo.mesh.setTransfiniteCurve(airfoilExtrado_lineTag, gridPts_alongNACA)
    gmsh.model.geo.mesh.setTransfiniteCurve(airfoilIntrado_lineTag, gridPts_alongNACA)
    gmsh.model.geo.mesh.setTransfiniteCurve(airfoilExtradoOffset_lineTag, gridPts_alongNACA)
    gmsh.model.geo.mesh.setTransfiniteCurve(airfoilIntradoOffset_lineTag, gridPts_alongNACA)

    gmsh.model.geo.mesh.setTransfiniteCurve(line_A, gridPts_inBL, "Progression", gridGeomProg_inBL)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_G, gridPts_inBL, "Progression", gridGeomProg_inBL)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_B, gridPts_inBL, "Progression", gridGeomProg_inBL)

    gmsh.model.geo.add_curve_loop([airfoilExtradoOffset_lineTag, -line_G, -airfoilExtrado_lineTag, line_A], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1 








    # # for i in range(2*spline_controlPts-2):
    # #     gmsh.model.geo.mesh.setTransfiniteCurve(airfoil_startLineTag+i, 2) # just create one cell between two consecutive NACA gridtPts 
    # #     gmsh.model.geo.mesh.setTransfiniteCurve(structGrid_startLineTag+i, 2) # just create one cell between two consecutive NACA gridtPts 

    # # for i in range(2*spline_controlPts-1):
    # #     gmsh.model.geo.mesh.setTransfiniteCurve(line_BLradii[i], gridPts_inBL, "Progression", gridGeomProg_inBL)

    # # BLstructStartSurfTag = surfaceTag+1
    # # for i in range(2*spline_controlPts-2):
    # #     gmsh.model.geo.add_curve_loop([structGrid_startLineTag+i, -line_BLradii[i+1], -(airfoil_startLineTag+i), line_BLradii[i]], surfaceTag+1)
    # #     gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    # #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    # #     gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # #     surfaceTag = surfaceTag+1 
    # # BLstructEndSurfTag = surfaceTag

    # # surf_BLstructGridUp = list(range(BLstructStartSurfTag, BLstructStartSurfTag+spline_controlPts-1))
    # # surf_BLstructGridLow = list(range(BLstructStartSurfTag+spline_controlPts-1, BLstructEndSurfTag+1))
    # # surf_BLstructGrid = [*surf_BLstructGridUp, *surf_BLstructGridLow]

    # # (structured / transfinite) TEpatch region

    # TEpatch_boundariesl = [line_C, line_Br, -line_L, line_B]
    # TEpatch_boundariesu = [line_M, line_Ar, line_D, -line_A]

    # gmsh.model.geo.add_curve_loop( TEpatch_boundariesl, surfaceTag+1)
    # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # surfaceTag = surfaceTag+1
    # surf_TEpatchLow = surfaceTag
    
    # gmsh.model.geo.add_curve_loop( TEpatch_boundariesu, surfaceTag+1)
    # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # surfaceTag = surfaceTag+1
    # surf_TEpatchUp = surfaceTag

    # if bluntTrailingEdge: 
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_El, gridPts_inTE-1, "Progression", gridGeomProg_inBL)
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_Eu, gridPts_inTE-1, "Progression", -gridGeomProg_inBL)
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_Fl, gridPts_inTE-1)
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_Fu, gridPts_inTE-1)

    #     TEpatch_boundariesml = [line_L, line_Fl, -line_K, -line_El]
    #     TEpatch_boundariesmu = [line_K, line_Fu, -line_M, -line_Eu]

    #     gmsh.model.geo.add_curve_loop( TEpatch_boundariesml, surfaceTag+1)
    #     gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    #     gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    #     surfaceTag = surfaceTag+1
    #     surf_TEpatchMidLow = surfaceTag

    #     gmsh.model.geo.add_curve_loop( TEpatch_boundariesmu, surfaceTag+1)
    #     gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    #     gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    #     surfaceTag = surfaceTag+1
    #     surf_TEpatchMidUp = surfaceTag
    # else:
    #     surf_TEpatchMidLow = -1
    #     surf_TEpatchMidUp = -1


    # gmsh.model.geo.mesh.setTransfiniteCurve(line_Ar, gridPts_inBL)
    # gmsh.model.geo.mesh.setTransfiniteCurve(line_Br, gridPts_inBL)

    # gmsh.model.geo.mesh.setTransfiniteCurve(line_C, gridPts_alongTEpatch)
    # gmsh.model.geo.mesh.setTransfiniteCurve(line_D, gridPts_alongTEpatch)

    # gmsh.model.geo.mesh.setTransfiniteCurve(line_K, gridPts_alongTEpatch,"Progression", gridGeomProg_alongTEpatch)

    # if bluntTrailingEdge:
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_M, gridPts_alongTEpatch,"Progression", gridGeomProg_alongTEpatch)
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_L, gridPts_alongTEpatch,"Progression", gridGeomProg_alongTEpatch)

    # # (structured / transfinite) wake region
    
    # # if bluntTrailingEdge:
    # #     wake_boundaries = [line_H, line_Ber, line_Jl, line_Ju, line_Aer, line_I, -line_Ar, -line_Fu, -line_Fl, -line_Br]
    # # else:
    # #     wake_boundaries = [line_H, line_Ber, line_Aer, line_I, -line_Ar, -line_Br]
    
    # # wake_curveLoop = gmsh.model.geo.add_curve_loop( wake_boundaries, surfaceTag+1)
    # # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)        
    # # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [point_lowRight, point_lowFarRight, point_upFarRight, point_upRight])
    # # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # # surfaceTag = surfaceTag+1
    # # surf_wake = surfaceTag

    # # if bluntTrailingEdge:
    # #     gmsh.model.geo.mesh.setTransfiniteCurve(line_J, 2*(gridPts_inBL+gridPts_inTE)-5) 
    # # else:
    # #     gmsh.model.geo.mesh.setTransfiniteCurve(line_J, 2*gridPts_inBL-1)


    # wake_boundariesl = [line_H, line_Ber, -line_O, -line_Br]
    # wake_boundariesu = [line_P, line_Aer, line_I, -line_Ar]

    # gmsh.model.geo.add_curve_loop( wake_boundariesl, surfaceTag+1)
    # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # surfaceTag = surfaceTag+1
    # surf_wakeLow = surfaceTag
    
    # gmsh.model.geo.add_curve_loop( wake_boundariesu, surfaceTag+1)
    # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # surfaceTag = surfaceTag+1
    # surf_wakeUp = surfaceTag

    # if bluntTrailingEdge:
    #     wake_boundariesml = [line_O, line_Jl, -line_N, -line_Fl]
    #     wake_boundariesmu = [line_N, line_Ju, -line_P, -line_Fu]

    #     gmsh.model.geo.add_curve_loop( wake_boundariesml, surfaceTag+1)
    #     gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    #     gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    #     surfaceTag = surfaceTag+1
    #     surf_wakeMidLow = surfaceTag

    #     gmsh.model.geo.add_curve_loop( wake_boundariesmu, surfaceTag+1)
    #     gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    #     gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    #     gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    #     surfaceTag = surfaceTag+1
    #     surf_wakeMidUp = surfaceTag
    # else:
    #     surf_wakeMidLow = -1
    #     surf_wakeMidUp = -1

    # gmsh.model.geo.mesh.setTransfiniteCurve(line_Aer, gridPts_inBL)
    # gmsh.model.geo.mesh.setTransfiniteCurve(line_Ber, gridPts_inBL)

    # if bluntTrailingEdge:
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_Jl, gridPts_inTE-1) 
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_Ju, gridPts_inTE-1)

    # gmsh.model.geo.mesh.setTransfiniteCurve(line_H, gridPts_alongWake,"Progression", gridGeomProg_alongWake)
    # gmsh.model.geo.mesh.setTransfiniteCurve(line_I, gridPts_alongWake,"Progression", -gridGeomProg_alongWake)
    # gmsh.model.geo.mesh.setTransfiniteCurve(line_N, gridPts_alongWake,"Progression", gridGeomProg_alongWake)
    # if bluntTrailingEdge:
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_P, gridPts_alongWake,"Progression", gridGeomProg_alongWake)
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_O, gridPts_alongWake,"Progression", gridGeomProg_alongWake)


    # # connection to regular CAA domain:

    pointTag_list = [point_LE, point_TE, point_TEu, point_TEl, point_TEwake, point_TEfarWake, point_left, point_up, point_upRight, point_upFarRight, point_low, point_lowRight, point_lowFarRight, point_upMidRight, point_lowMidRight, point_upMidFarRight, point_lowMidFarRight]
    lineTag_list = [[*line_airfoilUp], [*line_airfoilLow], [*line_BLup], [*line_BLlow], [*line_BLradii], line_A, line_B, line_C, line_D, line_Eu, line_El, line_Fu, line_Fl, line_G, line_H, line_I, line_Ju, line_Jl, line_K, line_L, line_M, line_N, line_O, line_P, line_Ar, line_Br, line_Aer, line_Ber]
    # surfaceTag_list = [surf_airfoil, [*surf_BLstructGrid], surf_BLstructGridUp, surf_BLstructGridLow, surf_TEpatchUp, surf_TEpatchLow, surf_TEpatchMidUp, surf_TEpatchMidLow, surf_wakeUp, surf_wakeLow, surf_wakeMidUp, surf_wakeMidLow]
    surfaceTag_list = [surfaceTag]

    return pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

























































NACA_type = '0012'

bluntTrailingEdge = True

gridPtsRichness = 0.2

elemOrder = 5

gridPts_alongNACA = 15 # int(75*gridPtsRichness)

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

[pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag] = gmeshed_airfoil_HO(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Generate visualise and export the mesh # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options

gmsh.model.geo.synchronize()

# 2D pavement
# gmsh.option.setNumber("Mesh.Smoothing", 3)
# gmsh.option.setNumber("Mesh.Algorithm", 11) # mesh 2D
gmsh.option.setNumber("Mesh.RecombineAll", 1)


# https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_11_1/tutorials/python/t5.py


# gmsh.option.setNumber("Mesh.HighOrderOptimize", 0) # (0: none, 1: optimization, 2: elastic+optimization, 3: elastic, 4: fast curving)
# gmsh.option.setNumber("Mesh.HighOrderNumLayers", gridGeomProg_inBL)

gmsh.option.setNumber("Mesh.ElementOrder", elemOrder) # gmsh.model.mesh.setOrder(elemOrder)
gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder) # just visualisation ??

gmsh.model.mesh.generate(2)

gmsh.model.setColor([(2, 1)], 255, 0, 0)  # Red

### tests to generate high order meshes:
# # set order after "generate" http://onelab.info/pipermail/gmsh/2019/012941.html
# # alssee: https://gitlab.onelab.info/gmsh/gmsh/-/issues/527
# elemOrder = 2 # (1=linear elements, N (<6) = elements of higher order)
# gmsh.model.mesh.setOrder(elemOrder) 
# gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
# # gmsh.option.setNumber("Mesh.HighOrderMaxInnerAngle", 0.5) 
# # gmsh.option.setNumber("Mesh.NumSubEdges", elemOrder) # just visualisation ??
# # gmsh.option.setNumber("Mesh.HighOrderPoissonRatio", 0.5) # Poisson ratio of the material used in the elastic smoother for high order meshes. Must be between -1.0 and 0.5, excluded
# gmsh.option.setNumber("Mesh.HighOrderOptimize", 2) # (0: none, 1: optimization, 2: elastic+optimization, 3: elastic, 4: fast curving)
# # gmsh.model.mesh.optimize('Netgen', True) # https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_8_0/demos/api/opt.py#L12
### -------------------------------------

# see JuanP74's gmsh config: https://calculix.discourse.group/t/help-error-in-e-c3d-nonpositive-jacobian/1320/25?page=2


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

[nodePerEntity, elemPerEntity] = countDOF()

gmsh.model.addPhysicalGroup(pb_2Dim, [surfaceTag], 1, "CFD")

# Write mesh data:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # when ASCII format 2.2 is selected "Mesh.SaveAll=1" discards the group definitions (to be avoided!).


gmsh.write("NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.msh")
gmsh.write("NACA"+NACA_type+"_foil_"+str(sum(elemPerEntity))+"elems.vtk")

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
