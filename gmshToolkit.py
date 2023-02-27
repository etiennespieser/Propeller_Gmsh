# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

# # Drawing of the NACA profile inspired from
# the Wikipedia page: https://en.wikipedia.org/wiki/NACA_airfoil
# JoshTheEngineer matlab's code: https://github.com/jte0419/NACA_4_Digit_Airfoil

# On the general use of Gmsh (correspondance of the .geo synthax with the python API provided)
# (C++ use of Gmsh is also supported, more consistent with MFEM?)
# https://gmsh.info/doc/texinfo/gmsh.html
# meshing with S.A.E. Miller, "Tutorial on Computational Grid Generation for CFD using GMSH"
# https://youtube.com/playlist?list=PLbiOzt50Bx-l2QyX5ZBv9pgDtIei-CYs_
# or tutorial of Bertrand Thierry:
# https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/ 


# >> Blocking close to the airfoil (inside the <<regular CAA>> block) <<  
#   
#                    line BLUp          point up              line D               point upRight         line I             point upFarRight
#    --------------------<----------------X --------------------<-------------------- X --------------------<-------------------- X
#    |                                    |                         <<TE patchUp>>    |                            <<wake Up>>    |
#    | <<BL structGridUp>>                /\ line A                                   /| line Ar                                  /| line Aer
#    |                    line airfoilUp  | point TEu         line M                  | point upMidRight  line P                  |
#    |                   -------<-------- X -------------------->-------------------- X -------------------->-------------------- X point upMidFarRight
#    |                  |                 |                                           |                                           |
#    |                  V                 /\ (if bluntTrailingEdge: line Eu)          /\ (if bluntTrailingEdge: line Fu)          /| (if bluntTrailingEdge: line Ju)                                           |
#    | point Left       | point LE        | point TE          line K                  | point TEwake     line N                   |
#    X -------<-------- X                 X -------------------->-------------------- X -------------------->-------------------- X point TEfarWake 
#    |     line G       |                 |                                           |                                           |
#    |                  V  <<airfoil>>    /\ (if bluntTrailingEdge: line El)          /\ (if bluntTrailingEdge: line Fl)          /| (if bluntTrailingEdge: line Jl)                                           |
#    |                  |                 | point TEl         line L                  | point lowMidRight line O                  |
#    |                   ------->-------- X -------------------->-------------------- X -------------------->-------------------- X point lowMidFarRight
#    |                   line airfoilLow  |                                           |                                           |
#    | <<BL structGridLow>>               V line B               <<TE patchLow>>      /| line Br                   <<wake Low>>   /| line Ber
#    |                                    |                                           |                                           |
#    -------------------->----------------X -------------------->-------------------- X -------------------->-------------------- X
#                    line BLlow        point low              line C              point lowRight         line H            point lowFarRight
#
#
# "point LE" and "point Left" belongs to the upper part of the airfoil lines and BL lines 
# when the TE is blunt: Tag TEu = Tag TEl = Tag TE
#
# "point TE" and the split into "line Eu" and "line El" in the case of a blunt TE is important 
# because it enables to define two grid geometric progression of the grid at the TE of the
# airfoil.
#
# To create transfinite volumes around the airfoil using the .geo. kernel, each cell of the
# airfoil needs to be manually defined. <<BL structGridUp>> and <<BL structGridLow>> are thus
# sliced in a loop of elementary surface panels. No similar issue for <<TE patchUp>>, <<TE patchLow>>
# or <<wake region>> since their shape is a regular quadrangle.
#
# ******************************************************************************************************************************************************************************


# Import modules:
import gmsh
import numpy as np
import matplotlib.pyplot as plt
pb_1Dim = 1
pb_2Dim = 2
pb_3Dim = 3

def NACAxxx(NACA_type, bluntTrailingEdge, AoA, chord, airfoilReference, height_LE, height_TE, optimisedGridSpacing, gridPts):
    # returns a list of (x,y) coordinates of a given NACA profile. which leading edge is located at (0,0)
    #
    # Extract percentage values of airfoil properties from type of airfoil
    M = int(NACA_type[0])/100   # maximum camber
    P = int(NACA_type[1])/10    # location of maximum camber
    T = int(NACA_type[2:4])/100 # maximum thickness
    #
    # Constants used in thickness calculation
    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843
    if bluntTrailingEdge:
        a4 = -0.1015        # Open trailing edge
    else:
        a4 = -0.1036        # Closed trailing edge
    #
    # Airfoil X points
    x = np.linspace(0,1,gridPts)    # Uniform spacing
    if optimisedGridSpacing:
        # x = 0.5*(1-np.cos(x*np.pi))                                    # Non-uniform spacing - v0
        MTP = 1/3*np.max([0.1,P]) # meshTransitionParam
        x = MTP*(1-np.cos(x*np.pi/(2*MTP)))*(x < MTP) + x*(x >= MTP)     # Non-uniform spacing - v1

    # Camber line and camber line gradient
    yc     = np.ones(gridPts)
    dyc_dx = np.ones(gridPts)
    theta  = np.ones(gridPts)
    for i in range(gridPts):
        if (x[i] >= 0) & (x[i] < P):
            yc[i]     = M/P**2*(2*P*x[i]-x[i]**2)
            dyc_dx[i] = 2*M/P**2*(P-x[i])
        elif (x[i] >=P) & (x[i] <=1):
            yc[i]     = M/(1-P)**2*(1-2*P+2*P*x[i]-x[i]**2)
            dyc_dx[i] = 2*M/(1-P)**2*(P-x[i])
        theta[i] = np.arctan(dyc_dx[i])
    #
    # Thickness distribution
    yt = 5*T*(a0*x**0.5 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
    dyt_dx = 5*T*(a0/(2*x[1:len(x)]**0.5) + a1 + 2*a2*x[1:len(x)] + 3*a3*x[1:len(x)]**2 + 4*a4*x[1:len(x)]**3)
    #
    # Upper surface points
    xu = x  - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    #
    # Lower surface points
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    #
    ht = height_LE/chord + (height_TE-height_LE)*x[1:len(x)]/chord # NB: turbulent boundary layers grow rather linearly, laminar ones rather like square root.
    x_offset = x[1:len(x)]-ht*dyt_dx/(1+dyt_dx**2)**0.5
    y_offset = yt[1:len(x)] + ht/(1+dyt_dx**2)**0.5
    x_offset = np.concatenate((np.array([x[0]- height_LE/chord]),x_offset))
    y_offset = np.concatenate((np.array([0]),y_offset))
    #
    xu_offset = x + (x_offset-x)*np.cos(theta) - y_offset*np.sin(theta)
    yu_offset = yc + y_offset*np.cos(theta) + (x_offset-x)*np.sin(theta)
    xl_offset = x + (x_offset-x)*np.cos(theta) + y_offset*np.sin(theta)
    yl_offset = yc - y_offset*np.cos(theta) + (x_offset-x)*np.sin(theta)
    #
    # Redefine the airfoil reference
    x = x - airfoilReference/chord
    xu = xu - airfoilReference/chord
    xl = xl - airfoilReference/chord
    xu_offset = xu_offset - airfoilReference/chord
    xl_offset = xl_offset - airfoilReference/chord
    #
    # Rotation of (x, yc), (xl, yl) and (xu, yu) to account for the AoA
    rotationMat = np.array([[np.cos(AoA*np.pi/180),np.sin(AoA*np.pi/180)],[-np.sin(AoA*np.pi/180),np.cos(AoA*np.pi/180)]])
    #
    # rotation performed around the axis (0 , 0)
    lower_rot = np.matmul(rotationMat,np.array([xl,yl]))
    upper_rot = np.matmul(rotationMat,np.array([xu,yu]))
    camber_rot = np.matmul(rotationMat,np.array([x,yc]))
    lower_offset_rot = np.matmul(rotationMat,np.array([xl_offset,yl_offset]))
    upper_offset_rot = np.matmul(rotationMat,np.array([xu_offset,yu_offset]))
    #
    # dimensionalise the profile
    x = x*chord
    lower_rot = lower_rot*chord
    upper_rot = upper_rot*chord
    camber_rot = camber_rot*chord
    lower_offset_rot = lower_offset_rot*chord
    upper_offset_rot = upper_offset_rot*chord
    #
    return np.flip(upper_rot,axis=1), lower_rot[:,1:], camber_rot, np.flip(upper_offset_rot,axis=1), lower_offset_rot[:,1:], 180*theta[-1]/np.pi-AoA

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def rotationMatrix(rotAnglesVec): # rotation around axis [axisZ, axisY, axisX]
    yawRot = np.array([[np.cos(rotAnglesVec[0]*np.pi/180), -np.sin(rotAnglesVec[0]*np.pi/180), 0.0],
                       [np.sin(rotAnglesVec[0]*np.pi/180), np.cos(rotAnglesVec[0]*np.pi/180), 0.0],
                       [0.0, 0.0, 1.0]])

    pitchRot = np.array([[np.cos(rotAnglesVec[1]*np.pi/180), 0.0, np.sin(rotAnglesVec[1]*np.pi/180)],
                        [0.0, 1.0, 0.0],
                       [-np.sin(rotAnglesVec[1]*np.pi/180), 0.0, np.cos(rotAnglesVec[1]*np.pi/180)]])

    rollRot = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rotAnglesVec[2]*np.pi/180), -np.sin(rotAnglesVec[2]*np.pi/180)],
                       [0.0, np.sin(rotAnglesVec[2]*np.pi/180), np.cos(rotAnglesVec[2]*np.pi/180)]])

    rotMat = np.matmul(yawRot,np.matmul(pitchRot,rollRot)) # https://en.wikipedia.org/wiki/Rotation_matrix
    return rotMat

def gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec):

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

    [upper_NACAfoil, lower_NACAfoil, camberLine, upper_offset, lower_offset, theta_TE] = NACAxxx(NACA_type, bluntTrailingEdge, AoA, chord, airfoilReferenceAlongChord, height_LE, height_TE, optimisedGridSpacing, gridPts_alongNACA)
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
    for i in range(gridPts_alongNACA):
        rotVec = np.matmul(rotMat, np.array([upper_NACAfoil[0,i], upper_NACAfoil[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_LE = pointTag
    if bluntTrailingEdge:
        for i in range(gridPts_alongNACA-1):
            rotVec = np.matmul(rotMat, np.array([lower_NACAfoil[0,i], lower_NACAfoil[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
            gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
        pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
        point_TEl = pointTag
        rotVec = np.matmul(rotMat, np.array([0.5*(upper_NACAfoil[0,0]+lower_NACAfoil[0,-1]), 0.5*(upper_NACAfoil[1,0]+lower_NACAfoil[1,-1]), airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_TE = pointTag
    else:
        for i in range(gridPts_alongNACA-2):
            rotVec = np.matmul(rotMat, np.array([lower_NACAfoil[0,i], lower_NACAfoil[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
            gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
        pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
        point_TEl = point_TEu
        point_TE = point_TEu

    # creation of the offset layer
    point_up = pointTag+1
    for i in range(gridPts_alongNACA):
        rotVec = np.matmul(rotMat, np.array([upper_offset[0,i], upper_offset[1,i], airfoilReferenceCoordinate[2]])) + shiftVec
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_left = pointTag
    for i in range(gridPts_alongNACA-1):
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
    airfoil_startLineTag = lineTag + 1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(point_TEu+i, point_TEu+i+1,lineTag+i+1)
    lineTag = lineTag+i+1 # store the last 'lineTag' from previous loop
    airfoil_LE_lineTag = lineTag
    if bluntTrailingEdge:
        for i in range(gridPts_alongNACA-1):
            gmsh.model.geo.add_line(point_LE+i, point_LE+i+1,lineTag+i+1)
    else:
        for i in range(gridPts_alongNACA-2):
            gmsh.model.geo.add_line(point_LE+i, point_LE+i+1,lineTag+i+1)
        # if the TE is sharp, the last point corresponds to the first one
        i = i+1
        gmsh.model.geo.add_line(point_LE+i, point_TEl, lineTag+i+1)
    lineTag = lineTag+i+1 # store the last 'lineTag' from previous loop
    airfoil_endLineTag = lineTag

    # creation of the offset layer
    structGrid_startLineTag = lineTag + 1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(point_up+i, point_up+i+1,lineTag+i+1)
    lineTag = lineTag+i+1
    structGrid_LE_lineTag = lineTag
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(point_left+i, point_left+i+1,lineTag+i+1)
    lineTag = lineTag+i+1 # store the last 'lineTag' from previous loop
    structGrid_endLineTag = lineTag

    line_A = lineTag+1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(point_TEu+i, point_up+i,lineTag+i+1)
    lineTag = lineTag+i+1
    line_G = lineTag+1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(point_LE+i, point_left+i,lineTag+i+1)
    i = i+1
    gmsh.model.geo.add_line(point_TEl, point_low,lineTag+i+1)
    lineTag = lineTag+i+1
    line_B = lineTag

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

    line_airfoilUp = list(range(airfoil_startLineTag, airfoil_LE_lineTag+1))
    line_airfoilLow = list(range(airfoil_LE_lineTag+1, airfoil_endLineTag+1))

    line_BLup = list(range(structGrid_startLineTag, structGrid_LE_lineTag+1))
    line_BLlow = list(range(structGrid_LE_lineTag+1, structGrid_endLineTag+1))
    line_BLradii = list(range(line_A, line_B+1))

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the surfaces # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # mesh of the airfoil interior (test --> OK)

    line_airfoil = list(range(airfoil_startLineTag, airfoil_endLineTag+1))
    if bluntTrailingEdge: # taking into account of the TE to close the contour
        airfoil_boundaries = [*line_airfoil, line_El, line_Eu]
    else:
        airfoil_boundaries = [*line_airfoil]
    
    gmsh.model.geo.add_curve_loop( airfoil_boundaries, surfaceTag+1) 
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
    if bluntTrailingEdge:
        if (gridPts_alongNACA <= gridPts_inTE):
            print("Warning. A struct mesh cannot be created with this ration of points in the blunt TE and along the airfoil. Choose gridPts_alongNACA/gridPts_inTE > 1")
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [point_TEl, point_TEu, point_LE-gridPts_inTE+2, point_LE+gridPts_inTE-2])
    else:
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [point_TE, point_TE+int(gridPts_alongNACA/2), point_LE, point_LE+int(gridPts_alongNACA/2)])
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_airfoil = surfaceTag

    # (structured / transfinite) boundary layer grid

    for i in range(2*gridPts_alongNACA-2):
        gmsh.model.geo.mesh.setTransfiniteCurve(airfoil_startLineTag+i, 2) # just create one cell between two consecutive NACA gridtPts 
        gmsh.model.geo.mesh.setTransfiniteCurve(structGrid_startLineTag+i, 2) # just create one cell between two consecutive NACA gridtPts 

    for i in range(2*gridPts_alongNACA-1):
        gmsh.model.geo.mesh.setTransfiniteCurve(line_BLradii[i], gridPts_inBL, "Progression", gridGeomProg_inBL)

    BLstructStartSurfTag = surfaceTag+1
    for i in range(2*gridPts_alongNACA-2):
        gmsh.model.geo.add_curve_loop([structGrid_startLineTag+i, -line_BLradii[i+1], -(airfoil_startLineTag+i), line_BLradii[i]], surfaceTag+1)
        gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1 
    BLstructEndSurfTag = surfaceTag

    surf_BLstructGridUp = list(range(BLstructStartSurfTag, BLstructStartSurfTag+gridPts_alongNACA-1))
    surf_BLstructGridLow = list(range(BLstructStartSurfTag+gridPts_alongNACA-1, BLstructEndSurfTag+1))
    surf_BLstructGrid = [*surf_BLstructGridUp, *surf_BLstructGridLow]

    # (structured / transfinite) TEpatch region

    TEpatch_boundariesl = [line_C, line_Br, -line_L, line_B]
    TEpatch_boundariesu = [line_M, line_Ar, line_D, -line_A]

    gmsh.model.geo.add_curve_loop( TEpatch_boundariesl, surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_TEpatchLow = surfaceTag
    
    gmsh.model.geo.add_curve_loop( TEpatch_boundariesu, surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_TEpatchUp = surfaceTag

    if bluntTrailingEdge: 
        gmsh.model.geo.mesh.setTransfiniteCurve(line_El, gridPts_inTE-1, "Progression", gridGeomProg_inBL)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_Eu, gridPts_inTE-1, "Progression", -gridGeomProg_inBL)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_Fl, gridPts_inTE-1)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_Fu, gridPts_inTE-1)

        TEpatch_boundariesml = [line_L, line_Fl, -line_K, -line_El]
        TEpatch_boundariesmu = [line_K, line_Fu, -line_M, -line_Eu]

        gmsh.model.geo.add_curve_loop( TEpatch_boundariesml, surfaceTag+1)
        gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_TEpatchMidLow = surfaceTag

        gmsh.model.geo.add_curve_loop( TEpatch_boundariesmu, surfaceTag+1)
        gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_TEpatchMidUp = surfaceTag
    else:
        surf_TEpatchMidLow = -1
        surf_TEpatchMidUp = -1


    gmsh.model.geo.mesh.setTransfiniteCurve(line_Ar, gridPts_inBL)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_Br, gridPts_inBL)

    gmsh.model.geo.mesh.setTransfiniteCurve(line_C, gridPts_alongTEpatch)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_D, gridPts_alongTEpatch)

    gmsh.model.geo.mesh.setTransfiniteCurve(line_K, gridPts_alongTEpatch,"Progression", gridGeomProg_alongTEpatch)

    if bluntTrailingEdge:
        gmsh.model.geo.mesh.setTransfiniteCurve(line_M, gridPts_alongTEpatch,"Progression", gridGeomProg_alongTEpatch)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_L, gridPts_alongTEpatch,"Progression", gridGeomProg_alongTEpatch)

    # (structured / transfinite) wake region
    
    # if bluntTrailingEdge:
    #     wake_boundaries = [line_H, line_Ber, line_Jl, line_Ju, line_Aer, line_I, -line_Ar, -line_Fu, -line_Fl, -line_Br]
    # else:
    #     wake_boundaries = [line_H, line_Ber, line_Aer, line_I, -line_Ar, -line_Br]
    
    # wake_curveLoop = gmsh.model.geo.add_curve_loop( wake_boundaries, surfaceTag+1)
    # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)        
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [point_lowRight, point_lowFarRight, point_upFarRight, point_upRight])
    # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    # surfaceTag = surfaceTag+1
    # surf_wake = surfaceTag

    # if bluntTrailingEdge:
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_J, 2*(gridPts_inBL+gridPts_inTE)-5) 
    # else:
    #     gmsh.model.geo.mesh.setTransfiniteCurve(line_J, 2*gridPts_inBL-1)


    wake_boundariesl = [line_H, line_Ber, -line_O, -line_Br]
    wake_boundariesu = [line_P, line_Aer, line_I, -line_Ar]

    gmsh.model.geo.add_curve_loop( wake_boundariesl, surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_wakeLow = surfaceTag
    
    gmsh.model.geo.add_curve_loop( wake_boundariesu, surfaceTag+1)
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_wakeUp = surfaceTag

    if bluntTrailingEdge:
        wake_boundariesml = [line_O, line_Jl, -line_N, -line_Fl]
        wake_boundariesmu = [line_N, line_Ju, -line_P, -line_Fu]

        gmsh.model.geo.add_curve_loop( wake_boundariesml, surfaceTag+1)
        gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_wakeMidLow = surfaceTag

        gmsh.model.geo.add_curve_loop( wake_boundariesmu, surfaceTag+1)
        gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_wakeMidUp = surfaceTag
    else:
        surf_wakeMidLow = -1
        surf_wakeMidUp = -1

    gmsh.model.geo.mesh.setTransfiniteCurve(line_Aer, gridPts_inBL)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_Ber, gridPts_inBL)

    if bluntTrailingEdge:
        gmsh.model.geo.mesh.setTransfiniteCurve(line_Jl, gridPts_inTE-1) 
        gmsh.model.geo.mesh.setTransfiniteCurve(line_Ju, gridPts_inTE-1)

    gmsh.model.geo.mesh.setTransfiniteCurve(line_H, gridPts_alongWake,"Progression", gridGeomProg_alongWake)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_I, gridPts_alongWake,"Progression", -gridGeomProg_alongWake)
    gmsh.model.geo.mesh.setTransfiniteCurve(line_N, gridPts_alongWake,"Progression", gridGeomProg_alongWake)
    if bluntTrailingEdge:
        gmsh.model.geo.mesh.setTransfiniteCurve(line_P, gridPts_alongWake,"Progression", gridGeomProg_alongWake)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_O, gridPts_alongWake,"Progression", gridGeomProg_alongWake)


    # # connection to regular CAA domain:

    pointTag_list = [point_LE, point_TE, point_TEu, point_TEl, point_TEwake, point_TEfarWake, point_left, point_up, point_upRight, point_upFarRight, point_low, point_lowRight, point_lowFarRight, point_upMidRight, point_lowMidRight, point_upMidFarRight, point_lowMidFarRight]
    lineTag_list = [[*line_airfoilUp], [*line_airfoilLow], [*line_BLup], [*line_BLlow], [*line_BLradii], line_A, line_B, line_C, line_D, line_Eu, line_El, line_Fu, line_Fl, line_G, line_H, line_I, line_Ju, line_Jl, line_K, line_L, line_M, line_N, line_O, line_P, line_Ar, line_Br, line_Aer, line_Ber]
    surfaceTag_list = [surf_airfoil, [*surf_BLstructGrid], surf_BLstructGridUp, surf_BLstructGridLow, surf_TEpatchUp, surf_TEpatchLow, surf_TEpatchMidUp, surf_TEpatchMidLow, surf_wakeUp, surf_wakeLow, surf_wakeMidUp, surf_wakeMidLow]

    return pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def gmeshed_disk(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec):

    pointTag = structTag[0]
    lineTag = structTag[1]
    surfaceTag = structTag[2]

    rodPos = GeomSpec[0]
    rodR = GeomSpec[1]
    rodBLwidth  = GeomSpec[2]

    gridPts_alongRod = GridPtsSpec[0]
    gridPts_inRodBL = GridPtsSpec[1]
    gridGeomProg_inRodBL = GridPtsSpec[2]

    shiftVec = np.array(shiftVec)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Points # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$

    rotVec = np.matmul(rotMat, np.array([rodPos[0], rodPos[1], rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodCenter = pointTag

    rotVec = np.matmul(rotMat, np.array([rodPos[0]+rodR, rodPos[1], rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodPt1 = pointTag
    rotVec = np.matmul(rotMat, np.array([rodPos[0], rodPos[1]+rodR, rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodPt2 = pointTag
    rotVec = np.matmul(rotMat, np.array([rodPos[0]-rodR, rodPos[1], rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodPt3 = pointTag
    rotVec = np.matmul(rotMat, np.array([rodPos[0], rodPos[1]-rodR, rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodPt4 = pointTag

    rotVec = np.matmul(rotMat, np.array([rodPos[0]+rodR+rodBLwidth, rodPos[1], rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodBLpt1 = pointTag
    rotVec = np.matmul(rotMat, np.array([rodPos[0], rodPos[1]+rodR+rodBLwidth, rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodBLpt2 = pointTag
    rotVec = np.matmul(rotMat, np.array([rodPos[0]-rodR-rodBLwidth, rodPos[1], rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_rodBLpt3 = pointTag
    rotVec = np.matmul(rotMat, np.array([rodPos[0], rodPos[1]-rodR-rodBLwidth, rodPos[2]])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], rodR/10, pointTag+1)
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


    pRod = [point_rodPt1, point_rodPt2, point_rodPt3, point_rodPt4]
    pRodBL = [point_rodBLpt1, point_rodBLpt2, point_rodBLpt3, point_rodBLpt4]
    
    lRodConnect = [line_rodConnect1, line_rodConnect2, line_rodConnect3, line_rodConnect4]
    lRodArc = [line_rodArc1, line_rodArc2, line_rodArc3, line_rodArc4]
    lRodBL = [line_rodBLarc1, line_rodBLarc2, line_rodBLarc3, line_rodBLarc4]

    pointTag_list = [point_rodCenter, [*pRod], [*pRodBL]]
    lineTag_list = [[*lRodConnect], [*lRodArc], [*lRodBL]]
    surfaceTag_list = [surf_rodStruct1, surf_rodStruct2, surf_rodStruct3, surf_rodStruct4]

    return pointTag_list, lineTag_list, surfaceTag_list, pointTag, lineTag, surfaceTag


# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the transverse lines # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def gmeshed_blade_tl(pTS, gridPts_alongNACA, radii_step, bluntTrailingEdge, lineTag):

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

    tlTS = [] # transverse lineTag_struct

    for i in range(len(radii_step)-1):
        # airfoil
        line_TEu = lineTag+1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_line(pTS[i][pTEu]+j,pTS[i+1][pTEu]+j,lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
        line_LE = lineTag +1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_line(pTS[i][pLE]+j,pTS[i+1][pLE]+j,lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
        if bluntTrailingEdge:
            j = j + 1
            gmsh.model.geo.add_line(pTS[i][pLE]+j,pTS[i+1][pLE]+j,lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
            line_TEl = lineTag 
            gmsh.model.geo.add_line(pTS[i][pTE], pTS[i+1][pTE],lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
            line_TE = lineTag
        else:
            line_TEl = line_TEu
            line_TE = line_TEu
        transverseline_airfoilUp = list(range(line_TEu, line_LE+1))
        if bluntTrailingEdge:
            transverseline_airfoilLow = list(range(line_LE, line_TEl+1))
        else:
            transverseline_airfoilLow = [*range(line_LE, line_LE+gridPts_alongNACA-1), line_TEl]
        transverseline_airfoil = [*transverseline_airfoilUp, *transverseline_airfoilLow[1:]]
        # BL
        line_up = lineTag+1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_line(pTS[i][pup]+j,pTS[i+1][pup]+j,lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
        line_left = lineTag +1
        for j in range(0,gridPts_alongNACA):
            gmsh.model.geo.add_line(pTS[i][pleft]+j,pTS[i+1][pleft]+j,lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
        line_low = lineTag
        transverseline_BLup = list(range(line_up, line_left+1))
        transverseline_BLlow = list(range(line_left, line_low+1))
        transverseline_BL = [*transverseline_BLup, *transverseline_BLlow[1:]]

        #TE patch
        gmsh.model.geo.add_line(pTS[i][pTEwake],pTS[i+1][pTEwake],lineTag+1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
        lineTag = lineTag + 1
        line_TEwake = lineTag
        gmsh.model.geo.add_line(pTS[i][pTEfarWake],pTS[i+1][pTEfarWake],lineTag+1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
        lineTag = lineTag + 1
        line_TEfarWake = lineTag
        if bluntTrailingEdge:
            gmsh.model.geo.add_line(pTS[i][pupMidRight], pTS[i+1][pupMidRight],lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
            line_upMidRight = lineTag
            gmsh.model.geo.add_line(pTS[i][plowMidRight], pTS[i+1][plowMidRight],lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
            line_lowMidRight = lineTag
            gmsh.model.geo.add_line(pTS[i][pupMidFarRight], pTS[i+1][pupMidFarRight],lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
            line_upMidFarRight = lineTag
            gmsh.model.geo.add_line(pTS[i][plowMidFarRight], pTS[i+1][plowMidFarRight],lineTag+1)
            gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
            lineTag = lineTag + 1
            line_lowMidFarRight = lineTag
        else:
            line_upMidRight = line_TEwake
            line_lowMidRight = line_TEwake
            line_upMidFarRight = line_TEfarWake
            line_lowMidFarRight = line_TEfarWake

        gmsh.model.geo.add_line(pTS[i][pupRight],pTS[i+1][pupRight],lineTag+1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
        lineTag = lineTag + 1
        line_upRight = lineTag

        gmsh.model.geo.add_line(pTS[i][plowRight],pTS[i+1][plowRight],lineTag+1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
        lineTag = lineTag + 1
        line_lowRight = lineTag

        gmsh.model.geo.add_line(pTS[i][pupFarRight],pTS[i+1][pupFarRight],lineTag+1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
        lineTag = lineTag + 1
        line_upFarRight = lineTag

        gmsh.model.geo.add_line(pTS[i][plowFarRight],pTS[i+1][plowFarRight],lineTag+1)
        gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, radii_step[i])
        lineTag = lineTag + 1
        line_lowFarRight = lineTag

        tlTS.append([[*transverseline_airfoil], [*transverseline_airfoilUp], [*transverseline_airfoilLow], line_TEu, line_LE, line_TEl, line_TE, line_TEwake, line_TEfarWake, [*transverseline_BL], [*transverseline_BLup], [*transverseline_BLlow], line_up, line_left, line_low, line_upRight, line_lowRight, line_upFarRight, line_lowFarRight, line_upMidRight, line_lowMidRight, line_upMidFarRight, line_lowMidFarRight])
    return tlTS, lineTag


# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the transverse surfaces # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def gmeshed_blade_ts(lTS, tlTS, gridPts_alongNACA, radii_step, bluntTrailingEdge, surfaceTag):

    # Tags for easily accessing the list elements

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

    tlairfoil = 0
    tlairfoilUp = 1
    tlairfoilLow = 2
    tlTEu = 3
    tlLE = 4
    tlTEl = 5
    tlTE = 6
    tlTEwake = 7
    tlTEfarWake = 8
    tlBL = 9
    tlBLup = 10
    tlBLlow = 11
    tlup = 12
    tlleft = 13
    tllow = 14
    tlupRight = 15
    tllowRight = 16
    tlupFarRight = 17
    tllowFarRight = 18
    tlupMidRight = 19
    tllowMidRight = 20
    tlupMidFarRight = 21
    tllowMidFarRight = 22

    tsTS = [] # transverse surfaceTag_struct

    for i in range(len(radii_step)-1):

        tsurf_airfoilExtradosTE = surfaceTag+1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_curve_loop( [ lTS[i][lairfoilUp][j], -tlTS[i][tlairfoilUp][j], -lTS[i+1][lairfoilUp][j], tlTS[i][tlairfoilUp][j+1] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
        tsurf_airfoilExtradosLE = surfaceTag
        tsurf_airfoilIntradosLE = surfaceTag+1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_curve_loop( [ lTS[i][lairfoilLow][j], -tlTS[i][tlairfoilLow][j], -lTS[i+1][lairfoilLow][j], tlTS[i][tlairfoilLow][j+1] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
        tsurf_airfoilIntradosTE = surfaceTag
        if bluntTrailingEdge:
            # addSurfaceFilling() only works with surfaces defined by 3 or 4 lines when geo kernel is used.
            # The blunt TE loop: lTS[i][lEl], lTS[i][lEu], tlTS[i][tlTEu], -lTS[i+1][lEu], -lTS[i+1][lEl], -tlTS[i][tlTEl]
            # is thus divided into to regions.
            # # # upper TE region:
            gmsh.model.geo.add_curve_loop( [ lTS[i][lEu], tlTS[i][tlTEu], -lTS[i+1][lEu], -tlTS[i][tlTE] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_airfoilTEu = surfaceTag
            # # # lower TE region:
            gmsh.model.geo.add_curve_loop( [ lTS[i][lEl], tlTS[i][tlTE], -lTS[i+1][lEl], -tlTS[i][tlTEl] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_airfoilTEl = surfaceTag
        else:
            tsurf_airfoilTEu = -1
            tsurf_airfoilTEl = -1

        tsurf_BLextradosTE = surfaceTag+1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_curve_loop( [ lTS[i][lBLup][j], -tlTS[i][tlBLup][j], -lTS[i+1][lBLup][j], tlTS[i][tlBLup][j+1] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
        tsurf_BLextradosLE = surfaceTag
        tsurf_BLintradosLE = surfaceTag+1
        for j in range(0,gridPts_alongNACA-1):
            gmsh.model.geo.add_curve_loop( [ lTS[i][lBLlow][j], -tlTS[i][tlBLlow][j], -lTS[i+1][lBLlow][j], tlTS[i][tlBLlow][j+1] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
        tsurf_BLintradosTE = surfaceTag
    
        tsurf_BLradiExtradosTE = surfaceTag+1
        for j in range(0,2*gridPts_alongNACA-1):
            gmsh.model.geo.add_curve_loop( [ lTS[i][lBLrad][j], - tlTS[i][tlairfoil][j], -lTS[i+1][lBLrad][j], tlTS[i][tlBL][j] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
        tsurf_BLradiIntradosTE = surfaceTag

        # TE patch
        gmsh.model.geo.add_curve_loop( [ lTS[i][lD], tlTS[i][tlup], -lTS[i+1][lD], -tlTS[i][tlupRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineD = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lC], -tlTS[i][tllow], -lTS[i+1][lC], tlTS[i][tllowRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineC = surfaceTag
    
        gmsh.model.geo.add_curve_loop( [ lTS[i][lAr], tlTS[i][tlupRight], -lTS[i+1][lAr], -tlTS[i][tlupMidRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineAr = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lBr], tlTS[i][tllowMidRight], -lTS[i+1][lBr], -tlTS[i][tllowRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineBr = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lAer], tlTS[i][tlupFarRight], -lTS[i+1][lAer], -tlTS[i][tlupMidFarRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineAer = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lBer], tlTS[i][tllowMidFarRight], -lTS[i+1][lBer], -tlTS[i][tllowFarRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineBer = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lK], tlTS[i][tlTEwake], -lTS[i+1][lK], -tlTS[i][tlTE] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineK = surfaceTag

        if bluntTrailingEdge:
            gmsh.model.geo.add_curve_loop( [ lTS[i][lL], tlTS[i][tllowMidRight], -lTS[i+1][lL], -tlTS[i][tlTEl] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineL = surfaceTag

            gmsh.model.geo.add_curve_loop( [ lTS[i][lM], tlTS[i][tlupMidRight], -lTS[i+1][lM], -tlTS[i][tlTEu] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineM = surfaceTag

            gmsh.model.geo.add_curve_loop( [ lTS[i][lFu], tlTS[i][tlupMidRight], -lTS[i+1][lFu], -tlTS[i][tlTEwake] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineFu = surfaceTag

            gmsh.model.geo.add_curve_loop( [ lTS[i][lFl], tlTS[i][tlTEwake], -lTS[i+1][lFl], -tlTS[i][tllowMidRight] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineFl = surfaceTag

        else:
            tsurf_lineL = tsurf_lineK
            tsurf_lineM = tsurf_lineK
            tsurf_lineFu = -1
            tsurf_lineFl = -1

        # wake
        gmsh.model.geo.add_curve_loop( [ lTS[i][lI], tlTS[i][tlupRight], -lTS[i+1][lI], -tlTS[i][tlupFarRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineI = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lH], -tlTS[i][tllowRight], -lTS[i+1][lH], tlTS[i][tllowFarRight] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineH = surfaceTag

        gmsh.model.geo.add_curve_loop( [ lTS[i][lN], tlTS[i][tlTEfarWake], -lTS[i+1][lN], -tlTS[i][tlTEwake] ], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        tsurf_lineN = surfaceTag

        # below the possible reason why hex mesh generation by refining is not working:
        # if bluntTrailingEdge:
        #     gmsh.model.geo.add_curve_loop( [ lTS[i][lBr], lTS[i][lFl], lTS[i][lFu], lTS[i][lAr], tlTS[i][tlupRight], -lTS[i+1][lAr], -lTS[i+1][lFu], -lTS[i+1][lFl], -lTS[i+1][lBr], -tlTS[i][tllowRight] ], surfaceTag+1)
        # else:
        #     gmsh.model.geo.add_curve_loop( [ lTS[i][lBr], lTS[i][lAr], tlTS[i][tlupRight], -lTS[i+1][lAr], -lTS[i+1][lBr], -tlTS[i][tllowRight] ], surfaceTag+1)
        # gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1)
        # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1, "LeftRight", [pTS[i][pupRight], pTS[i+1][pupRight], pTS[i+1][plowRight], pTS[i][plowRight]])
        # gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        # surfaceTag = surfaceTag+1
        # tsurf_lineF = surfaceTag

        if bluntTrailingEdge:
            gmsh.model.geo.add_curve_loop( [ lTS[i][lP], tlTS[i][tlupMidFarRight], -lTS[i+1][lP], -tlTS[i][tlupMidRight] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineP = surfaceTag

            gmsh.model.geo.add_curve_loop( [ lTS[i][lO], tlTS[i][tllowMidFarRight], -lTS[i+1][lO], -tlTS[i][tllowMidRight] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineO = surfaceTag

            gmsh.model.geo.add_curve_loop( [ lTS[i][lJu], tlTS[i][tlupMidFarRight], -lTS[i+1][lJu], -tlTS[i][tlTEfarWake] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineJu = surfaceTag

            gmsh.model.geo.add_curve_loop( [ lTS[i][lJl], tlTS[i][tlTEfarWake], -lTS[i+1][lJl], -tlTS[i][tllowMidFarRight] ], surfaceTag+1)
            gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
            surfaceTag = surfaceTag+1
            tsurf_lineJl = surfaceTag
        else:
            tsurf_lineP = tsurf_lineN
            tsurf_lineO = tsurf_lineN
            tsurf_lineJu = -1
            tsurf_lineJl = -1

        tsurf_airfoil = list(range(tsurf_airfoilExtradosTE, tsurf_airfoilIntradosTE+1))
        tsurf_airfoilExtrados = list(range(tsurf_airfoilExtradosTE, tsurf_airfoilExtradosLE+1))
        tsurf_airfoilIntrados = list(range(tsurf_airfoilIntradosLE, tsurf_airfoilIntradosTE+1))
        if bluntTrailingEdge: # taking into account of the TE to close the contour
            tsurf_airfoilSkin = [*tsurf_airfoil, tsurf_airfoilTEl, tsurf_airfoilTEu]
        else:
            tsurf_airfoilSkin = [*tsurf_airfoil]
        tsurf_airfoilExtIntrados = [*tsurf_airfoilExtrados, *tsurf_airfoilIntrados]

        tsurf_BL = list(range(tsurf_BLextradosTE, tsurf_BLintradosTE+1))
        tsurf_BLextrados = list(range(tsurf_BLextradosTE, tsurf_BLextradosLE+1))
        tsurf_BLintrados = list(range(tsurf_BLintradosLE, tsurf_BLintradosTE+1))
        tsurf_BLradii = list(range(tsurf_BLradiExtradosTE, tsurf_BLradiIntradosTE+1))
        tsurf_BLskin = [*tsurf_BLextrados, *tsurf_BLintrados, tsurf_BLradiExtradosTE, tsurf_BLradiIntradosTE, *tsurf_airfoil]

        tsTS.append([[*tsurf_airfoilSkin], [*tsurf_airfoilExtIntrados], [*tsurf_airfoilExtrados], [*tsurf_airfoilIntrados], tsurf_airfoilTEu, tsurf_airfoilTEl, [*tsurf_BL], [*tsurf_BLextrados], [*tsurf_BLintrados], [*tsurf_BLradii], tsurf_BLradiExtradosTE, tsurf_BLradiIntradosTE, [*tsurf_BLskin], tsurf_lineC, tsurf_lineD, tsurf_lineFu, tsurf_lineFl, tsurf_lineH, tsurf_lineI, tsurf_lineJu, tsurf_lineJl, tsurf_lineK, tsurf_lineL, tsurf_lineM, tsurf_lineN, tsurf_lineO, tsurf_lineP, tsurf_lineAr, tsurf_lineBr, tsurf_lineAer, tsurf_lineBer])
    return tsTS, surfaceTag


# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # creation of the volumes # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def gmeshed_blade_vol(sTS, tsTS, gridPts_alongNACA, radii_step, bluntTrailingEdge, volumeTag):

    # Tags for easily accessing the list elements

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

    tsairfoilSkin = 0
    tsairfoilExtIntrados = 1
    tsairfoilExtrados = 2
    tsairfoilIntrados = 3
    tsairfoilTEu = 4
    tsairfoilTEl = 5
    tsBL = 6
    tsBLextrados = 7
    tsBLintrados = 8
    tsBLrad = 9
    tsBLTEu = 10
    tsBLTEl = 11
    tsBLskin = 12
    tsA = tsBLTEu
    tsB = tsBLTEl
    tsC = 13
    tsD = 14
    tsEu = tsairfoilTEu
    tsEl = tsairfoilTEl
    tsFu = 15
    tsFl = 16
    tsH = 17
    tsI = 18
    tsJu = 19
    tsJl = 20
    tsK = 21
    tsL = 22
    tsM = 23
    tsN = 24
    tsO = 25
    tsP = 26
    tsAr = 27
    tsBr = 28
    tsAer = 29
    tsBer = 30

    # BLskin = []
    # BLskin.extend(sTS[0][sBLstructGrid])
    # BLskin.extend(sTS[-1][sBLstructGrid])
    # for i in range(len(radii_vec)-1):
    #     BLskin.extend([*tsTS[i][tsBL]])
    #     BLskin.extend([*tsTS[i][tsairfoilExtIntrados]])
    #     BLskin.extend([tsTS[i][tsBLrad][0], tsTS[i][tsBLrad][-1]])

    # gmsh.model.geo.addSurfaceLoop([*BLskin], volumeTag+1)
    # gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    # # gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
    # gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    # volumeTag = volumeTag+1
    # BLvolume = volumeTag

    BLstartVolumeTag = volumeTag+1
    for j in range(0,2*gridPts_alongNACA-2):
        for i in range(len(radii_step)-1):
            BLskinElem = []
            BLskinElem.extend([sTS[i][sBLstructGrid][j],sTS[i+1][sBLstructGrid][j]])
            BLskinElem.append(tsTS[i][tsBL][j])
            BLskinElem.append(tsTS[i][tsairfoilExtIntrados][j])
            BLskinElem.extend([tsTS[i][tsBLrad][j], tsTS[i][tsBLrad][j+1]])

            gmsh.model.geo.addSurfaceLoop([*BLskinElem], volumeTag+1)
            gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
            gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
            volumeTag = volumeTag+1
    BLendVolumeTag = volumeTag
    vol_BL = list(range(BLstartVolumeTag, BLendVolumeTag+1))

    wakeUpStartVolumeTag = volumeTag+1
    for i in range(len(radii_step)-1):
        wakeUpVol = []
        wakeUpVol.extend([tsTS[i][tsAr], tsTS[i][tsI], tsTS[i][tsAer], tsTS[i][tsP]])
        wakeUpVol.extend([sTS[i][swakeUp], sTS[i+1][swakeUp]])

        gmsh.model.geo.addSurfaceLoop([*wakeUpVol], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    wakeUpEndVolumeTag = volumeTag
    vol_wakeUp = list(range(wakeUpStartVolumeTag, wakeUpEndVolumeTag+1))

    wakeLowStartVolumeTag = volumeTag+1
    for i in range(len(radii_step)-1):
        wakeLowVol = []
        wakeLowVol.extend([tsTS[i][tsBr], tsTS[i][tsO], tsTS[i][tsBer], tsTS[i][tsH]])
        wakeLowVol.extend([sTS[i][swakeLow], sTS[i+1][swakeLow]])

        gmsh.model.geo.addSurfaceLoop([*wakeLowVol], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    wakeLowEndVolumeTag = volumeTag
    vol_wakeLow = list(range(wakeLowStartVolumeTag, wakeLowEndVolumeTag+1))

    TEpatchUpStartVolumeTag = volumeTag+1
    for i in range(len(radii_step)-1):
        TEpatchUpVol = []
        TEpatchUpVol.extend([tsTS[i][tsA], tsTS[i][tsM], tsTS[i][tsD], tsTS[i][tsAr]])
        TEpatchUpVol.extend([sTS[i][sTEpatchUp], sTS[i+1][sTEpatchUp]])

        gmsh.model.geo.addSurfaceLoop([*TEpatchUpVol], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    TEpatchUpEndVolumeTag = volumeTag
    vol_TEpatchUp = list(range(TEpatchUpStartVolumeTag, TEpatchUpEndVolumeTag+1))

    TEpatchLowStartVolumeTag = volumeTag+1
    for i in range(len(radii_step)-1):
        TEpatchLowVol = []
        TEpatchLowVol.extend([tsTS[i][tsB], tsTS[i][tsL], tsTS[i][tsC], tsTS[i][tsBr]])
        TEpatchLowVol.extend([sTS[i][sTEpatchLow], sTS[i+1][sTEpatchLow]])

        gmsh.model.geo.addSurfaceLoop([*TEpatchLowVol], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    TEpatchLowEndVolumeTag = volumeTag
    vol_TEpatchLow = list(range(TEpatchLowStartVolumeTag, TEpatchLowEndVolumeTag+1))

    if bluntTrailingEdge:
        TEpatchMidUpStartVolumeTag = volumeTag+1
        for i in range(len(radii_step)-1):
            TEpatchMidUpVol = []
            TEpatchMidUpVol.extend([tsTS[i][tsEu], tsTS[i][tsM], tsTS[i][tsFu], tsTS[i][tsK]])
            TEpatchMidUpVol.extend([sTS[i][sTEpatchMidUp], sTS[i+1][sTEpatchMidUp]])

            gmsh.model.geo.addSurfaceLoop([*TEpatchMidUpVol], volumeTag+1)
            gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
            gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
            volumeTag = volumeTag+1
        TEpatchMidUpEndVolumeTag = volumeTag
        vol_TEpatchMidUp = list(range(TEpatchMidUpStartVolumeTag, TEpatchMidUpEndVolumeTag+1))

        TEpatchMidLowStartVolumeTag = volumeTag+1
        for i in range(len(radii_step)-1):
            TEpatchMidLowVol = []
            TEpatchMidLowVol.extend([tsTS[i][tsEl], tsTS[i][tsL], tsTS[i][tsFl], tsTS[i][tsK]])
            TEpatchMidLowVol.extend([sTS[i][sTEpatchMidLow], sTS[i+1][sTEpatchMidLow]])

            gmsh.model.geo.addSurfaceLoop([*TEpatchMidLowVol], volumeTag+1)
            gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
            gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
            volumeTag = volumeTag+1
        TEpatchMidLowEndVolumeTag = volumeTag
        vol_TEpatchMidLow = list(range(TEpatchMidLowStartVolumeTag, TEpatchMidLowEndVolumeTag+1))

        wakeMidUpStartVolumeTag = volumeTag+1
        for i in range(len(radii_step)-1):
            wakeMidUpVol = []
            wakeMidUpVol.extend([tsTS[i][tsFu], tsTS[i][tsP], tsTS[i][tsJu], tsTS[i][tsN]])
            wakeMidUpVol.extend([sTS[i][swakeMidUp], sTS[i+1][swakeMidUp]])

            gmsh.model.geo.addSurfaceLoop([*wakeMidUpVol], volumeTag+1)
            gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
            gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
            volumeTag = volumeTag+1
        wakeMidUpEndVolumeTag = volumeTag
        vol_wakeMidUp = list(range(wakeMidUpStartVolumeTag, wakeMidUpEndVolumeTag+1))

        wakeMidLowStartVolumeTag = volumeTag+1
        for i in range(len(radii_step)-1):
            wakeMidLowVol = []
            wakeMidLowVol.extend([tsTS[i][tsFl], tsTS[i][tsO], tsTS[i][tsJl], tsTS[i][tsN]])
            wakeMidLowVol.extend([sTS[i][swakeMidLow], sTS[i+1][swakeMidLow]])

            gmsh.model.geo.addSurfaceLoop([*wakeMidLowVol], volumeTag+1)
            gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
            gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
            gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
            volumeTag = volumeTag+1
        wakeMidLowEndVolumeTag = volumeTag
        vol_wakeMidLow = list(range(wakeMidLowStartVolumeTag, wakeMidLowEndVolumeTag+1))
    else:
        vol_TEpatchMidUp = [-1]
        vol_TEpatchMidLow = [-1]
        vol_wakeMidUp = [-1]
        vol_wakeMidLow = [-1]

    volumeTag_list = [[*vol_BL], [*vol_wakeUp], [*vol_wakeLow], [*vol_wakeMidUp], [*vol_wakeMidLow], [*vol_TEpatchUp], [*vol_TEpatchLow], [*vol_TEpatchMidUp], [*vol_TEpatchMidLow]]

    return volumeTag_list, volumeTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def gmeshed_bladeTip_vol(sTS_slice, tsTS_tip, gridPts_alongNACA, bluntTrailingEdge, volumeTag):
    
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

    sTip_sliceTS = 0 
    sTip_airfoilSkin = 1
    sTip_airfoilSkinTEu = 2
    sTip_airfoilSkinLEu = 3
    sTip_airfoilSkinLEl = 4
    sTip_airfoilSkinTEl = 5
    sTip_BLstruct = 6
    sTip_BLstructUp = 7
    sTip_BLstructLeftU = 8
    sTip_BLstructLeftL = 9
    sTip_BLstructLow = 10
    tsTip_BLstruct = 11
    tsTip_BLstructUp = 12
    tsTip_BLstructLeftU = 13
    tsTip_BLstructLeftL = 14
    tsTip_BLstructLow = 15
    sTip_LEconnex = 16
    sTip_TEstructGridUpConnex = 17
    sTip_TEpatchMidUpConnex = 18
    sTip_TEpatchUpConnex = 19
    sTip_lDu = 20
    sTip_lDl = 21
    sTip_lMu = 22
    sTip_lMl = 23
    tsTip_TEu = 24
    tsTip_TEl = 25
    tsTip_TEwakeU = 26
    tsTip_TEwakeL = 27
    tsTip_TEpatchUpU = 28
    tsTip_TEpatchUpL = 29

    sTS_tip = tsTS_tip[sTip_sliceTS] # the Tag Struct (TS) of the support surface (s) is stored in the TS of the transverse surfaces (ts) for convenience

    ### tip Struct BL ###
    tipStructBLstartVolumeTag = volumeTag+1
    if bluntTrailingEdge:
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[tsTip_BLstructUp], tsTS_tip[tsTip_BLstructUp]+1, tsTS_tip[sTip_BLstructUp], tsTS_tip[sTip_airfoilSkinTEu], sTS_slice[sBLstructGrid][0], sTS_tip[sBLstructGrid][0]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    else:
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[tsTip_BLstructUp], tsTS_tip[tsTip_BLstructUp]+1, tsTS_tip[sTip_BLstructUp], tsTS_tip[sTip_airfoilSkinTEu], sTS_slice[sBLstructGrid][0], tsTS_tip[sTip_TEstructGridUpConnex]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        # gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) ## not working... By spliting the volume into a pyramid with square basis and a prism with triangular basis, still not working!
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1

    for i in range(1,gridPts_alongNACA-2):
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[tsTip_BLstructUp]+i, tsTS_tip[tsTip_BLstructUp]+1+i, tsTS_tip[sTip_BLstructUp]+i, tsTS_tip[sTip_airfoilSkinTEu]+i, sTS_slice[sBLstructGrid][i], sTS_tip[sBLstructGrid][i]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1

    gmsh.model.geo.addSurfaceLoop([sTS_slice[sBLstructGrid][gridPts_alongNACA-2], tsTS_tip[sTip_LEconnex], tsTS_tip[tsTip_BLstructUp]+gridPts_alongNACA-2, tsTS_tip[sTip_airfoilSkinLEu], tsTS_tip[sTip_BLstructLeftU]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) # gridPts_tipSide needs to be > 2 otherwise this transfinite operation fails !
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1
    gmsh.model.geo.addSurfaceLoop([sTS_slice[sBLstructGrid][gridPts_alongNACA-1], tsTS_tip[sTip_LEconnex], tsTS_tip[tsTip_BLstructUp]+gridPts_alongNACA-1, tsTS_tip[sTip_airfoilSkinLEl], tsTS_tip[sTip_BLstructLeftL]], volumeTag+1)
    gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
    gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) # gridPts_tipSide needs to be > 2 otherwise this transfinite operation fails !
    gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
    volumeTag = volumeTag+1

    for i in range(1,gridPts_alongNACA-2):
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[sTip_BLstructLeftL]+i, tsTS_tip[tsTip_BLstructLeftL]+i-1, tsTS_tip[tsTip_BLstructLeftL]+i, tsTS_tip[sTip_airfoilSkinLEl]+i, sTS_slice[sBLstructGrid][gridPts_alongNACA+i-1], sTS_tip[sBLstructGrid][gridPts_alongNACA-i-2]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1

    if bluntTrailingEdge:
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[tsTip_BLstructLow], tsTS_tip[tsTip_BLstructLow]-1, tsTS_tip[sTip_BLstructLow], tsTS_tip[sTip_airfoilSkinTEl], sTS_slice[sBLstructGrid][-1], sTS_tip[sBLstructGrid][0]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    else:
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[tsTip_BLstructLow], tsTS_tip[tsTip_BLstructLow]-1, tsTS_tip[sTip_BLstructLow], tsTS_tip[sTip_airfoilSkinTEl], sTS_slice[sBLstructGrid][-1], tsTS_tip[sTip_TEstructGridUpConnex]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        # gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1) ## not working... By spliting the volume into a pyramid with square basis and a prism with triangular basis, still not working!
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    tipStructBLendVolumeTag = volumeTag

    tipStructBL = list(range(tipStructBLstartVolumeTag, tipStructBLendVolumeTag+1))

    ### tip TE patch ###
    tipTEpatchStartVolumeTag = volumeTag+1
    if bluntTrailingEdge:
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[sTip_TEpatchMidUpConnex], tsTS_tip[tsTip_TEu], tsTS_tip[tsTip_TEwakeU], tsTS_tip[sTip_lMu], sTS_slice[sTEpatchMidUp]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[sTip_TEpatchMidUpConnex], tsTS_tip[tsTip_TEl], tsTS_tip[tsTip_TEwakeL], tsTS_tip[sTip_lMl], sTS_slice[sTEpatchMidLow]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
        gmsh.model.geo.addSurfaceLoop([sTS_tip[sTEpatchUp], sTS_slice[sTEpatchUp], tsTS_tip[tsTip_BLstructUp], tsTS_tip[tsTip_TEpatchUpU], tsTS_tip[sTip_lDu], tsTS_tip[sTip_lMu]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
        gmsh.model.geo.addSurfaceLoop([sTS_tip[sTEpatchUp], sTS_slice[sTEpatchLow], tsTS_tip[tsTip_BLstructLow], tsTS_tip[tsTip_TEpatchUpL], tsTS_tip[sTip_lDl], tsTS_tip[sTip_lMl]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    else:
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[sTip_lDu], sTS_slice[sTEpatchUp], tsTS_tip[tsTip_TEpatchUpU], tsTS_tip[sTip_TEpatchUpConnex], tsTS_tip[tsTip_BLstructUp]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
        gmsh.model.geo.addSurfaceLoop([tsTS_tip[sTip_lDl], sTS_slice[sTEpatchLow], tsTS_tip[tsTip_TEpatchUpL], tsTS_tip[sTip_TEpatchUpConnex], tsTS_tip[tsTip_BLstructLow]], volumeTag+1)
        gmsh.model.geo.addVolume([volumeTag+1], volumeTag+1)
        gmsh.model.geo.mesh.setTransfiniteVolume(volumeTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_3Dim, volumeTag+1) # To create quadrangles instead of triangles
        volumeTag = volumeTag+1
    tipTEpatchEndVolumeTag = volumeTag

    tipTEpatch = list(range(tipTEpatchStartVolumeTag, tipTEpatchEndVolumeTag+1))


    volumeTag_list = [[*tipStructBL],[*tipTEpatch]]

    return volumeTag_list, volumeTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************


def gmeshed_bladeTip_ts(pTS_slice, lTS_slice, GeomSpec, GridPtsSpec, rotMat, shiftVec, pointTag, lineTag, surfaceTag):

    # GeomSpec = [NACA_type, bluntTrailingEdge, optimisedGridSpacing, pitch_vecAngle[i], chord_vec[i], airfoilReferenceAlongChord_c*chord_vec[i], airfoilReferenceCoordinate[i], height_LE_c*chord_vec[i], height_TE_c*chord_vec[i], TEpatchLength_c*chord_vec[i]*np.cos(pitch_vecAngle[i]*np.pi/180), TEpatchGridFlaringAngle, wakeLength_c*chord_vec[i]*np.cos(pitch_vecAngle[i]*np.pi/180), wakeGridFlaringAngle]
    pitch = GeomSpec[2]
    GeomSpec[0] = '0012' # force 'NACA_type' to be 0012

    structTag = [pointTag, lineTag, surfaceTag]

    # take into account the blade 3D rotation to define the tip connection
    bladeShiftVec = shiftVec
    airfoilReferenceCoordinate = GeomSpec[5]
    shiftVec = -np.matmul(rotMat, np.array([airfoilReferenceCoordinate[0], airfoilReferenceCoordinate[1], -airfoilReferenceCoordinate[2]])) + bladeShiftVec
    GeomSpec[5] = [0.0, 0.0, 0.0] # airfoilReferenceCoordinate = [0.0, 0.0, 0.0]
    # rotMat = np.matmul(rotMat, rotationMatrix([-pitch, -pitch, 90.0])) # angles in degree
    rotMat = np.matmul(rotMat, rotationMatrix([-pitch, pitch, -90.0])) # angles in degree

    [pTL_tip, lTL_tip, sTS_tip, pointTag, lineTag, surfaceTag] = gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat, shiftVec)

    # GeomSpec = ['0012', bluntTrailingEdge, optimisedGridSpacing, pitch, chord, airfoilReferenceAlongChord, airfoilReferenceCoordinate, height_LE, height_TE, TEpatchLength, TEpatchGridFlaringAngle, wakeLength, wakeGridFlaringAngle]
    # GridPtsSpec = [gridPts_alongNACA, gridPts_inBL, gridPts_inTE, gridPts_alongTEpatch, gridPts_alongWake, gridGeomProg_inBL, gridGeomProg_alongTEpatch, gridGeomProg_alongWake]

    bluntTrailingEdge = GeomSpec[1]

    gridPts_alongNACA = GridPtsSpec[0]
    gridPts_inBL = GridPtsSpec[1]
    gridPts_inTE = GridPtsSpec[2]
    gridGeomProg_inBL = GridPtsSpec[5]

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

    gridPts_tipSide = max(gridPts_inTE,3) # enforce this to be able to connect with the propeller geom

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the lines # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$

    # connecting together the skeleton of the new generting airfoil and the last propeller slice
    gmsh.model.geo.add_line(pTL_tip[pLE]-1, pTS_slice[pLE], lineTag+1)
    lineTag = lineTag+1
    line_tipConnectionToLE = lineTag
    gmsh.model.geo.add_line(pTL_tip[pleft]-1, pTS_slice[pleft], lineTag+1)
    lineTag = lineTag+1
    line_tipConnectionToLeft = lineTag

    if not bluntTrailingEdge:
        gmsh.model.geo.add_line(pTS_slice[pTE], pTL_tip[pTEu]+1, lineTag+1)
        lineTag = lineTag+1
        line_tipConnectionToTEalongAirfoil = lineTag

        gmsh.model.geo.add_line(pTS_slice[pTEwake], pTL_tip[pupRight], lineTag+1)
        lineTag = lineTag+1
        line_tipConnectionToUpRight = lineTag

        gmsh.model.geo.add_line(pTS_slice[pTE], pTL_tip[pup], lineTag+1)
        lineTag = lineTag+1
        line_tipConnectionToUp = lineTag

    ### creating the oblique/transverse lines ###
    # airfoil skin
    line_TEuTipU = lineTag+1
    for i in range(gridPts_alongNACA-1):
        if not(bluntTrailingEdge is False and i==0): # to avoid creating a 0 distance line when TE is sharp
            gmsh.model.geo.add_line(pTS_slice[pTEu]+i, pTL_tip[pTEu]+i, lineTag+1) 
        lineTag = lineTag+1
    line_LEtipU = lineTag
    line_LEtipL = lineTag+1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(pTS_slice[pLE]+1+i, pTL_tip[pLE]-1-i, lineTag+1)
        lineTag = lineTag+1
    line_TEuTipL = lineTag

    # BL skin
    line_upTipU = lineTag+1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(pTS_slice[pup]+i, pTL_tip[pup]+i, lineTag+1)
        lineTag = lineTag+1
    line_leftTipU = lineTag
    line_leftTipL = lineTag+1
    for i in range(gridPts_alongNACA-1):
        gmsh.model.geo.add_line(pTS_slice[pleft]+1+i, pTL_tip[pleft]-1-i, lineTag+1)
        lineTag = lineTag+1
    line_upTipL = lineTag

    gmsh.model.geo.add_line(pTS_slice[pupRight], pTL_tip[pupRight], lineTag+1)
    lineTag = lineTag+1
    line_upRightTipU = lineTag
    gmsh.model.geo.add_line(pTS_slice[plowRight], pTL_tip[pupRight], lineTag+1)
    lineTag = lineTag+1
    line_upRightTipL = lineTag

    if bluntTrailingEdge:
        gmsh.model.geo.add_line(pTS_slice[pupMidRight], pTL_tip[pupMidRight], lineTag+1)
        lineTag = lineTag+1
        line_upMidRightTipU = lineTag

        gmsh.model.geo.add_line(pTS_slice[plowMidRight], pTL_tip[pupMidRight], lineTag+1)
        lineTag = lineTag+1
        line_upMidRightTipL = lineTag

        gmsh.model.geo.add_line(pTS_slice[pTE], pTL_tip[pTEu], lineTag+1)
        lineTag = lineTag+1
        line_TEuTipM = lineTag

        gmsh.model.geo.add_line(pTS_slice[pTEwake], pTL_tip[pupMidRight], lineTag+1)
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
    gmsh.model.geo.add_curve_loop([-line_tipConnectionToLE, -lTS_slice[lG], line_tipConnectionToLeft, (lTL_tip[lG]-1)], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_tipLEconnectionStructGridUp = surfaceTag
    # TE:
    if bluntTrailingEdge:
        gmsh.model.geo.add_curve_loop([line_TEuTipM, lTL_tip[lM], -line_upMidRightTipM, -lTS_slice[lK]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEconnectionTEpatchMidUp = surfaceTag

        surf_tipTEconnectionStructGridUp = -1
        surf_tipTEconnectionTEpatch = -1
    else:
        gmsh.model.geo.add_curve_loop([line_tipConnectionToTEalongAirfoil, lTL_tip[lBLrad][1], -lTL_tip[lBLup][0], -line_tipConnectionToUp], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEconnectionStructGridUp = surfaceTag

        gmsh.model.geo.add_curve_loop([-line_tipConnectionToUp, lTS_slice[lK], line_tipConnectionToUpRight, lTL_tip[lD]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEconnectionTEpatch = surfaceTag

        surf_tipTEconnectionTEpatchMidUp = -1

    ### airfoil skin ###
    # airfoil tipSkin uper side
    airfoilStructStartSurfTag_tipU = surfaceTag+1
    if bluntTrailingEdge:
        gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][0], line_TEuTipU, -lTS_slice[lairfoilUp][0], -(line_TEuTipU+1)], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    else:
        gmsh.model.geo.add_curve_loop([-lTS_slice[lairfoilUp][0], -(line_TEuTipU+1), line_tipConnectionToTEalongAirfoil], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1

    for i in range(1,gridPts_alongNACA-2):
        gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][i], line_TEuTipU+i, -lTS_slice[lairfoilUp][i], -(line_TEuTipU+i+1)], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    gmsh.model.geo.add_curve_loop([line_tipConnectionToLE, (line_TEuTipU+gridPts_alongNACA-2), -lTS_slice[lairfoilUp][gridPts_alongNACA-2]], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    airfoilStructEndSurfTag_tipU = surfaceTag

    # airfoil tipSkin lower side
    airfoilStructStartSurfTag_tipL = surfaceTag+1
    gmsh.model.geo.add_curve_loop([-line_tipConnectionToLE, -lTS_slice[lairfoilLow][0], -line_LEtipL], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    for i in range(1,gridPts_alongNACA-2):
        gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][-i-1], -(line_LEtipL+i-1), lTS_slice[lairfoilLow][i], (line_LEtipL+i)], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    if bluntTrailingEdge:
        gmsh.model.geo.add_curve_loop([lTL_tip[lairfoilUp][-gridPts_alongNACA+1], -(line_LEtipL+gridPts_alongNACA-3), lTS_slice[lairfoilLow][gridPts_alongNACA-2], (line_LEtipL+gridPts_alongNACA-2)], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    else:
        gmsh.model.geo.add_curve_loop([-line_tipConnectionToTEalongAirfoil, -lTS_slice[lairfoilLow][-1], (line_LEtipL+gridPts_alongNACA-3)], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    airfoilStructEndSurfTag_tipL = surfaceTag

    ### BL skin ###
    # airfoil tipSkin uper side
    BLstructStartSurfTag_tipU = surfaceTag+1
    for i in range(gridPts_alongNACA-2):
        gmsh.model.geo.add_curve_loop([line_upTipU+i, lTL_tip[lBLup][i], -(line_upTipU+1+i), -lTS_slice[lBLup][i]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    # handling the two triangular surfaces appearing at the LE   
    gmsh.model.geo.add_curve_loop([lTS_slice[lBLup][gridPts_alongNACA-2], -line_leftTipU, -line_tipConnectionToLeft], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    BLstructEndSurfTag_tipU = surfaceTag
    BLstructStartSurfTag_tipL = surfaceTag +1
    gmsh.model.geo.add_curve_loop([lTS_slice[lBLlow][0], line_leftTipL, line_tipConnectionToLeft], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    # airfoil tipSkin lower side
    for i in range(1,gridPts_alongNACA-1):
        gmsh.model.geo.add_curve_loop([line_leftTipL+i-1, -lTL_tip[lBLup][-i-1], -(line_leftTipL+i), -lTS_slice[lBLlow][i]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    BLstructEndSurfTag_tipL = surfaceTag

    ### TE patch ###
    gmsh.model.geo.add_curve_loop([-lTS_slice[lD], line_upRightTipU, lTL_tip[lD], -line_upTipU], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    lDsurf_tipU = surfaceTag

    gmsh.model.geo.add_curve_loop([lTS_slice[lC], line_upRightTipL, lTL_tip[lD], -line_upTipL], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    lDsurf_tipL = surfaceTag

    if bluntTrailingEdge:
        gmsh.model.geo.add_curve_loop([line_TEuTipU, lTL_tip[lM], -line_upMidRightTipU, -lTS_slice[lM]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        lMsurf_tipU = surfaceTag

        gmsh.model.geo.add_curve_loop([line_TEuTipL, lTL_tip[lM], -line_upMidRightTipL, -lTS_slice[lL]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        lMsurf_tipL = surfaceTag
    else:
        lMsurf_tipU = -1
        lMsurf_tipL = -1

    ### Generating transverse surfaces ###
    BLstructStartTransverseSurfTag_tipU = surfaceTag+1
    if bluntTrailingEdge:
        gmsh.model.geo.add_curve_loop([line_TEuTipU, lTL_tip[lBLrad][0], -line_upTipU, -lTS_slice[lBLrad][0]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    else:
        gmsh.model.geo.add_curve_loop([line_tipConnectionToUp, -line_upTipU, -lTS_slice[lBLrad][0]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    for i in range(1,gridPts_alongNACA-1):
        gmsh.model.geo.add_curve_loop([line_TEuTipU+i, lTL_tip[lBLrad][i], -(line_upTipU+i), -lTS_slice[lBLrad][i]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    BLstructEndTransverseSurfTag_tipU = surfaceTag
    BLstructStartTransverseSurfTag_tipL = surfaceTag +1
    for i in range(1,gridPts_alongNACA-1):
        gmsh.model.geo.add_curve_loop([line_LEtipL+i-1, lTL_tip[lBLrad][gridPts_alongNACA-1-i], -(line_leftTipL+i-1), -lTS_slice[lBLrad][gridPts_alongNACA-1+i]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    if bluntTrailingEdge:
        gmsh.model.geo.add_curve_loop([line_LEtipL+gridPts_alongNACA-2, lTL_tip[lBLrad][0], -(line_leftTipL+gridPts_alongNACA-2), -lTS_slice[lBLrad][2*gridPts_alongNACA-2]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    else:
        gmsh.model.geo.add_curve_loop([line_tipConnectionToUp, -line_upTipL, -lTS_slice[lBLrad][-1]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
    BLstructEndTransverseSurfTag_tipL = surfaceTag

    ### TE patch ###
    if bluntTrailingEdge:
        # TE patch Mid Up
        gmsh.model.geo.add_curve_loop([-line_TEuTipM, lTS_slice[lEu], line_TEuTipU], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEtransverseConnectionU = surfaceTag
        gmsh.model.geo.add_curve_loop([-line_TEuTipM, -lTS_slice[lEl], line_TEuTipL], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEtransverseConnectionL = surfaceTag
        gmsh.model.geo.add_curve_loop([-line_upMidRightTipM, lTS_slice[lFu], line_upMidRightTipU], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEwakeTransverseConnectionU = surfaceTag
        gmsh.model.geo.add_curve_loop([-line_upMidRightTipM, -lTS_slice[lFl], line_upMidRightTipL], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEwakeTransverseConnectionL = surfaceTag

        # TE patch Up
        gmsh.model.geo.add_curve_loop([line_upMidRightTipU, lTL_tip[lAr], -line_upRightTipU, -lTS_slice[lAr]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEpatchUpTransverseConnectionU = surfaceTag
        gmsh.model.geo.add_curve_loop([line_upMidRightTipL, lTL_tip[lAr], -line_upRightTipL, lTS_slice[lBr]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEpatchUpTransverseConnectionL = surfaceTag
    else:
        gmsh.model.geo.add_curve_loop([-line_tipConnectionToUpRight, line_upRightTipU, lTS_slice[lAr]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEpatchUpTransverseConnectionU = surfaceTag
        gmsh.model.geo.add_curve_loop([-line_tipConnectionToUpRight, line_upRightTipL, -lTS_slice[lBr]], surfaceTag+1)
        gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
        gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
        surfaceTag = surfaceTag+1
        surf_tipTEpatchUpTransverseConnectionL = surfaceTag

        surf_tipTEtransverseConnectionU = -1
        surf_tipTEtransverseConnectionL = -1
        surf_tipTEwakeTransverseConnectionU = -1
        surf_tipTEwakeTransverseConnectionL = -1

    if bluntTrailingEdge:
        surf_airfoilTip = [*list(range(airfoilStructStartSurfTag_tipU, airfoilStructEndSurfTag_tipU+1)), *list(range(airfoilStructStartSurfTag_tipL, airfoilStructEndSurfTag_tipL+1)), surf_tipTEtransverseConnectionU, surf_tipTEtransverseConnectionL]
    else:
        surf_airfoilTip = [*list(range(airfoilStructStartSurfTag_tipU, airfoilStructEndSurfTag_tipU+1)), *list(range(airfoilStructStartSurfTag_tipL, airfoilStructEndSurfTag_tipL+1))]

    surf_BLtip = [*list(range(BLstructStartSurfTag_tipU, BLstructEndSurfTag_tipU+1)), *list(range(BLstructStartSurfTag_tipL, BLstructEndSurfTag_tipL+1))]

    tsurf_BLtip = [*list(range(BLstructStartTransverseSurfTag_tipU, BLstructEndTransverseSurfTag_tipU+1)), *list(range(BLstructStartTransverseSurfTag_tipL, BLstructEndTransverseSurfTag_tipL+1))]


    surfaceTag_list = [[*sTS_tip], # the Tag Struct (TS) of the support surface (s) is stored in the TS of the transverse surfaces (ts) for convenience
                       [*surf_airfoilTip], airfoilStructStartSurfTag_tipU, airfoilStructEndSurfTag_tipU, airfoilStructStartSurfTag_tipL, airfoilStructEndSurfTag_tipL,
                       [*surf_BLtip], BLstructStartSurfTag_tipU, BLstructEndSurfTag_tipU, BLstructStartSurfTag_tipL, BLstructEndSurfTag_tipL,
                       [*tsurf_BLtip], BLstructStartTransverseSurfTag_tipU, BLstructEndTransverseSurfTag_tipU, BLstructStartTransverseSurfTag_tipL, BLstructEndTransverseSurfTag_tipL,
                       surf_tipLEconnectionStructGridUp, surf_tipTEconnectionStructGridUp, surf_tipTEconnectionTEpatchMidUp, surf_tipTEconnectionTEpatch,
                       lDsurf_tipU, lDsurf_tipL, lMsurf_tipU, lMsurf_tipL,
                       surf_tipTEtransverseConnectionU, surf_tipTEtransverseConnectionL, surf_tipTEwakeTransverseConnectionU, surf_tipTEwakeTransverseConnectionL,
                       surf_tipTEpatchUpTransverseConnectionU, surf_tipTEpatchUpTransverseConnectionL]
    
    return surfaceTag_list, pointTag, lineTag, surfaceTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def returnAirfoilContour(lTS, bluntTrailingEdge):

    # Tags for easily accessing the list elements

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

    #               ~       ~       ~

    lAirfoilContour = []
    lAirfoilContour.extend([*lTS[lairfoilUp], *lTS[lairfoilLow]])
    if bluntTrailingEdge:
        lAirfoilContour.extend([lTS[lEl], lTS[lEu]])

    return lAirfoilContour


def returnStructGridOuterContour(lTS, bluntTrailingEdge):

    # Tags for easily accessing the list elements

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

    #               ~       ~       ~

    lStructGridContour = []
    lStructGridContour.extend([lTS[lAer], lTS[lI], lTS[lD], *lTS[lBLup], *lTS[lBLlow], lTS[lC], lTS[lH], lTS[lBer]])
    if bluntTrailingEdge:
        lStructGridContour.extend([lTS[lJl], lTS[lJu]])

    return lStructGridContour

def returnStructGridSide(sTS, bluntTrailingEdge):

    # Tags for easily accessing the list elements

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

    #               ~       ~       ~

    if bluntTrailingEdge: # Watch out for the order of the list outputed.. When creating extrusion and periodic BC, patch need to correspond !
        sStructGridSide = [*sTS[sBLstructGrid], sTS[sTEpatchUp], sTS[sTEpatchLow], sTS[sTEpatchMidUp], sTS[sTEpatchMidLow], sTS[swakeUp], sTS[swakeLow], sTS[swakeMidUp], sTS[swakeMidLow]]
    else:
        sStructGridSide = [*sTS[sBLstructGrid], sTS[sTEpatchUp], sTS[sTEpatchLow], sTS[swakeUp], sTS[swakeLow] ]

    return sStructGridSide

def returnStructGridOuterShell_withoutTip(sTS, tsTS, radii_step, bluntTrailingEdge):
    
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

    tsairfoilSkin = 0
    tsairfoilExtIntrados = 1
    tsairfoilExtrados = 2
    tsairfoilIntrados = 3
    tsairfoilTEu = 4
    tsairfoilTEl = 5
    tsBL = 6
    tsBLextrados = 7
    tsBLintrados = 8
    tsBLrad = 9
    tsBLTEu = 10
    tsBLTEl = 11
    tsBLskin = 12
    tsA = tsBLTEu
    tsB = tsBLTEl
    tsC = 13
    tsD = 14
    tsEu = tsairfoilTEu
    tsEl = tsairfoilTEl
    tsFu = 15
    tsFl = 16
    tsH = 17
    tsI = 18
    tsJu = 19
    tsJl = 20
    tsK = 21
    tsL = 22
    tsM = 23
    tsN = 24
    tsO = 25
    tsP = 26
    tsAr = 27
    tsBr = 28
    tsAer = 29
    tsBer = 30

    #               ~       ~       ~

    sStructGridSkin = []
    sStructGridSkin.extend(sTS[0][sBLstructGrid])
    sStructGridSkin.append(sTS[0][sairfoil])
    sStructGridSkin.extend([sTS[0][swakeUp], sTS[0][swakeLow]])
    sStructGridSkin.extend([sTS[0][sTEpatchUp], sTS[0][sTEpatchLow]])
    sStructGridSkin.extend(sTS[-1][sBLstructGrid])
    sStructGridSkin.append(sTS[-1][sairfoil])
    sStructGridSkin.extend([sTS[-1][swakeUp], sTS[-1][swakeLow]])
    sStructGridSkin.extend([sTS[-1][sTEpatchUp], sTS[-1][sTEpatchLow]])
    if bluntTrailingEdge:
        sStructGridSkin.extend([sTS[0][sTEpatchMidUp], sTS[0][sTEpatchMidLow]])
        sStructGridSkin.extend([sTS[-1][sTEpatchMidUp], sTS[-1][sTEpatchMidLow]])
        sStructGridSkin.extend([sTS[0][swakeMidUp], sTS[0][swakeMidLow]])
        sStructGridSkin.extend([sTS[-1][swakeMidUp], sTS[-1][swakeMidLow]])
    for i in range(len(radii_step)-1):
        sStructGridSkin.extend(tsTS[i][tsBL])
        sStructGridSkin.append(tsTS[i][tsC])
        sStructGridSkin.append(tsTS[i][tsD])
        sStructGridSkin.append(tsTS[i][tsH])
        sStructGridSkin.append(tsTS[i][tsI])
        sStructGridSkin.append(tsTS[i][tsAer])
        sStructGridSkin.append(tsTS[i][tsBer])
        if bluntTrailingEdge:
            sStructGridSkin.append(tsTS[i][tsJu])
            sStructGridSkin.append(tsTS[i][tsJl])
    
    #               ~       ~       ~

    sAirfoilSkin = []
    sAirfoilSkin.append(sTS[0][sairfoil])
    sAirfoilSkin.append(sTS[-1][sairfoil])
    for i in range(len(radii_step)-1):
                sAirfoilSkin.extend(tsTS[i][tsairfoilSkin])

    return sStructGridSkin, sAirfoilSkin

def returnStructGridOuterShell(sTS, tsTS, tsTS_tip, radii_step, bluntTrailingEdge):
    
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

    tsairfoilSkin = 0
    tsairfoilExtIntrados = 1
    tsairfoilExtrados = 2
    tsairfoilIntrados = 3
    tsairfoilTEu = 4
    tsairfoilTEl = 5
    tsBL = 6
    tsBLextrados = 7
    tsBLintrados = 8
    tsBLrad = 9
    tsBLTEu = 10
    tsBLTEl = 11
    tsBLskin = 12
    tsA = tsBLTEu
    tsB = tsBLTEl
    tsC = 13
    tsD = 14
    tsEu = tsairfoilTEu
    tsEl = tsairfoilTEl
    tsFu = 15
    tsFl = 16
    tsH = 17
    tsI = 18
    tsJu = 19
    tsJl = 20
    tsK = 21
    tsL = 22
    tsM = 23
    tsN = 24
    tsO = 25
    tsP = 26
    tsAr = 27
    tsBr = 28
    tsAer = 29
    tsBer = 30

    sTip_sliceTS = 0 
    sTip_airfoilSkin = 1
    sTip_airfoilSkinTEu = 2
    sTip_airfoilSkinLEu = 3
    sTip_airfoilSkinLEl = 4
    sTip_airfoilSkinTEl = 5
    sTip_BLstruct = 6
    sTip_BLstructUp = 7
    sTip_BLstructLeftU = 8
    sTip_BLstructLeftL = 9
    sTip_BLstructLow = 10
    tsTip_BLstruct = 11
    tsTip_BLstructUp = 12
    tsTip_BLstructLeftU = 13
    tsTip_BLstructLeftL = 14
    tsTip_BLstructLow = 15
    sTip_LEconnex = 16
    sTip_TEstructGridUpConnex = 17
    sTip_TEpatchMidUpConnex = 18
    sTip_TEpatchUpConnex = 19
    sTip_lDu = 20
    sTip_lDl = 21
    sTip_lMu = 22
    sTip_lMl = 23
    tsTip_TEu = 24
    tsTip_TEl = 25
    tsTip_TEwakeU = 26
    tsTip_TEwakeL = 27
    tsTip_TEpatchUpU = 28
    tsTip_TEpatchUpL = 29

    #               ~       ~       ~

    sStructGridSkin = []
    sStructGridSkin.extend(sTS[0][sBLstructGrid])
    sStructGridSkin.append(sTS[0][sairfoil])
    sStructGridSkin.extend([sTS[0][swakeUp], sTS[0][swakeLow]])
    sStructGridSkin.extend([sTS[0][sTEpatchUp], sTS[0][sTEpatchLow]])
    sStructGridSkin.extend(tsTS_tip[sTip_BLstruct])
    sStructGridSkin.extend([tsTS_tip[sTip_lDu], tsTS_tip[sTip_lDl]])
    sStructGridSkin.extend([tsTS_tip[tsTip_TEpatchUpU], tsTS_tip[tsTip_TEpatchUpL]])
    sStructGridSkin.extend([sTS[-1][swakeUp], sTS[-1][swakeLow]])
    if bluntTrailingEdge:
        sStructGridSkin.extend([sTS[0][sTEpatchMidUp], sTS[0][sTEpatchMidLow]])
        sStructGridSkin.extend([sTS[0][swakeMidUp], sTS[0][swakeMidLow]])
        sStructGridSkin.extend([sTS[-1][swakeMidUp], sTS[-1][swakeMidLow]])
        sStructGridSkin.extend([tsTS_tip[tsTip_TEwakeU], tsTS_tip[tsTip_TEwakeL]])
    for i in range(len(radii_step)-1):
        sStructGridSkin.extend(tsTS[i][tsBL])
        sStructGridSkin.append(tsTS[i][tsC])
        sStructGridSkin.append(tsTS[i][tsD])
        sStructGridSkin.append(tsTS[i][tsH])
        sStructGridSkin.append(tsTS[i][tsI])
        sStructGridSkin.append(tsTS[i][tsAer])
        sStructGridSkin.append(tsTS[i][tsBer])
        if bluntTrailingEdge:
            sStructGridSkin.append(tsTS[i][tsJu])
            sStructGridSkin.append(tsTS[i][tsJl])
    
    #               ~       ~       ~

    sAirfoilSkin = []
    sAirfoilSkin.append(sTS[0][sairfoil])
    sAirfoilSkin.extend(tsTS_tip[sTip_airfoilSkin])
    for i in range(len(radii_step)-1):
                sAirfoilSkin.extend(tsTS[i][tsairfoilSkin])

    return sStructGridSkin, sAirfoilSkin

def returnStructGridVol_withoutTip(vTS, bluntTrailingEdge):

    vBL = 0
    vwakeUp = 1
    vwakeLow = 2
    vwakeMidUp = 3
    vwakeMidLow = 4
    vTEup = 5
    vTElow = 6
    vTEmidUp = 7
    vTEmidLow = 8

    returnStructGridVol = []
    returnStructGridVol.extend([*vTS[vBL], *vTS[vwakeUp], *vTS[vwakeLow], *vTS[vTEup], *vTS[vTElow]])

    if bluntTrailingEdge:
        returnStructGridVol.extend([*vTS[vTEmidUp], *vTS[vTEmidLow]])
        returnStructGridVol.extend([*vTS[vwakeMidUp], *vTS[vwakeMidLow]])

    return returnStructGridVol

def returnStructGridVol(vTS, vTS_tip, bluntTrailingEdge):

    vBL = 0
    vwakeUp = 1
    vwakeLow = 2
    vwakeMidUp = 3
    vwakeMidLow = 4
    vTEup = 5
    vTElow = 6
    vTEmidUp = 7
    vTEmidLow = 8

    vTipBL = 0
    vTipPatch = 1

    returnStructGridVol = []
    returnStructGridVol.extend([*vTS[vBL], *vTS[vwakeUp], *vTS[vwakeLow], *vTS[vTEup], *vTS[vTElow]])
    if bluntTrailingEdge:
        returnStructGridVol.extend([*vTS[vTEmidUp], *vTS[vTEmidLow]])
        returnStructGridVol.extend([*vTS[vwakeMidUp], *vTS[vwakeMidLow]])

    returnStructGridVol.extend([*vTS_tip[vTipBL], *vTS_tip[vTipPatch]])

    return returnStructGridVol



# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************


def gmeshed_cylinder_surf(y_min_cyl, y_max_cyl, r_cyl, elemSize_cyl, pointTag, lineTag, surfaceTag):

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Points # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$

    gmsh.model.geo.addPoint(0.0, y_max_cyl, 0.0, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylTopCenter = pointTag

    gmsh.model.geo.addPoint(r_cyl, y_max_cyl, 0.0, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylTopPt1 = pointTag
    gmsh.model.geo.addPoint(0.0, y_max_cyl, r_cyl, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylTopPt2 = pointTag
    gmsh.model.geo.addPoint(-r_cyl, y_max_cyl, 0.0, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylTopPt3 = pointTag
    gmsh.model.geo.addPoint(0.0, y_max_cyl, -r_cyl, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylTopPt4 = pointTag

    gmsh.model.geo.addPoint(0.0, y_min_cyl, 0.0, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylBotCenter = pointTag

    gmsh.model.geo.addPoint(r_cyl, y_min_cyl, 0.0, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylBotPt1 = pointTag
    gmsh.model.geo.addPoint(0.0, y_min_cyl, r_cyl, r_cyl/10,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylBotPt2 = pointTag
    gmsh.model.geo.addPoint(-r_cyl, y_min_cyl, 0.0, r_cyl/100,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylBotPt3 = pointTag
    gmsh.model.geo.addPoint(0.0, y_min_cyl, -r_cyl, r_cyl/100,pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_cylBotPt4 = pointTag

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Lines # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$

    gmsh.model.geo.add_line(point_cylBotPt1, point_cylTopPt1,lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((y_max_cyl-y_min_cyl)/elemSize_cyl))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylEdgePlusX = lineTag

    gmsh.model.geo.add_line(point_cylBotPt2, point_cylTopPt2,lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((y_max_cyl-y_min_cyl)/elemSize_cyl))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylEdgePlusZ = lineTag

    gmsh.model.geo.add_line(point_cylBotPt3, point_cylTopPt3,lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((y_max_cyl-y_min_cyl)/elemSize_cyl))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylEdgeMinusX = lineTag

    gmsh.model.geo.add_line(point_cylBotPt4, point_cylTopPt4,lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((y_max_cyl-y_min_cyl)/elemSize_cyl))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylEdgeMinusZ = lineTag

    gmsh.model.geo.addCircleArc(point_cylTopPt1, point_cylTopCenter, point_cylTopPt2, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylTopArc1 = lineTag
    gmsh.model.geo.addCircleArc(point_cylTopPt2, point_cylTopCenter, point_cylTopPt3, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylTopArc2 = lineTag
    gmsh.model.geo.addCircleArc(point_cylTopPt3, point_cylTopCenter, point_cylTopPt4, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylTopArc3 = lineTag
    gmsh.model.geo.addCircleArc(point_cylTopPt4, point_cylTopCenter, point_cylTopPt1, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylTopArc4 = lineTag

    gmsh.model.geo.addCircleArc(point_cylBotPt1, point_cylBotCenter, point_cylBotPt2, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylBotArc1 = lineTag
    gmsh.model.geo.addCircleArc(point_cylBotPt2, point_cylBotCenter, point_cylBotPt3, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylBotArc2 = lineTag
    gmsh.model.geo.addCircleArc(point_cylBotPt3, point_cylBotCenter, point_cylBotPt4, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylBotArc3 = lineTag
    gmsh.model.geo.addCircleArc(point_cylBotPt4, point_cylBotCenter, point_cylBotPt1, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int(2*np.pi*r_cyl/elemSize_cyl/4))
    lineTag = lineTag+1 # store the last 'lineTag' from previous loop
    line_cylBotArc4 = lineTag

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Surfaces # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    gmsh.model.geo.add_curve_loop([line_cylBotArc1, line_cylEdgePlusZ, -line_cylTopArc1, -line_cylEdgePlusX], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1 
    surf_cylSide1 = surfaceTag

    gmsh.model.geo.add_curve_loop([line_cylBotArc2, line_cylEdgeMinusX, -line_cylTopArc2, -line_cylEdgePlusZ], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1 
    surf_cylSide2 = surfaceTag

    gmsh.model.geo.add_curve_loop([line_cylBotArc3, line_cylEdgeMinusZ, -line_cylTopArc3, -line_cylEdgeMinusX], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1 
    surf_cylSide3 = surfaceTag

    gmsh.model.geo.add_curve_loop([line_cylBotArc4, line_cylEdgePlusX, -line_cylTopArc4, -line_cylEdgeMinusZ], surfaceTag+1)
    gmsh.model.geo.addSurfaceFilling([surfaceTag+1], surfaceTag+1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1 
    surf_cylSide4 = surfaceTag

    cylTopEdges = [line_cylTopArc1, line_cylTopArc2, line_cylTopArc3, line_cylTopArc4]
    gmsh.model.geo.add_curve_loop( [*cylTopEdges], surfaceTag+1) 
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_cylTop = surfaceTag

    cylBotEdges = [line_cylBotArc1, line_cylBotArc2, line_cylBotArc3, line_cylBotArc4]
    gmsh.model.geo.add_curve_loop( [*cylBotEdges], surfaceTag+1) 
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
    # gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_cylBot = surfaceTag

    return [surf_cylSide1, surf_cylSide2, surf_cylSide3, surf_cylSide4, surf_cylTop, surf_cylBot], pointTag, lineTag, surfaceTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag, rotMat, shiftVec):

    shiftVec = np.array(shiftVec)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Points # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$

    rotVec = np.matmul(rotMat, np.array([x_min, y_min, 0.0])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], elemSize_rect, pointTag+1)
    pointTag = pointTag+1 
    point_SW = pointTag
    rotVec = np.matmul(rotMat, np.array([x_max, y_min, 0.0])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], elemSize_rect, pointTag+1)
    pointTag = pointTag+1
    point_SE = pointTag
    rotVec = np.matmul(rotMat, np.array([x_max, y_max, 0.0])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], elemSize_rect, pointTag+1)
    pointTag = pointTag+1
    point_NE = pointTag
    rotVec = np.matmul(rotMat, np.array([x_min, y_max, 0.0])) + shiftVec
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], elemSize_rect, pointTag+1)
    pointTag = pointTag+1
    point_NW = pointTag

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Lines # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$

    gmsh.model.geo.add_line(point_SW, point_SE,lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((x_max-x_min)/elemSize_rect))
    lineTag = lineTag+1
    line_S = lineTag
    gmsh.model.geo.add_line(point_SE, point_NE, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((y_max-y_min)/elemSize_rect))
    lineTag = lineTag+1
    line_E = lineTag
    gmsh.model.geo.add_line(point_NE, point_NW, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((x_max-x_min)/elemSize_rect))
    lineTag = lineTag+1
    line_N = lineTag
    gmsh.model.geo.add_line(point_NW, point_SW, lineTag+1)
    gmsh.model.geo.mesh.setTransfiniteCurve(lineTag+1, int((y_max-y_min)/elemSize_rect))
    lineTag = lineTag+1
    line_W = lineTag

    return [line_S, line_E, line_N, line_W], pointTag, lineTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def extrude_rodBL(sTL_rod, span, gridPts_alongSpan):

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

    return ExtrudRodBL_vol, ExtrudRodBL_symFace, ExtrudRodBL_skin

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def extrude_airfoilStruct(sTL_airfoil, bluntTrailingEdge, gridPts_alongNACA, span, gridPts_alongSpan):

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


    return ExtrudAirfoildStruct_vol, ExtrudStructAirfoil_symFace, ExtrudStructAirfoil_skin


# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def extrude_unstructCFD(surf_unstructCFD, span, gridPts_alongSpan):

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

    return ExtrudUnstructCFD_vol, ExtrudUnstructCFD_symFace

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def extrude_unstructBUFF(surf_unstructBUFF, span, gridPts_alongSpan):

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
    
    return ExtrudUnstructBUFF_vol, ExtrudUnstructBUFF_symFace, ExtrudUnstructBUFF_innerSkin, [ExtrudUnstructBUFF_inlet, ExtrudUnstructBUFF_bottom, ExtrudUnstructBUFF_outlet, ExtrudUnstructBUFF_top]


# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def countDOF():
    # Get the number of elements to uniquely tag the mesh:
    # ref: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorials/python/x1.py
    entities = gmsh.model.getEntities()

    nodePerEntity = []
    elemPerEntity = []
    for e in entities:
        # Dimension and tag of the entity:
        dim = e[0]
        tag = e[1]
        # Get the mesh nodes for the entity (dim, tag):
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)
        # Get the mesh elements for the entity (dim, tag):
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        # * Number of mesh nodes and elements:
        numElem = sum(len(i) for i in elemTags)
        nodePerEntity.append(len(nodeTags))
        elemPerEntity.append(numElem)
    print(" - Mesh has " + str(sum(nodePerEntity)) + " nodes and " + str(sum(elemPerEntity)) +
           " elements")

    return nodePerEntity, elemPerEntity

import re # Zhicheng: use regex to replace ',' and '\t' with space
def read_geometry(geometry_file):
    with open(geometry_file, "r") as datafile:
        radii_vec = []
        chord_vec = []
        twist_vec = []
        rake_vec = []
        skew_vec = []
        for line in datafile:
            if any(char.isupper() for char in line):
                continue    
            # data = line.split()
            data = re.sub('[,\t]+', ' ', line).split() # Zhicheng: use regex to replace ',' and '\t' with space
            floats = []
            for x in data:
                floats.append(float(x))
            if len(floats) > 0: # if the line is empty, do nothing
                radii_vec.append(floats[0])
                chord_vec.append(floats[1])
                twist_vec.append(floats[2])
                rake_vec.append(floats[3])
                skew_vec.append(floats[4])

    datafile.close()

    # if geometry_file == "SP2_geom.dat":
    #     # add a last section to model the tip termination
    #     radii_vec.append(radii_vec[-1]+chord_vec[-1]/4)
    #     chord_vec.append(chord_vec[-1]*0.7)
    #     twist_vec.append(twist_vec[-1])
    #     rake_vec.append(rake_vec[-1])
    #     skew_vec.append(skew_vec[-1]+0.1*skew_vec[-1])


    return np.array(radii_vec), np.array(chord_vec), np.array(twist_vec), np.array(rake_vec), np.array(skew_vec)
