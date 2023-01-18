# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------


# aims at reproducing the rod-airfoil benchmark, Casalino, Jacob and Roger aiaaj03 DOI: 10.2514/2.1959

# # Drawing of the NACA profile inspired from
# the Wikipedia page: https://en.wikipedia.org/wiki/NACA_airfoil
# JoshTheEngineer matlab's code: https://github.com/jte0419/NACA_4_Digit_Airfoil

# On the general use of Gmsh (correspondance of the .geo synthax with the python API provided)
# (C++ use of Gmsh is also supported, more consistent with MFEM?)
# https://gmsh.info/doc/texinfo/gmsh.html
# meshing with S.A.E. Miller, "Tutorial on Computational Grid Generation for CFD using GMSH"
# https://youtube.com/playlist?list=PLbiOzt50Bx-l2QyX5ZBv9pgDtIei-CYs_


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
    if optimisedGridSpacing:
        x = np.linspace(0,1,gridPts+1)  # Uniform spacing
        x = 0.5*(1-np.cos(x*np.pi))     # Non-uniform spacing
        x = np.concatenate((x[0:-2],[x[-1]])) # remove penultimate point that is systematically to close to TE
    else:
        x = np.linspace(0,1,gridPts)    # Uniform spacing
    #
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

def rotationMatrix(rotAnglesVec):
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

def gmeshed_airfoil(structTag, GeomSpec, GridPtsSpec, rotMat):

    pointTag = structTag[0]
    lineTag = structTag[1]
    surfaceTag = structTag[2]

    NACA_type = GeomSpec[0]
    bluntTrailingEdge = GeomSpec[1]
    optimisedGridSpacing = GeomSpec[2]
    AoA = GeomSpec[3]
    chord = GeomSpec[4]
    airfoilReferenceAlongChord = GeomSpec[5]
    airfoilReferenceCoordinate = GeomSpec[6]
    height_LE = GeomSpec[7]
    height_TE = GeomSpec[8]
    TEpatchLength = GeomSpec[9]
    TEpatchGridFlaringAngle = GeomSpec[10]
    wakeLength = GeomSpec[11]
    wakeGridFlaringAngle = GeomSpec[12]

    gridPts_alongNACA = GridPtsSpec[0]
    gridPts_inBL = GridPtsSpec[1]
    gridPts_inTE = GridPtsSpec[2]
    gridPts_alongTEpatch = GridPtsSpec[3]
    gridPts_alongWake = GridPtsSpec[4]
    gridGeomProg_inBL = GridPtsSpec[5]
    gridGeomProg_alongTEpatch = GridPtsSpec[6]
    gridGeomProg_alongWake = GridPtsSpec[7]

    airfoilReferenceCoordinate = np.array(airfoilReferenceCoordinate)
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
        rotVec = np.matmul(rotMat, np.array([upper_NACAfoil[0,i], upper_NACAfoil[1,i], airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_LE = pointTag
    if bluntTrailingEdge:
        for i in range(gridPts_alongNACA-1):
            rotVec = np.matmul(rotMat, np.array([lower_NACAfoil[0,i], lower_NACAfoil[1,i], airfoilReferenceCoordinate[2]]))
            gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
        pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
        point_TEl = pointTag
        rotVec = np.matmul(rotMat, np.array([0.5*(upper_NACAfoil[0,0]+lower_NACAfoil[0,-1]), 0.5*(upper_NACAfoil[1,0]+lower_NACAfoil[1,-1]), airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_TE = pointTag
    else:
        for i in range(gridPts_alongNACA-2):
            rotVec = np.matmul(rotMat, np.array([lower_NACAfoil[0,i], lower_NACAfoil[1,i], airfoilReferenceCoordinate[2]]))
            gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100,pointTag+i+1)
        pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
        point_TEl = point_TEu
        point_TE = point_TEu

    # creation of the offset layer
    point_up = pointTag+1
    for i in range(gridPts_alongNACA):
        rotVec = np.matmul(rotMat, np.array([upper_offset[0,i], upper_offset[1,i], airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_left = pointTag
    for i in range(gridPts_alongNACA-1):
        rotVec = np.matmul(rotMat, np.array([lower_offset[0,i], lower_offset[1,i], airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_low = pointTag

    # creation of the TEpatch
    point_lowRight = pointTag+1
    for i in range(np.size(x_TEpatch)):
        rotVec = np.matmul(rotMat, np.array([x_TEpatch[i], y_TEpatch[i], airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_upRight = pointTag

    point_lowFarRight = pointTag+1
    for i in range(np.size(x_wake)):
        rotVec = np.matmul(rotMat, np.array([x_wake[i], y_wake[i], airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+i+1)
    pointTag = pointTag+i+1 # store the last 'pointTag' from previous loop
    point_upFarRight = pointTag

    rotVec = np.matmul(rotMat, np.array([0.5*(x_TEpatch[0]+x_TEpatch[-1]), 0.5*(y_TEpatch[0]+y_TEpatch[-1]), airfoilReferenceCoordinate[2]]))
    gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
    pointTag = pointTag+1 # store the last 'pointTag' from previous loop
    point_TEwake = pointTag

    rotVec = np.matmul(rotMat, np.array([0.5*(x_wake[0]+x_wake[-1]), 0.5*(y_wake[0]+y_wake[-1]), airfoilReferenceCoordinate[2]]))
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
                                airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_lowMidRight = pointTag

        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_TEpatch[0]+0.5*(1+alphaStretch)*x_TEpatch[-1], 
                                0.5*(1-alphaStretch)*y_TEpatch[0]+0.5*(1+alphaStretch)*y_TEpatch[-1],
                                airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_upMidRight = pointTag

        # ~~~
        
        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_wake[-1]+0.5*(1+alphaStretch)*x_wake[0], 
                                0.5*(1-alphaStretch)*y_wake[-1]+0.5*(1+alphaStretch)*y_wake[0],
                                airfoilReferenceCoordinate[2]]))
        gmsh.model.geo.addPoint(rotVec[0], rotVec[1], rotVec[2], chord/100, pointTag+1)
        pointTag = pointTag+1 # store the last 'pointTag' from previous loop
        point_lowMidFarRight = pointTag

        rotVec = np.matmul(rotMat, np.array([0.5*(1-alphaStretch)*x_wake[0]+0.5*(1+alphaStretch)*x_wake[-1], 
                                0.5*(1-alphaStretch)*y_wake[0]+0.5*(1+alphaStretch)*y_wake[-1],
                                airfoilReferenceCoordinate[2]]))
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

    sStructGridSide = []
    sStructGridSide.extend([*sTS[sBLstructGrid], sTS[sTEpatchUp], sTS[sTEpatchLow], sTS[swakeUp], sTS[swakeLow] ])
    if bluntTrailingEdge:
        sStructGridSide.extend([sTS[sTEpatchMidUp], sTS[sTEpatchMidLow], sTS[swakeMidUp], sTS[swakeMidLow]])

    return sStructGridSide

def returnStructGridOuterShell(sTS, tsTS, radii_step, bluntTrailingEdge):
    
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

def returnStructGridVol(vTS, bluntTrailingEdge):

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
    returnStructGridVol.extend([vTS[vBL], vTS[vwakeUp], vTS[vwakeLow], vTS[vTEup], vTS[vTElow]])

    if bluntTrailingEdge:
        returnStructGridVol.extend([vTS[vTEmidUp], vTS[vTEmidLow]])
        returnStructGridVol.extend([vTS[vwakeMidUp], vTS[vwakeMidLow]])

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
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_cylTop = surfaceTag

    cylBotEdges = [line_cylBotArc1, line_cylBotArc2, line_cylBotArc3, line_cylBotArc4]
    gmsh.model.geo.add_curve_loop( [*cylBotEdges], surfaceTag+1) 
    gmsh.model.geo.addPlaneSurface([surfaceTag+1], surfaceTag+1) # mesh inside the airfoil
    gmsh.model.geo.mesh.setTransfiniteSurface(surfaceTag+1)
    gmsh.model.geo.mesh.setRecombine(pb_2Dim, surfaceTag+1) # To create quadrangles instead of triangles
    surfaceTag = surfaceTag+1
    surf_cylBot = surfaceTag

    return [surf_cylSide1, surf_cylSide2, surf_cylSide3, surf_cylSide4, surf_cylTop, surf_cylBot], pointTag, lineTag, surfaceTag

# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************

def gmeshed_rectangle_contour(x_min, x_max, y_min, y_max, elemSize_rect, pointTag, lineTag):

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # creation of the Points # #
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$

    gmsh.model.geo.addPoint(x_min, y_min, 0.0, elemSize_rect, pointTag+1)
    pointTag = pointTag+1 
    point_SW = pointTag
    gmsh.model.geo.addPoint(x_max, y_min, 0.0, elemSize_rect, pointTag+1)
    pointTag = pointTag+1
    point_SE = pointTag
    gmsh.model.geo.addPoint(x_max, y_max, 0.0, elemSize_rect, pointTag+1)
    pointTag = pointTag+1
    point_NE = pointTag
    gmsh.model.geo.addPoint(x_min, y_max, 0.0, elemSize_rect, pointTag+1)
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
            data = line.split()
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

    return np.array(radii_vec), np.array(chord_vec), np.array(twist_vec), np.array(rake_vec), np.array(skew_vec)
