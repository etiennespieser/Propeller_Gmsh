# Copyright (c) 2022 Étienne Spieser (Tiānài), member of AANTC (https://aantc.ust.hk/)
# available under MIT licence at: https://github.com/etiennespieser  
# ------------------------------------------------------------------------------------

# source: Pope, Turbulent Flows, Cambridge university press, 2000
#         https://www.cfd-online.com/Wiki/Y_plus_wall_distance_estimation
#         https://www.cfd-online.com/Wiki/Skin_friction_coefficient

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../WaveMap59/')
import grimoireOfPlots as gop

Re = 5*10**4
u_ext = 3.75 # m/s
kin_vis = 0.000015 # m**2/s
target_yPlus = 1.0

plotCfModels = False

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Skin friction coefficient model # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Re_range = 10**np.linspace(4, 10, 10000)

Cf_powerLaw_1_7 = 0.0576*Re_range**(-1/5)                  # for  5*10**5 < Re < 10**7
Cf_powerLaw_1_7_recalibrated = 0.0592*Re_range**(-1/5)     # for  5*10**5 < Re < 10**7
Cf_Schlichting = (2.0*np.log10(Re_range)-0.65)**(-2.3)     # for            Re < 10**9
Cf_SchultzGrunov = 0.37*np.log10(Re_range)**(-2.584)
Cf_Prandtl = 0.074*Re_range**(-1/5)                                                 # year 1927
Cf_Telfer = 0.34*Re_range**(-1/3) + 0.0012                                          # year 1927
Cf_PrandtlSchlichting = 0.455*np.log10(Re_range)**(-2.58)                           # year 1932
# Cf_Schoenherr = 0.0586*np.log10(Re_range*Cf_Schoenherr)**(-2)                     # year 1932
# Cf_vonKarmanTheodore = (4.15*np.log10(Re_range*Cf_vonKarmanTheodore)+1.7)**(-2)   # year 1934
Cf_SchultzGrunov2 = 0.427*(np.log10(Re_range)-0.407)**(-2.64)                       # year 1940
Cf_KempfKarman = 0.055*Re_range**(-0.182)                                           # year 1951
# Cf_LapTroost = 0.0648*(np.log10(Re_range*Cf_LapTroost**0.5)-0.9526)**(-2)         # year 1952
Cf_Landweber =  0.0816*(np.log10(Re_range)-1.703)**(-2)                             # year 1953
Cf_Hughes = 0.067*(np.log10(Re_range)-2)**(-2)                                      # year 1954
Cf_Wieghard = 0.52*np.log10(Re_range)**(-2.685)                                     # year 1955
Cf_ITTC = 0.075*(np.log10(Re_range)-2)**(-2)                                        # year 1957
Cf_Gadd = 0.0113*(np.log10(Re_range)-3.7)**(-1.15)                                  # year 1967
Cf_Granville = 0.0776*(np.log10(Re_range)-1.88)**(-2) + 60/Re_range                 # year 1977
# Cf_DateTurnock = (4.06*np.log10(Re_range*Cf_DateTurnock)-0.729)**(-2)             # year 1999

if plotCfModels:
    windowsWidth = 8 # cm
    axisEqual = False
    HV_ratio = 0.75
    fig, ax = gop.createPaintingFrame(windowsWidth, Re_range[0], Re_range[-1], 10**(-3), 0.1, axisEqual, HV_ratio)
    plt.loglog(Re_range, Cf_powerLaw_1_7, "")
    plt.loglog(Re_range, Cf_powerLaw_1_7_recalibrated, "")
    plt.loglog(Re_range, Cf_Schlichting, "")
    plt.loglog(Re_range, Cf_SchultzGrunov, "")
    plt.loglog(Re_range, Cf_Prandtl, "")
    plt.loglog(Re_range, Cf_Telfer, "")
    plt.loglog(Re_range, Cf_PrandtlSchlichting, "")
    plt.loglog(Re_range, Cf_SchultzGrunov2, "")
    plt.loglog(Re_range, Cf_KempfKarman, "")
    plt.loglog(Re_range, Cf_Landweber, "")
    plt.loglog(Re_range, Cf_Hughes, "")
    plt.loglog(Re_range, Cf_Wieghard, "+")
    plt.loglog(Re_range, Cf_ITTC, "")
    plt.loglog(Re_range, Cf_Gadd, "")
    plt.loglog(Re_range, Cf_Granville, "")
    plt.xlabel(r'$\rm{Re}$')
    plt.ylabel(r'$C_f$')
    plt.show()

# selection of the model for the skin friction coefficient:
Cf_range = Cf_Wieghard

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # compute of the dimensional thickness y+ # #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Cf = np.interp(Re, Re_range, Cf_range)
frictionVelo = np.sqrt(Cf*u_ext**2/2)
y = target_yPlus*kin_vis/frictionVelo

print("for a target y+ of "+str(target_yPlus)+", the first cell height must be "'{:.2e}'.format(y)+" m.")



