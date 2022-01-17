"""
This file is used to define global vehicle parameters which are not meant to
change during a simulation.
"""

import numpy as np

# global redline, fdr, gears, r

# Vehicle parameters used in multiple files
redline = 12000 # [rpm]
fdr = 36/11 # final drive ratio
gears = 2.073*np.array([31/12, 32/16, 30/18, 26/18, 27/21, 23/20]); # crank:sprocket
r = 9.875*0.0254; # wheel radius [meters]

# Vehicle parameters used only in accelSim
m = (450+175)/2.20462; # vehicle mass with driver [lbs]
wdist = 0.42; # CG location from rear [%]
wb = 5.325*12*2.54/100; # wheelbase [meters]
cgh = .655833333*12*2.54/100; # CG height [meters]
uk = 1.4; # tire coefficient of friction
cp = 0.45; # aerodynamic downforce center of pressure from front [%]
g = 9.81; # gravitational acceleration [m/s^2]
dteff = 0.85; # driveline efficiency
# Fuel map
input_FEPW = np.array([0, 2.4192, 2.5564, 3.10415, 3.7058, 3.7513, 3.9263,
                       3.9697, 4.0796, 4.0922, 4.0957, 4.01625, 4.08975,
                       4.2679, 4.4856, 4.7614, 5.319475, 5.327816, 5.19843,
                       5.07126, 4.98575, 4.956, 5.01025, 4.74985, 4.45573,
                       4.421025])*1.015*1.014; # [ms]
# Torque map
input_Torque_lbft = np.array([0.3, 15.17, 15.93, 22.67, 26.13, 27.57, 29.07,
                              28.83, 30.07, 31.17, 30.97, 31.6, 31.53, 31.83,
                              33, 35.33, 38.1, 37.67, 36.67, 35, 33.67, 31.83, 
                              31, 29.13, 27.66, 25.5])-0.3; # [lb-ft]
RPM = np.hstack([0,np.arange(2000,14500,500)]); # rpm query points for maps
Lambda = np.array([1.,     1.046,  0.95,   0.9667, 0.982,  0.9314, 0.9608,
                   0.9296, 0.9347, 0.9348, 0.9357, 0.9305, 0.9203, 0.9215,
                   0.9274, 0.919,  0.9171, 0.9181, 0.9003, 0.8841, 0.9235,
                   0.9071, 0.8763, 0.8519, 0.8711, 0.8509]); # Reference lambda