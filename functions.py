"""
Function library.
Currently, only standalone functions, i.e., ones not using global variables,
are included in this file.
"""

import numpy as np

def find_downforce(curr_WSSf_mph):
    """ Find drag at current vehicle speed in mph
    Fit obtained from 2019 FCA wind tunnel data below
        x = [35, 45, 55]'; % speed [mph]
        f = [324.6733759, 519.6826238, 774.942625]'; % downforce [N]

    @param curr_WSSf_mph Vehicle speed [mph]

    @return Downforce [N]
    """

    downforce_N = 0.257206872977041*curr_WSSf_mph**2; # [N]
    return downforce_N

def find_drag(curr_WSSf_mph):
    """ Find drag at current vehicle speed in mph
    Fit obtained from 2019 FCA wind tunnel data below
        x = [35, 45, 55]; % speed [mph]
        f = [197.960913, 317.1329839, 469.0276004]; % drag [N]

    @param curr_WSSf_mph Vehicle speed [mph]

    @return Drag [N]
    """

    drag_N = 0.15614997429366*curr_WSSf_mph**2; # [N]
    return drag_N

def points_delta(time_sec=0, fuel_l=0):
    """ Calculate the point gain (or loss) from a perturbation
    in the lap time and/or fuel consumption. Based on FSAEM 2019 results.
    Points are calculated as a sum of endurance and efficiency events.

    @param time_sec Lap time perturbation [sec]
    @param fuel_l Lap fuel consumption perturbation [liters]

    @return Linearized competition points delta
    """

    numlaps = 11
    original = 92840/139.45 - 0.15418*139.45*4.112 - 382.95
    new = 92840/(139.45+time_sec/numlaps) \
          - 0.15418*(139.45+time_sec/numlaps)*(4.112+fuel_l) - 382.95
    delta = new - original
    return delta

def points_delta_lin(time_sec=0, fuel_l=0):
    """ Calculate the point gain (or loss) from a perturbation
    in the lap time and/or fuel consumption. Based on FSAEM 2019 results.
    Points are calculated as a sum of endurance and efficiency events.
    This function is the linearization of the original nonlinear cost function.
    One second per lap is equivalent to about 22.86 cc of fuel per lap.

    @param time_sec Lap time perturbation [sec]
    @param fuel_l Lap fuel consumption perturbation [liters]

    @return Linearized competition points delta
    """

    h = 1e-8
    grad_time = (points_delta(h, 0) - points_delta(-h, 0)) / (2*h)
    grad_fuel = (points_delta(0, h) - points_delta(0, -h)) / (2*h)
    delta = time_sec*grad_time + fuel_l*grad_fuel
    return delta

def find_gear4(currVehV, fdr, gears, redline, r):
    """ Find gear given only vehicle velocity
    Same as findGear2 in accelSim but meant to be used as a standalone function

    @param currVehV_mph Current vehicle velocity in miles per hour
    @param fdr Final drive ratio
    @param gears Gear ratios
    @param redline Engine redline
    @param r Wheel radius [m]

    @return Current gear
    """

    gear = 0;
    j = 6;
    while ((j>=1) and (currVehV/60*1609.4/(2*np.pi*r)*fdr*gears[j-1] < redline)):
        gear = j;
        j = j-1;
    return gear