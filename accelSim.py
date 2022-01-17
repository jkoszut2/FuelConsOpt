import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import time
from functions import findDownforce, findDrag
from vehicleParams import *

def accelSim(input_FEPW, input_Torque_lbft, input_Dist_meters=75,
             frontVel_meterspersec=10, rearVel_meterspersec=10, gear=2,
             modifierArray=np.array([0]), producePlots=0):

    global dt, shifttime, shifttimer, shiftflag, FEPW, Torque_Nm, RPM, LC_wss

    dt = 0.001; # forward Euler discretization length [secs]

    FEPW = input_FEPW/((Lambda+modifierArray[0])/Lambda);
    Torque_Nm = input_Torque_lbft/2.20462*g*12*2.54/100; # [Nm]
    Torque_Nm = Torque_Nm*(-2.65*modifierArray[0]**2 - 0.0437*modifierArray[0] + 1);
    # Make torque drop off faster at higher rpm
    # Torque_Nm[0:20] = Torque_Nm[0:20]*(-2.65*modifierArray[0]**2 - 0.0437*modifierArray[0] + 1);
    # Torque_Nm[20:] = Torque_Nm[20:]*(-2.65*modifierArray[0]**3 - 5*modifierArray[0]**2 + 1);
    # Enable coasting
    # if modifierArray[0] == 0.4:
    #     FEPW = FEPW*0
    #     Torque_Nm = Torque_Nm*0
    # LC_wss = np.array([0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 25]);
    shifttime = 0; # [sec]
    shifttimer = 0; # [sec]
    shiftflag = 0; # boolean to signify upshift
    if redline > max(RPM):
        warnings.warn("Error! Engine data is not sufficiently populated!\n")
        warnings.warn("Simulation may not run correctly if time step is too large.\n")

    # NOTE: No warning if rpm input to findTorque > redline
    # Result for above is 0 and NOT nan, thus need another case to detect

    distMax = input_Dist_meters;

    # This is used to preallocate a time array assuming the acceleration event
    # will take no longer than tMax seconds. Once the acceleration run has been
    # simulated, only the nonzero data entries are given as the function output.
    tMax = 10; # seconds

    # Initialize all states and set their indices
    # NOTE: If adding a new state, it must also be added to the header variable
    header = ['time [sec]', 'pos [m]', 'vel [m/s]', 'acc [m/s^2]', 'gear', 'rpm', 'wssfl [m/s]', 'wssrl [m/s]', \
              'fuelcons [cc]', 'cumfuelcons [cc]', 'Torque [Nm]', 'wheel_Torque [Nm]', 'slip' , \
              'Weight_Transfer_Acc [N]', 'Fx [N]', 'MaxFx [N]', 'Drag [N]', 'Downforce [N]', 'FEPW [ms]', 'IDC [%]'];
    # Time
    index_Time = 1-1;
    init_Time = 0; # seconds
    # Position
    index_Pos = 2-1;
    init_Pos = 0; # meters
    # Velocity
    index_Vel = 3-1;
    init_Vel = frontVel_meterspersec; # meters/sec
    # Gear
    index_Gear = 5-1;
    init_Gear = gear; # gear 
    # Wheel Speed (Front)
    index_WSSFL = 7-1;
    init_WSSFL = frontVel_meterspersec*3600/1609.4; # miles/hr
    # Wheel Speed (Rear)
    index_WSSRL = 8-1;
    init_WSSRL = rearVel_meterspersec*3600/1609.4; # miles/hr
    # Drag
    index_Drag = 17-1;
    init_Drag = findDrag(init_WSSFL);
    # Downforce
    index_Downforce = 18-1;
    init_Downforce = findDownforce(init_WSSFL);
    # RPM
    index_RPM = 6-1;
    init_RPM = rearVel_meterspersec/(np.pi*2*r)*60*fdr*gears[init_Gear-1]; # rpm
    # Acceleration
    index_Acc = 4-1;
    init_Acc = ((findTorque(init_RPM)*fdr*gears[init_Gear-1])/r-init_Drag)\
                /m/g*dteff; # [g]
    # Instantaneous Fuel Consumption
    index_FuelCons = 9-1;
    init_FuelCons = 0; # cc
    # Cumulative Fuel Consumption
    index_CumFuelCons = 10-1;
    init_CumFuelCons = 0; # cc
    # Engine Torque
    index_Torque_Nm = 11-1;
    init_Torque_Nm = findTorque(init_RPM); # Nm
    # Wheel Torque
    index_Wheel_Torque_Nm = 12-1;
    init_Wheel_Torque_Nm = init_Torque_Nm; # Nm
    # Slip
    index_Slip = 13-1;
    init_Slip = 0; # boolean
    # Weight transfer due to longitudinal accel
    index_WeightTransferAcc = 14-1;
    init_WeightTransferAcc = m*init_Acc*(cgh/wb)*g; # N
    # Fx of rear tires
    index_Fx = 15-1;
    init_Fx = init_Torque_Nm*fdr*gears[init_Gear-1]*dteff/r; # N
    # Max Fx
    index_MaxFx = 16-1;
    init_MaxFx = ((1-wdist)*m*9.8 + init_Downforce*cp \
                 + init_WeightTransferAcc)*uk; # N
    # FEPW
    index_FEPW = 19-1;
    init_FEPW = np.interp(init_RPM, RPM, FEPW);
    # Injector Duty Cycle
    index_IDC = 20-1;
    init_IDC = init_FEPW/(1/(init_RPM/60/2)*1000)*100;

    currlocals = list(locals().keys())
    states = [currlocals[i] for i in range(len(currlocals)) \
            if currlocals[i].startswith('index_') ]
    stateinits = [currlocals[i] for i in range(len(currlocals)) \
            if currlocals[i].startswith('init_') ]
    tmp3 = [];
    for i in range(len(states)):
        tmp2 = eval(states[i]);
        tmp3.append(tmp2);
    x = np.zeros((int(tMax/dt)+1, max(tmp3)+1));
    x[:] = np.nan
    del tmp2, tmp3

    A = np.array([[0, 1],
                  [0, 0]]);
    B = np.array([[0, 0],
                  [1/m, -1/m]]);

    for i in range(len(states)):
        x[0, eval(states[i])] = eval(stateinits[i]);


    t = np.arange(0, tMax+dt, dt);
    i = 1; # Start calculations the next sample after t=0 sec
    while ((x[i-1,index_Pos] < distMax) and i <= (tMax/dt)):
    #     if producePlots
    #         fprintf('Time: #0.5f\n', t(i))
    #     end
        # Using previous solution, find wheel speed, gear, and max Fx
        prev_WSSFL = x[i-1,index_WSSFL]; # front wheel speed [mph]
        prev_WSSRL = x[i-1,index_WSSRL]; # rear wheel speed [mph]
        prev_Gear = findGear(prev_WSSRL, x[i-1, index_Gear]);
        prev_Drag = findDrag(prev_WSSFL); # N
        prev_Downforce = findDownforce(prev_WSSFL); # N
        prev_WeightTransferAcc = m*x[i-1, index_Acc]*(cgh/wb)*g; # N
        prev_WeightRear = (1-wdist)*m*9.8 + prev_Downforce*cp \
                          + prev_WeightTransferAcc; # N
        maxFx = prev_WeightRear*uk; # N

        # Find engine speed and torque output
        prev_rpm = findRPM(prev_WSSRL, prev_Gear);
        # Find engine torque output
        prev_engine_Torque_Nm = findTorque(prev_rpm);
        prev_wheel_Torque_Nm = x[i-1, index_Wheel_Torque_Nm];

        # Find longitudinal force at rear wheels assuming no slip
        prev_Fx = prev_engine_Torque_Nm*fdr*gears[prev_Gear-1]*dteff/r;

        # Solve state space model
        u = np.array([[prev_Fx],
                      [prev_Drag]]);
        dxdt = A@np.array([[x[i-1, index_Pos]], [x[i-1, index_Vel]]]) + B@u;
        x[i, index_Time] = t[i];
        x[i, index_Pos] = x[i-1, index_Pos] + dxdt[1-1, 0]*dt;
        x[i, index_Vel] = x[i-1, index_Vel] + dxdt[2-1, 0]*dt;
        x[i, index_Acc] = dxdt[2-1, 0]/g;
        x[i, index_Gear] = prev_Gear;
        x[i, index_WSSFL] = x[i, index_Vel]*3600/1609.4;
        x[i, index_WSSRL] = x[i-1, index_WSSRL] + dxdt[2-1, 0]*dt*3600/1609.4;
        curr_rpm = findRPM(x[i, index_WSSRL], prev_Gear);
        x[i, index_RPM] = curr_rpm;
        currFuelCons = findFuelCons(curr_rpm);
        x[i, index_FuelCons] = currFuelCons;
        x[i, index_CumFuelCons] = x[i-1, index_CumFuelCons] + currFuelCons;
        x[i, index_Torque_Nm] = prev_engine_Torque_Nm;
        x[i, index_Wheel_Torque_Nm] = prev_wheel_Torque_Nm;
        x[i, index_WeightTransferAcc] = prev_WeightTransferAcc;
        x[i, index_Fx] = prev_Fx;
        x[i, index_MaxFx] = maxFx;
        x[i, index_Drag] = prev_Drag;
        x[i, index_Downforce] = prev_Downforce;
        x[i, index_FEPW] = np.interp(curr_rpm, RPM, FEPW);
        x[i, index_IDC] = x[i, index_FEPW]/(1/(curr_rpm/60/2)*1000)*100;

        if prev_Fx > maxFx:
            slip = 1;
        else:
            slip = 0;

        x[i,index_Slip] = slip;
        i = i+1;

    lastindex = i-1; # Avoid plotting zeros located at end of state array
    if (x[lastindex, index_Pos] >= input_Dist_meters):
        # Create state array
        stateData = pd.DataFrame(data=x[:lastindex+1,:], index=x[:lastindex+1,0],
                                 columns=header)

        # Calculate accel time and fuel consumption
        # accel_time = t[np.searchsorted(x[:,index_Pos], distMax)]; # seconds
        accel_time = np.interp(input_Dist_meters, stateData['pos [m]'],
                               stateData['time [sec]']); # seconds
        fuel_consumed = np.interp(input_Dist_meters, stateData['pos [m]'],
                                  stateData['cumfuelcons [cc]']); # seconds

        # Interpolate last row in state array so final distance matches
        # input distance
        stateData2 = stateData.copy()
        stateData2 = stateData2.append(stateData2.iloc[[-1]])
        stateData2.iloc[-2, :] = np.nan
        # Don't interpolate gear to prevent it from becoming a non-integer in
        # case a gear shift happens at the last time step
        stateData2['gear'].values[-2] = stateData2['gear'].values[-1]
        # Set last index to be the accel time
        stateData2.index.values[-2] = accel_time
        stateData2 = stateData2.interpolate(method='index')
        # Drop the last row
        stateData2 = stateData2.head(-1)
    else:
        # Should not get in here but double check anyways to simplify debugging in
        # the event that tMax is not large enough
        raise ValueError(f'Simulation ended before target distance was covered. ' \
                f'Increase tMax to provide simulation with enough time to ' \
                f'cover distance set by function input.' \
                f'\nCurrent tMax = {tMax:0.0f} sec' \
                f'\nDistance input = {distMax:0.3f} meters' \
                f'\nDistance traveled in simulation = ' \
                f'{x[lastindex, index_Pos]:0.3f} meters' \
                f'\nInitial RPM = {x[0, index_RPM]:0.3f}' \
                f'\nInitial WSSFL = {x[0, index_WSSFL]:0.3f}')

    if producePlots:
        print(f'Accel time: {accel_time*1000:0.1f} msec\n', )
        print(f'Fuel consumed: {fuel_consumed*1000:0.4f} mcc\n')

        plt.figure(figsize=(10, 6))
        plt.subplot(2,2,1)
        plt.grid()
        plt.plot(t[:lastindex+1], x[:lastindex+1, index_Pos])
        plt.plot(t[:lastindex+1], x[:lastindex+1, index_WSSFL])
        plt.hlines(y=input_Dist_meters, xmin=0, xmax=t[lastindex], color='k', linestyle='--');
        plt.legend(['Position [m]', 'WSS_front [mph]', '75m'], loc='lower right')
        plt.xlim([0, t[lastindex]])
        plt.xlabel('Time [sec]')
        plt.ylabel('Position [m]')

        plt.subplot(2,2,2)
        plt.grid()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(t[:lastindex+1], x[:lastindex+1, index_Acc], label='Acceleration [g]')
        ax2.plot(t[:lastindex+1], x[:lastindex+1, index_Torque_Nm], 'r-', label='Engine Torque [Nm]')
        # ax2.plot(t[:lastindex+1], x[:lastindex+1, index_Wheel_Torque_Nm], 'k-')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
        plt.xlim([0, t[lastindex]])
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('Acceleration [g]')
        ax2.set_ylabel('Engine Torque [Nm]', rotation=270, va='bottom')

        plt.subplot(2,2,3)
        plt.grid()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(t[:lastindex+1], x[:lastindex+1, index_Fx], linewidth=3, label='Fx [N]')
        ax1.plot(t[:lastindex+1], x[:lastindex+1, index_MaxFx], '--', linewidth=2, label='Max Fx [N]')
        ax2.plot(t[:lastindex+1], x[:lastindex+1, index_Gear], 'C2', label='Gear')
        plt.fill_between(t[:lastindex+1], x[:lastindex+1, index_Slip], facecolor='C3', edgecolor='C3', label='Slip')
        plt.ylim([0, max(x[:lastindex,index_Gear])+0.5])
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
        plt.xlim([0, t[lastindex]])
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('Force [N]')
        ax2.set_ylabel('Gear/Slip', rotation=270, va='bottom')

        plt.subplot(2,2,4)
        plt.grid()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(t[:lastindex+1], x[:lastindex+1, index_RPM], label='RPM')
        # ax2.plot(t[:lastindex+1], x[:lastindex+1, index_FuelCons])
        ax2.plot(t[:lastindex+1], x[:lastindex+1, index_CumFuelCons], 'C1', label='Fuel Consumption [cc]')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        plt.xlim([0, t[lastindex]])
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('RPM')
        ax2.set_ylabel('Fuel Consumption [cc]', rotation=270, va='bottom')

        #plt.tight_layout()
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.4, hspace = 0.3)
        plt.show()

    return accel_time, stateData2

def findFuelCons(curr_rpm):
    """ Find fuel injection quantity at current engine speed and time
    discretization
    Global variables: dt FEPW RPM redline
    """
    # Injector flow rate below from Ford injector characterization data
    # Ford injector data provided in mass flow rate, so needs to be converted
    # to volumetric flow rate. Also needs to be adjusted based off Bernoulli's
    # principle as Ford testing was performed at 270 kPa
    # Average of gauge fuel pressure from FSAEM and FSAEL is 44.03psi
    # 44.03psi --> 303.57616 kPa
    inj_flowrate = 2.58*60/0.738*np.sqrt(303.57616/270); # cc/min
    # Calculating volume for now so don't need fuel density
    # fuel_density = 0.73; # kg/L

    if (curr_rpm < redline):
        curr_fepw = np.interp(curr_rpm, RPM, FEPW);
    else:
        # Should only happen right before a shift
        curr_fepw = np.interp(redline, RPM, FEPW);

    inj_max_ms = 1/(curr_rpm/60/2)*1000; # max pulsewidth per cycle [ms]
    inj_duty = curr_fepw/inj_max_ms*100; # duty cycle [%]
    fuelcons = inj_duty/100*dt*inj_flowrate/60*4; # cc
    return fuelcons   

def findGear(currVehV, prevGear):
    """ Find gear given only vehicle velocity
    currVehV is current vehicle velocity in miles per hour
    prevGear is previous gear (min=1, max = 6) and is used to determine if
    a shift should occur
    Global variables: fdr gears redline shiftflag r
    """
    global shiftflag
    prevGear = round(prevGear);
    gear = 0;
    j = 6;
    while ((j>=1) and (currVehV/60*1609.4/(2*np.pi*r)*fdr*gears[j-1] < redline)):
        gear = j;
        j = j-1;

    # Avoid downshifting
    if (prevGear > gear):
        gear = prevGear;

    # Check if upshift present
    if (gear != prevGear):
        shiftflag = 1;
    else:
        shiftflag = 0;
    return gear

def findGear2(currVehV):
    """ Find gear given only vehicle velocity
    Same as findGear but with the change of not using a shift flag
    Use this function to initialize the accel sim
    currVehV is current vehicle velocity in miles per hour
    Global variables: fdr gears redline r
    """
    gear = 0;
    j = 6;
    while ((j>=1) and (currVehV/60*1609.4/(2*np.pi*r)*fdr*gears[j-1] < redline)):
        gear = j;
        j = j-1;

def findLCrpm(curr_wheelspeed_mph):
    # Global variables: LC_rpm LC_wss
    rpm_launch = np.interp(curr_wheelspeed_mph, LC_wss, LC_rpm);
    return rpm_launch

def findRPM(currVehV, currGear):
    """ Find RPM given vehicle velocity and gear 
    currVehV is current vehicle velocity in miles per hour
    currGear is current gear (min=1, max = 6)
    Global variables: r, fdr, gears
    """
    currGear = round(currGear);
    rpm = currVehV/60*1609.4/(2*np.pi*r)*fdr*gears[currGear-1];
    return rpm

def findTorque(curr_rpm):
    """ Find engine torque at current engine speed
    Global variables: dt, RPM, Torque_Nm, redline,
                                        shiftflag, shifttime, shifttimer
    """
    global shifttimer, shiftflag
    if (curr_rpm <= redline):
        torque = np.interp(curr_rpm, RPM, Torque_Nm);
    else:
        # Should only happen right before a shift
        torque = np.interp(redline, RPM, Torque_Nm);

    # Check to make sure not shifting gears
    if (shifttimer - dt > 0):
        shifttimer = shifttimer - dt;
        torque = 0;
    elif (shifttimer - dt < 0):
        shifttimer = 0;

    if (shiftflag and shifttime>0):
        torque = 0;
        shiftflag = 0;
        shifttimer = shifttimer + shifttime;
    return torque

def findVehV(currRPM, currGear):
    """ Finds vehicle velocity in miles per hour
    Global variables: r, fdr, gears
    """
    currGear = round(currGear);
    VehV = currRPM*60/1609.4*(2*np.pi*r)/gears[currGear-1]/fdr; # [mph]
    return VehV_mph

# start = time.time()
# [accelTime, stateData] = accelSim(input_FEPW, input_Torque_lbft)
# end = time.time()
# print(f"Runtime: {end - start:0.5f} sec")
# accelSim(input_FEPW, input_Torque_lbft, producePlots=1)