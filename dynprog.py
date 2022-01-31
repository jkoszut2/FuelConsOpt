import numpy as np
import matplotlib.pyplot as plt
import time
from vehicleparams import *
from accelsim import *
from functions import points_delta, find_gear4

# NOTE: Set vehicle parameters, e.g., gear ratios and redline, in the
# vehicleParams.py file

def dyn_prog(init_vel=15, dist_total=35, dist_interval=1, vel_size=5,
             plots=0, debug_mode=0):
    """
    Given an initial velocity and travel distance, find the optimal
    lambda (input) to maximize points from the efficiency and endurance events
    using the method of dynamic programming.
    
    @param init_vel Initial velocity [m/s]
    @param dist_total Travel distance [m]
    @param dist_interval Spatial discretization length [m]
    @param vel_size State discretization grid width for each 
    @param plots Enable/disable plot outputs
    @param debug_mode Enable/disable printing of costs for each possible step

    @return Arrays for optimal input (lambda deviation), velocity [m/s], gear,
            time [sec], fuel consumption [cc], and cost [competition points]
    
    """

    # Simulation parameters
    # init_vel = 19 # [m/s] FSAEM mean
    # init_vel = 15 # [m/s] FSAEL mean
    # dist_total = 40 # [m] FSAEM Mean
    # dist_total = 35 # [m] FSAEL Mean
    init_velF = init_vel # [m/s]
    init_velR = init_vel # [m/s]
    init_gear = find_gear4(init_vel*3600/1609.4, fdr, gears, redline, r)
    debug_mode = 0

    # Lambda (input) perturbation grid
    lambdadevs = np.arange(0, 0.42, 0.02)

    # Compute reachable time and approximated fuel consumption
    tstart = time.time()
    distance_steps = np.arange(0, dist_total, dist_interval)
    dist_iters = len(distance_steps)
    results = np.zeros((dist_iters, 2))
    vel_reach = np.zeros((dist_iters, 2))
    time_reach = np.zeros((dist_iters, 2))
    fuel_reach = np.zeros((dist_iters, 2))

    for i in [0, -1]:
        for j in range(dist_iters):
            lamdev = np.array([lambdadevs[i]])
            if j == 0:
                [accelTime, stateData] = accel_sim(input_FEPW, input_Torque_lbft,
                                                   dist_interval, init_velF,
                                                   init_velR, init_gear,
                                                   lamdev, 0)
            else:
                velF = stateData['vel [m/s]'].values[-1]
                velR = velF
                gear = round(stateData['gear'].values[-1])
                [accelTime, stateData] = accel_sim(input_FEPW, input_Torque_lbft,
                                                   dist_interval, velF,
                                                   velR, gear,
                                                   lamdev, 0)
            results[j, i] = accelTime
            vel_reach[j, i] = stateData['vel [m/s]'].values[-1]
            time_reach[j, i] = stateData['time [sec]'].values[-1]
            fuel_reach[j, i] = stateData['cumfuelcons [cc]'].values[-1]
    tend = time.time()
    print(f"Reachable space computation runtime = {tend-tstart:0.2f} sec")

     # Dynamic programming
    tstart = time.time()
    # Velocity grid is discretized state space of starting velocities for each spatial step
    vel_grid = np.zeros((dist_iters, vel_size))
    # Initialize grid with same velocity for k=0
    vel_grid[0,:] = init_vel*np.ones((1, vel_size))
    for j in range(1, dist_iters):
        vel_grid[j,:] = np.linspace(vel_reach[j-1, 0], vel_reach[j-1, 1], vel_size)

    j_opt = np.zeros((dist_iters, vel_grid.shape[1])) # j[k]*
    u_opt = np.zeros((dist_iters, vel_grid.shape[1])) # u[k]*
    vel_opt = np.zeros((dist_iters, 2*vel_grid.shape[1])) # x[k+1]*

    for k in range(dist_iters-1, -1, -1):
        for xIdx in range(vel_size):
            curr_V = vel_grid[k, xIdx]
            j_best = -np.inf;
            u_best = np.NaN;
            x_best = 0;
            if debug_mode: print("="*20, f"\nk={k}, xIdx={xIdx}", 
                                 "\nu, costCurr, costToGo, costNext")
            for uIdx in range(len(lambdadevs)):
                gear = find_gear4(curr_V*3600/1609.4, fdr, gears, redline, r)
                [accelTime, stateData] = accel_sim(input_FEPW, input_Torque_lbft,
                                                   dist_interval, curr_V,
                                                   curr_V, gear,
                                                   np.array([lambdadevs[uIdx]]), 0)
                x_next = stateData['vel [m/s]'].values[-1];
                # Time perturbation [sec]
                delta_t = stateData['time [sec]'].values[-1] - time_reach[k, 0];
                # Fuel perturbation [cc]
                delta_f = stateData['cumfuelcons [cc]'].values[-1] - fuel_reach[k, 0];
                costCurr = points_delta(delta_t, delta_f/1000)
                if k == dist_iters-1:
                    # Terminal Cost
                    # Using extrapolation and the assumption that velocity is
                    # constant at the end of the straight, calculate the time
                    # and fuel required to intersect the nominal braking
                    # trajectory. Use this for the terminal cost.
                    max_braking = 1.5 # g's
                    t_brake_nom = (vel_reach[-1, 0] - x_next)/(max_braking*g)
                    distance = vel_reach[-1, 0]*t_brake_nom - \
                               0.5*max_braking*g*t_brake_nom**2
                    # Extrapolation
                    t_brake_new = stateData['time [sec]'].values[-1]*distance/stateData['pos [m]'].values[-1]
                    f_brake_new = stateData['cumfuelcons [cc]'].values[-1]*distance/stateData['pos [m]'].values[-1]
                    # Assume no fuel used during braking, i.e., "f_brake_nom" is 0
                    costToGo = points_delta(t_brake_new-t_brake_nom, f_brake_new/1000)
                else:
                    costToGo = np.interp(x_next, vel_grid[k+1,:][::-1],
                                         j_opt[k+1,:][::-1]);
                costNext = costCurr + costToGo
                if debug_mode: print(lambdadevs[uIdx], costCurr, costToGo,
                                     costNext)
                if costNext > j_best: # Want to maximize points
                    # Record current best cost and input
                    j_best = costNext;
                    u_best = lambdadevs[uIdx];
                    x_best = np.array([stateData['vel [m/s]'].values[-1],
                                       stateData['rpm'].values[-1]]);
                # Store optimal cost and input for this initial velocity
                j_opt[k, xIdx] = j_best;
                u_opt[k, xIdx] = u_best;
                vel_opt[k, 2*xIdx:2*xIdx+2] = x_best;
            if debug_mode: print(f"u*={u_best}, costNext*={j_best}")
    tend = time.time()
    print(f"Dynamic programming runtime = {tend-tstart:0.2f} sec")

    # Generate optimal trajectory
    tstart = time.time()
    uk_opt = np.zeros((dist_iters, ))
    xk_opt = np.zeros((dist_iters+1, ))
    gear_opt = np.zeros((dist_iters+1, ), dtype=int)
    time_opt = np.zeros((dist_iters+1, ))
    fuel_opt = np.zeros((dist_iters+1, ))
    cost_opt = np.zeros((dist_iters+1, ))
    maxIdx = np.argmax(j_opt[0,:])
    uk_opt[0] = u_opt[0, maxIdx]
    xk_opt[0] = init_vel
    gear_opt[0] = init_gear
    time_opt[0] = 0
    fuel_opt[0] = 0
    cost_opt[0] = 0
    xk_opt[1] = vel_opt[0, ::2][maxIdx]
    gear_opt[1] = find_gear4(xk_opt[1]*3600/1609.4, fdr, gears, redline, r)
    [accelTime, stateData] = accel_sim(input_FEPW, input_Torque_lbft,
                                       dist_interval, xk_opt[0], xk_opt[0],
                                       gear_opt[0],
                                       np.array([uk_opt[0]]), 0)
    time_opt[1] = stateData['time [sec]'].values[-1]
    fuel_opt[1] = stateData['cumfuelcons [cc]'].values[-1]
    cost_opt[1] = points_delta(time_opt[1]-time_reach[0, 0],
                               (fuel_opt[1]-fuel_reach[0, 0])/1000)
    for i in range(1, dist_iters):
        uk_opt[i] = np.interp(xk_opt[i], vel_grid[i, :][::-1],
                              u_opt[i, :][::-1])
        [accelTime, stateData] = accel_sim(input_FEPW, input_Torque_lbft,
                                           dist_interval, xk_opt[i],
                                           xk_opt[i], gear_opt[i],
                                           np.array([uk_opt[i]]), 0)
        xk_opt[i+1] = stateData['vel [m/s]'].values[-1]
        gear_opt[i+1] = stateData['gear'].values[-1]
        currtime = stateData['time [sec]'].values[-1]
        currfuel = stateData['cumfuelcons [cc]'].values[-1]
        time_opt[i+1] = time_opt[i] + currtime
        fuel_opt[i+1] = fuel_opt[i] + currfuel
        cost_opt[i+1] = points_delta(currtime-time_reach[i, 0],
                                     (currfuel-fuel_reach[i, 0])/1000)
    # NOTE: cost_opt is an array of discretized nonlinear costs which
    # is not the same as the overall nonlinear cost using the total
    # time and fuel consumption, but it is close and might be informative
    # to look at graphically since it portrays the cost delta at each spatial
    # step as opposed to one final number
    # The error can be found by doing np.sum(cost_opt) - cost_opt_total
    # If the linear cost function is used instead (points_delta_lin),
    # then this error should be 0, ignoring the terminal cost penalty.
    cost_opt_total = points_delta(time_opt[-1]-np.sum(time_reach[:, 0]),
                                  (fuel_opt[-1]-np.sum(fuel_reach[:, 0]))/1000)
    # Add terminal cost penalty (intersecting with nominal braking trajectory)
    t_brake_nom = (vel_reach[-1, 0] - xk_opt[-1])/(max_braking*g)
    distance = vel_reach[-1, 0]*t_brake_nom - 0.5*max_braking*g*t_brake_nom**2
    t_brake_new = distance/xk_opt[-1]
    t_brake_new = currtime*distance/stateData['pos [m]'].values[-1]
    f_brake_new = currfuel*distance/stateData['pos [m]'].values[-1]
    # Assume no fuel used during braking, i.e., "f_brake_nom" is 0
    costToGo = points_delta(t_brake_new-t_brake_nom, f_brake_new/1000)
    cost_opt_total = cost_opt_total + costToGo
    tend = time.time()
    print(f"Optimal trajectory generation runtime = {tend-tstart:0.2f} sec")

    if plots == 1:
        # Font size code taken from
        # https://stackoverflow.com/questions/3899980/
        # how-to-change-the-font-size-on-a-matplotlib-plot
        plt.rcParams['figure.dpi'] = 100
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.figure(figsize=(10, 8))
        plt.suptitle(fr"Optimal Trajectory for $\Delta x={dist_total}$ m, $v_0={init_vel}$ m/s")
        # Plot optimal input trajectory as a function of rpm
        plt.subplot(2,2,1)
        plt.grid()
        for i in range(6):
            inds = np.where(gear_opt==i+1)[0]
            if len(inds) > 0:
                rpm_opt = xk_opt[:-1]*60/(2*np.pi*9.875*0.0254)*fdr*gears[i]
                plt.plot(rpm_opt[inds[0]:inds[-1]+1], uk_opt[inds[0]:inds[-1]+1],
                         'o-', markersize=3,
                         label=f'Optimal Input (Gear={i+1})')
        plt.hlines(y=lambdadevs[0], xmin=plt.gca().get_xlim()[0], xmax=redline,
                   color='k', linestyle='--', label='Input Boundary');
        plt.hlines(y=lambdadevs[-1], xmin=plt.gca().get_xlim()[0], xmax=redline,
                   color='k', linestyle='--');
        plt.legend(loc='best')
        plt.xlabel('RPM')
        plt.ylabel('Optimal Input')

        # Plot optimal input trajectory as a function of velocity
        plt.subplot(2,2,2)
        plt.grid()
        plt.plot(xk_opt[:-1], uk_opt, 'o-', markersize=3, label='Optimal Input')
        plt.hlines(y=lambdadevs[0], xmin=plt.gca().get_xlim()[0],
                   xmax=np.max(xk_opt[:-1]), color='k', linestyle='--',
                   label='Input Boundary');
        plt.hlines(y=lambdadevs[-1], xmin=plt.gca().get_xlim()[0],
                   xmax=np.max(xk_opt[:-1]), color='k', linestyle='--');
        plt.legend()
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Optimal Input')

        # Plot optimal velocity and velocity bounds as a function of position
        plt.subplot(2,2,3)
        plt.grid()
        plt.plot(np.hstack([distance_steps, dist_total]),
                 np.hstack([init_vel, vel_reach[:, 1]]),
                 'ko--', markersize=2, label='Vel Boundary')
        plt.plot(np.hstack([distance_steps, dist_total]),
                 np.hstack([init_vel, vel_reach[:, 0]]),
                 'ko--', markersize=2)
        plt.plot(np.hstack([distance_steps, dist_total]),
                 xk_opt, 'o-', markersize=3, label='Optimal Velocity')
        plt.legend(loc='best')
        plt.xlabel('Position [m]')
        plt.ylabel('Velocity [m/s]')

        # Plot optimal velocity and gear as a function of position
        plt.subplot(2,2,4)
        plt.grid()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(np.hstack([distance_steps, dist_total]), xk_opt,
                 'o-', markersize=3, label='Optimal Trajectory')
        ax2.plot(np.hstack([distance_steps, dist_total]), gear_opt,
                 '-', color='C1', markersize=3, label='Gear')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.set_xlabel('Position [m]')
        ax1.set_ylabel('Velocity [m/s]')
        ax2.set_ylabel('Gear', rotation=270, va='bottom')

        # Plot optimal time and time boundary
        plt.figure(figsize=(5, 4))
        plt.grid()
        plt.plot(np.hstack([distance_steps, dist_total]),
                 np.hstack([0, np.cumsum(time_reach[:, 1])]),
                 'ko--', markersize=2, label='Time Boundary')
        plt.plot(np.hstack([distance_steps, dist_total]),
                 np.hstack([0, np.cumsum(time_reach[:, 0])]),
                 'ko--', markersize=2)
        plt.plot(np.hstack([distance_steps, dist_total]),
                 time_opt, 'o-', markersize=3, label='Optimal Time')
        plt.legend(loc='best')
        plt.xlabel('Position [m]')
        plt.ylabel('Time [sec]')

        # Plot optimal fuel consumption and fuel boundary
        plt.figure(figsize=(5, 4))
        plt.grid()
        # This is not a guaranteed lower bound on fuel consumption but it will
        # be plotted for illustrative purposes
        plt.plot(np.hstack([distance_steps, dist_total]),
                 np.hstack([0, np.cumsum(fuel_reach[:, 1])]),
                 'ko--', markersize=2)
        plt.plot(np.hstack([distance_steps, dist_total]),
                 np.hstack([0, np.cumsum(fuel_reach[:, 0])]),
                 'ko--', markersize=2, label='Fuel Consumption Boundary')
        plt.plot(np.hstack([distance_steps, dist_total]),
                 fuel_opt, 'o-', markersize=3, label='Optimal Fuel Consumption')
        plt.legend(loc='best')
        plt.xlabel('Position [m]')
        plt.ylabel('Fuel Consumption [cc]')

        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
                            wspace = 0.4, hspace = 0.3)
        plt.show()

    return uk_opt, xk_opt, gear_opt, time_opt, fuel_opt, cost_opt_total