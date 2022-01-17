import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from tqdm import tqdm
import pandas as pd
from vehicleParams import *
from functions import findGear4
from dynProg import DP

# NOTE: Set vehicle parameters, e.g., engine torque, in the
# vehicleParams.py file

# Import straights data to iterate over
log_name = "StraightsData_FSAEL.csv"
straights_data = pd.read_csv(log_name)
if log_name == "StraightsData_FSAEM.csv":
    title = "FSAEM"
elif log_name == "StraightsData_FSAEL.csv":
    title = "FSAEL"
vels = straights_data['Init FL Vel [mph]'].values*1609.4/3600 # [m/s]
dists = straights_data['Distance [miles]'].values*1609.4 # [m]
dist_int = 0.1 # Spatial discretization length [m]
vel_size = 10 # State space grid width
num_CPU = 12 # Number of cores to use for multiprocessing

# Round distances since dynProg discretizes spatially at 1 meter by default.
# The rounding can be updated if a smaller disretization length is chosen.
dists = np.array([int(i) for i in np.round(dists)])    

# Sample run
# vels = np.array([10, 14, 18]) # [m/s]
# dists = np.array([20, 20, 30]) # [m/s]

# Shared Memory Preallocation
# Simulating varying distances results in arrays of various sizes being
# returned holding the optimal input and state sequences. As a result,
# some rows of each opts array will end up having NaN values to pad the 
# optimal sequence to fully fill the respective row. There might be a
# cleaner way to store the results but the current method works.
# Using int(round()) because int() simply truncates everything after the decimal
# while round() rounds to the nearest whole number (but still returns a float).
uopts = np.zeros((len(vels), int(np.round(np.max(dists)/dist_int)))) * np.NaN
xopts = np.zeros((len(vels), int(np.round(np.max(dists)/dist_int))+1)) * np.NaN
gearopts = np.zeros((len(vels), int(np.round(np.max(dists)/dist_int))+1)) * np.NaN
jopts = np.zeros((len(vels), ))

def parallelDP(i):
    """ Wrapper for DP function to be used for multiprocessing """
    [uopt, xopt, gearopt, _, _, jopt] = DP(init_vel=vels[i],
                                           dist_total=dists[i],
                                           dist_interval=dist_int,
                                           vel_size=vel_size)
    return uopt, xopt, gearopt, jopt

if __name__ == '__main__':

    print(f"Number of iterations = {len(vels)}")
    print(f"Discretization length = {dist_int} m")
    print(f"State space grid width = {vel_size}")

    tstart = time.time()

    pool = multiprocessing.Pool(num_CPU)
    # for ind, res in enumerate(pool.imap(parallelDP, range(len(vels)))):
    # Wrap pool.imap with tqdm to get printed progress statements
    for ind, res in enumerate(tqdm(pool.imap(parallelDP, range(len(vels))),
                              desc="Progress", total=len(vels))):
        uopts[ind, :len(res[0])] = res[0]
        xopts[ind, :len(res[1])] = res[1]
        gearopts[ind, :len(res[2])] = res[2]
        jopts[ind] = res[3]

    tend = time.time()
    print(f"Parallelized dynamic programming runtime = {tend-tstart:0.2f} sec")

    print(f"Points gain using unique optimal input for each" \
          f" iteration = {np.sum(jopts):0.2f} pts")

    tstart = time.time()

    # Compute the average optimal policy
    nanmin_xopts = np.nanmin(xopts)
    nanmax_xopts = 0
    for j in range(len(vels)):
        nanmax_curr = np.nanmax(xopts[j, :][~np.isnan(xopts[j, :])][:-1])
        if nanmax_curr > nanmax_xopts:
            nanmax_xopts = nanmax_curr
    shiftpts = [redline/(1/60*1609.4/(2*np.pi*r)*fdr*gears[i])*1609.4/3600 \
                for i in range(5)]
    vel_space = 0.2
    i = 0
    while nanmin_xopts > shiftpts[i]:
        i += 1
    policy_x = np.arange(nanmin_xopts, shiftpts[i+1] + vel_space, vel_space)
    while nanmax_xopts < shiftpts[i+1]:
        policy_xtmp = np.arange(shiftpts[i] + vel_space, shiftpts[i+1],
                                vel_space)
        policy_x = np.hstack([policy_x, policy_xtmp])
        i += 1
    policy_xtmp = np.arange(shiftpts[i+1] + vel_space, nanmax_xopts + vel_space,
                            vel_space)
    policy_x = np.hstack([policy_x, policy_xtmp])
    policy_x = np.arange(nanmin_xopts, nanmax_xopts + vel_space, vel_space)
    policy_size = len(policy_x)
    policy_gear = np.array([findGear4(vel*3600/1609.4, fdr, gears, redline, r)
                            for vel in policy_x])
    policy_u = np.zeros((policy_size, )) * np.NaN
    policy_u_stdev = np.zeros((policy_size, )) * np.NaN
    policy_u_samples = np.zeros((policy_size, ))
    policy_rpm = np.array([9500, 10000, 10500, 11000, 11500, 12000])
    for i in range(policy_size):
        xcurr = policy_x[i]
        # Find mean input (lambda deviation)
        us = []
        for j in range(len(vels)):
            # Don't consider NaN values in opts arrays
            xcurropts = xopts[j, :][~np.isnan(xopts[j, :])][:-1]
            ucurropts = uopts[j, :][~np.isnan(uopts[j, :])]
            if xcurr>=np.nanmin(xcurropts) and xcurr<=np.nanmax(xcurropts):
                us.append(np.interp(xcurr, xcurropts, ucurropts))
        if len(us) != 0:
            policy_u[i] = np.mean(us)
            policy_u_stdev[i] = np.std(us)
            policy_u_samples[i] = len(us)

    tend = time.time()
    print(f"Policy computation runtime = {tend-tstart:0.2f} sec")

    plt.figure()
    plt.grid()
    plt.title(f"{title}, {len(vels)} Straights")
    for i in range(len(vels)):
        plt.plot(xopts[i, :][:-1], uopts[i, :], zorder=1)
        plt.scatter(xopts[i, 0], uopts[i, 0], zorder=1e5)
    plt.plot(policy_x, policy_u, 'k--', linewidth=3, label='Mean', zorder=1e10)
    plt.fill_between(policy_x, policy_u - policy_u_stdev,
                     policy_u + policy_u_stdev, facecolor='k', alpha=0.25)
    plt.hlines(y=0, xmin=plt.gca().get_xlim()[0], xmax=np.max(policy_x),
               color='k', linestyle='--', label='Input Boundary');
    plt.hlines(y=0.4, xmin=plt.gca().get_xlim()[0], xmax=np.max(policy_x),
               color='k', linestyle='--');
    i = 0
    while np.max(policy_x) >= shiftpts[i]:
        plt.vlines(x=shiftpts[i], ymin=0, ymax=0.4, color="C{}".format(i),
                   linestyle=':', label=f"{i+1}-{i+2} Gear Shift");
        i += 1
    plt.legend(loc='best')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Input')

    # Plot optimal input trajectory as a function of rpm
    plt.figure()
    plt.grid()
    plt.title(f"{title}, {len(vels)} Straights")
    every = 2 # Plot every {every} sample
    for i in range(6):
        inds = np.where(policy_gear==i+1)[0]
        if len(inds) > 0:
            # num_samples = np.sum(policy_u_samples[inds[0]:inds[-1]+1])
            num_samples = np.count_nonzero(gearopts == i+1)
            rpm_opt = policy_x*60/(2*np.pi*9.875*0.0254)*fdr*gears[i]
            plt.plot(rpm_opt[inds[0]:inds[-1]+1][::every],
                     policy_u[inds[0]:inds[-1]+1][::every],
                     'o-', markersize=3,
                     label=f'Gear={i+1} (n={int(num_samples)})')
            plt.fill_between(rpm_opt[inds[0]:inds[-1]+1][::every],
                             policy_u[inds[0]:inds[-1]+1][::every] - \
                             policy_u_stdev[inds[0]:inds[-1]+1][::every],
                             policy_u[inds[0]:inds[-1]+1][::every] + \
                             policy_u_stdev[inds[0]:inds[-1]+1][::every], alpha=0.25)
    plt.hlines(y=0, xmin=plt.gca().get_xlim()[0], xmax=redline,
               color='k', linestyle='--', label='Input Boundary');
    plt.hlines(y=0.4, xmin=plt.gca().get_xlim()[0], xmax=redline,
               color='k', linestyle='--');
    plt.legend(loc='best')
    plt.xlabel('RPM')
    plt.ylabel('Input')

    plt.show()