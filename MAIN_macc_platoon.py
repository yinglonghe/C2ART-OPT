#%%
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from itertools import product
from functools import partial
import c2art_env.sim_env.macc_platoon.setup_utils as macc_su


#%% 
if __name__ == "__main__":
    ####################
    # Paths and inputs #
    ####################
    Project = 'ASTAZERO'
    Database = 'Exp'
    Precision = 5   # Decimal
    pathRoot = os.path.dirname(__file__)
    pathInput = os.path.join(pathRoot, 'Data', Project, Database)
    pathResult = os.path.join(pathRoot, 'PyResults', Project, Database)
    Exp_Names = [
        'ASta_040719_platoon5',  # P-1 (ID)
        'ASta_040719_platoon6',  # P-2
        'ASta_040719_platoon7',  # P-3
        'ASta_040719_platoon8',  # P-4
        'ASta_040719_platoon9',  # P-5
        'ASta_050719_platoon1',  # P-6
        'ASta_050719_platoon2',  # P-7
    ]

    del pathRoot, Project, Database

    ##################
    # Initialization #
    ##################
    numVeh = 8          # Leading + following
    numFoll = numVeh-1
    dt = 0.1            # Time step (s)
    vehLen = 5          # Vehicle length (m)
    d = 5               # Initial spacing

    tau_p = [1.0]*numFoll
    tau_a = [0.1]*numFoll
    k_p = [0.053]*numFoll
    k_d = [0.293]*numFoll
    t_h = [1.937]*numFoll
    d_0 = [3.0]*numFoll
    w_macc = [0.5]*numFoll
    a_min = [-5.0]*numFoll
    a_max = [5.0]*numFoll
    veh_len = [5]*numVeh

    t0, p0, v0, a0 = macc_su.setup_data(pathInput, Exp_Names[0])
    numStep = len(t0)
    iniStep = max([int(tau_p[i] / dt) for i in range(len(tau_p))])

    Postion = np.zeros((numVeh, numStep))         # Postion of following vehicles
    Postion[0] = p0
    Velocity = np.zeros((numVeh, numStep))        # Velocity of following vehicles
    Velocity[0] = v0
    Acceleration = np.zeros((numVeh, numStep))    # Braking or tracking acceleration of following vehicles
    Acceleration[0] = a0
    Spacing = np.zeros((numVeh, numStep))         # Spacing
    U = np.zeros((numVeh, numStep))               # Desired braking or tracking acceleration of following vehicles
    
    for i in range(iniStep):
        for j in range(1, numVeh):
            Postion[j][i]  = Postion[j-1][i] - (d+vehLen)
            Velocity[j][i] = Velocity[j-1][i]
            Acceleration[j][i] = Acceleration[j-1][i]
            Spacing[j][i]  = Postion[j-1][i] - Postion[j][i]
    del i, j

    ##############
    # Simulation #
    ##############
    for i in range(iniStep, numStep):
        for j in range(1, numVeh):   # IDX of followers (1, 2, 3 ...)
            
            ipd = int(tau_p[j-1] / dt)  # Steps of perception delay of the ego vehicle

            state_pre_veh_p = np.array([
                Postion[j-1][i-1-ipd],
                Velocity[j-1][i-1-ipd],
                Acceleration[j-1][i-1-ipd],
            ])
            state_ego_veh_p = np.array([
                Postion[j][i-1-ipd],
                Velocity[j][i-1-ipd],
                Acceleration[j][i-1-ipd],
            ])

            if j == 1:
                U[j][i] = macc_su.acc_step(
                    state_pre_veh_p,
                    state_ego_veh_p,
                    veh_len[j-1],        # Length of pre veh
                    t_h[j-1],
                    d_0[j-1],
                    k_p[j-1],
                    k_d[j-1],
                )
            else:
                state_pre_veh_1_p = np.array([
                    Postion[j-2][i-1-ipd],
                    Velocity[j-2][i-1-ipd],
                    Acceleration[j-2][i-1-ipd],
                ])

                U[j][i] = macc_su.macc_step(
                    state_pre_veh_1_p,
                    state_pre_veh_p,
                    state_ego_veh_p,
                    veh_len[j-2],       # Length of pre_1 veh
                    veh_len[j-1],       # Length of pre veh
                    t_h[j-1],
                    d_0[j-1],
                    k_p[j-1],
                    k_d[j-1],
                    w_macc[j-1],
                )

            Postion[j][i], Velocity[j][i], Acceleration[j][i] = macc_su.vehicle_dynamic(
                U[j][i],  
                Postion[j][i-1],
                Velocity[j][i-1],
                Acceleration[j][i-1],
                dt,
                tau_a[j-1],
                a_min[j-1],
                a_max[j-1],
            )

            Spacing[j][i]  = Postion[j-1][i] - Postion[j][i]
            if Spacing[j][i] <= 0:
                break

    #############
    # Save data #
    #############
    col_ = ['Time'] + ['P'+str(m) for m in range(numVeh)] + \
        ['V'+str(m) for m in range(numVeh)] + ['A'+str(m) for m in range(numVeh)] + \
            ['S'+str(m) for m in range(1, numVeh)]

    data_ = np.vstack((t0, Postion, Velocity, Acceleration, Spacing[1:]))

    df_ = pd.DataFrame(data=data_.transpose(), columns=col_)
    if True:
        df_.to_csv('res_time_series.csv', index=False)

    #############
    # Plot data #
    #############
    plt.figure()
    for i in range(numVeh):
        plt.plot(t0, Velocity[i], label='v'+str(i))
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    for i in range(numVeh):
        plt.plot(t0, Acceleration[i], label='a'+str(i))
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s2)')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    for i in range(1, numVeh):
        plt.plot(t0, Spacing[i], label='s'+str(i))
    plt.xlabel('Time (s)')
    plt.ylabel('Spacing (m)')
    plt.legend()
    plt.show()
    plt.close()

