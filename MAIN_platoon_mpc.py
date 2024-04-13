import os, math, time, datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import njit, types
from numba.typed import Dict
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import networkx as nx
from c2art_env.veh_model.mfc.mfc_constraints import mfc_curves
from c2art_env.veh_model.longitudinal_dyn_veh import nonlinear_lon_dyn, linear_lon_dyn, none_lon_dyn
from c2art_env import utils
from c2art_env.sim_env.platooning_mpc import setup_utils as su


if __name__ == '__main__':

    ### [33333, 1458, 8422, 50010] ###
    carIDs = [33333, 33333, 1458, 8422, 50010, 1458, 8422, 50010]   # Including the platoon leader
    numVeh = len(carIDs)                    # The number of vehicles in a platoon (incl. the leader)
    timeStep = 0.1                          # Time step
    timeSim = 10                            # Trip duration
    numStep = math.floor(timeSim/timeStep)  # Simulation setps
    Np = 20                                 # Number of prediction horizon (steps)
    d  = 20                                 # Desired spacing
    Time = np.arange(0, timeSim, timeStep)

    a_max, a_min, f_0, f_1, f_2, phi, tau_a, veh_mass, mfc_array, mfc_slice \
        = su.parameter_init(carIDs[1:])

    ### ['PF', 'PLF', 'BD', 'BDL', 'TPF', 'TPLF'] ###
    matA, matP = su.communication_init(numVeh-1, 'TPLF')

    U, Postion, Velocity, Acceleration, p0, v0, a0, \
        Pend, Vend, Aend = su.mpc_simulation(
        d,
        Np,
        numVeh,
        numStep,
        timeStep,
        # Communication topology
        matA, 
        matP,
        # Vehicle type
        a_max, 
        a_min, 
        f_0, 
        f_1, 
        f_2, 
        veh_mass, 
        phi, 
        tau_a, 
        mfc_array, 
        mfc_slice
    )
    
    col_ = ['Time'] + ['P'+str(m) for m in range(numVeh)] + \
        ['V'+str(m) for m in range(numVeh)] + ['A'+str(m) for m in range(numVeh)] + \
            ['U'+str(m) for m in range(1, numVeh)]

    data_ = np.vstack((Time, p0, Postion, v0, Velocity, a0, Acceleration, U))

    df_ = pd.DataFrame(data=data_[:,:-Np].transpose(), columns=col_)
    if True:
        df_.to_csv('mpc_time_series.csv', index=False)

    if True:
        plt.figure()
        plt.plot(a0[:-Np])
        for i in range(numVeh-1):
            plt.plot(Acceleration[i][:-Np])
        plt.title('Acceleration')
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(v0[:-Np])
        for i in range(numVeh-1):
            plt.plot(Velocity[i][:-Np])
        plt.title('Velocity')
        plt.show()
        plt.close()

        plt.figure()
        for i in range(numVeh-1):
            plt.plot(Postion[i][:-Np] - (p0[:-Np] - (i+1)*d))
        plt.title('LEAD Pos Error')
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(p0[:-Np] - d - Postion[0][:-Np])
        for i in range(1, numVeh-1):
            plt.plot(Postion[i-1][:-Np] - d - Postion[i][:-Np])
        plt.title('NRB Pos Error')
        plt.show()
        plt.close()

        plt.figure()
        for i in range(numVeh-1):
            plt.plot(U[i][:-Np])
        plt.title('U')
        plt.show()
        plt.close()

        plt.figure()
        for i in range(numVeh-1):
            plt.plot(Pend[i][1:-Np])
        plt.title('Pend')
        plt.show()
        plt.close()

        plt.figure()
        for i in range(numVeh-1):
            plt.plot(Vend[i][1:-Np])
        plt.title('Vend')
        plt.show()
        plt.close()

        plt.figure()
        for i in range(numVeh-1):
            plt.plot(Aend[i][1:-Np])
        plt.title('Aend')
        plt.show()
        plt.close()
