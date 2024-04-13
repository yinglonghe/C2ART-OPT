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


def parameter_init(follIDs):

    mfc_veh_load = 250
    mfc_rolling_coef = 0.009
    mfc_aero_coef = {
        '33333': 0.27, 
        '1458': 0.35, 
        '8422': 0.27, 
        '50010': 0.30
    }
    mfc_res_coef_1 = 0.84
    mfc_res_coef_2 = -71.735
    mfc_res_coef_3 = 2.7609
    mfc_ppar_0 = 0.0045
    mfc_ppar_1 = -0.1710
    mfc_ppar_2 = -1.8835
    mfc_mh_base = 0.75

    ctrl_a_max = 3.0                           # Acceleration bound (m/s2)
    ctrl_a_min = -3.0                         # Deceleration bound (m/s2)
    a_max, a_min, f_0, f_1, f_2, veh_mass, phi, tau_a, mfc_array, mfc_slice = \
        [], [], [], [], [], [], [], [], [], [0]

    for i in range(len(follIDs)):
        mfc_model = mfc_curves(
            follIDs[i],
            mfc_veh_load,
            mfc_rolling_coef,
            mfc_aero_coef[str(follIDs[i])],
            mfc_res_coef_1,
            mfc_res_coef_2,
            mfc_res_coef_3,
            mfc_ppar_0,
            mfc_ppar_1,
            mfc_ppar_2,
            mfc_mh_base)

        total_mass = mfc_model['car_mass'] + mfc_veh_load

        f_0.append(mfc_model['mfc_f_0'])
        f_1.append(mfc_model['mfc_f_1'])
        f_2.append(mfc_model['mfc_f_2'])
        veh_mass.append(total_mass)
        tau_a.append(0.5 +(total_mass - 1000)/1000 * 0.3)     # Time lag
        phi.append(mfc_model['car_phi'])
        a_max.append(ctrl_a_max)
        a_min.append(ctrl_a_min)
        
        if i == 0:
            mfc_array = np.array([
                mfc_model['mfc_speed'],
                mfc_model['mfc_acc'],
                mfc_model['mfc_dec']
            ])
        else:
            mfc_array = np.append(mfc_array, 
                    np.array([
                        mfc_model['mfc_speed'],
                        mfc_model['mfc_acc'],
                        mfc_model['mfc_dec']
                    ]), axis=1)
        mfc_slice.append(len(mfc_array[0]))

    return np.array(a_max), np.array(a_min), np.array(f_0), np.array(f_1), np.array(f_2), \
        np.array(phi), np.array(tau_a), np.array(veh_mass), mfc_array, np.array(mfc_slice)


@njit
def leader_init(
    numStep,
    timeStep
    ):
    p0 = np.zeros(numStep)
    v0 = np.zeros(numStep)
    a0 = np.zeros(numStep) 

    # Transient process of leader, which is given in advance
    p0[0] = 0
    v0[0] = 20
    a0[int(1/timeStep):int(2/timeStep)] = 2

    ### Integration I ###
    # for i in range(1, numStep):
    #     v0[i] = v0[i-1] + a0[i] * timeStep
    #     p0[i] = p0[i-1] + v0[i] * timeStep

    ### Integration II ###
    for i in range(numStep-1):
        state_next = utils.motion_integ(
            p0[i],
            v0[i],
            a0[i+1],
            timeStep
        )
        p0[i+1], v0[i+1], a0[i+1]= state_next[0], state_next[1], state_next[2]

    return p0, v0, a0


@njit
def variable_init(
    numFoll,
    numStep,
    timeStep,
    Np,              # Predictive horizon 
    d,
    f_0, 
    f_1, 
    f_2, 
    phi, 
    tau_a,
    veh_mass, 
    mfc_array, 
    mfc_slice
    ):

    sin_theta = 0
    # Initial Virables 
    Postion = np.zeros((numFoll, numStep))         # Postion of following vehicles
    Velocity = np.zeros((numFoll, numStep))        # Velocity of following vehicles
    Acceleration = np.zeros((numFoll, numStep))    # Braking or tracking acceleration of following vehicles
    U = np.zeros((numFoll, numStep))               # Desired braking or tracking acceleration of following vehicles

    Cost = np.zeros((numFoll, numStep))            # Cost function
    Exitflg = np.zeros((numFoll, numStep))         # Stop flag - solvers

    # Leading vehicle
    p0, v0, a0 = leader_init(numStep, timeStep)

    # Zero initial error for the followers
    for i in range(numFoll):
        Postion[i][0]  = p0[0] - (i+1)*d
        Velocity[i][0] = v0[0]
        Acceleration[i][0] = a0[0]

    # Distributed MPC assumed state                          
    Pa = np.zeros((numFoll, Np+1))           # Assumed postion of each vehicle
    Va = np.zeros((numFoll, Np+1))           # Assumed velocity of each vehicle
    Aa = np.zeros((numFoll, Np+1))           # Assumed acceleration of each vehicle
    ua = np.zeros((numFoll, Np))           # Assumed Braking or Tracking Torque input of each vehicle

    Pa_next = np.zeros((numFoll, Np+1))    # 1(0): Assumed postion of each vehicle at the next time step
    Va_next = np.zeros((numFoll, Np+1))    # Assumed velocity of each vehicle at the next time step
    Aa_next = np.zeros((numFoll, Np+1))
    ua_next = np.zeros((numFoll, Np+1))    # Assumed Braking or Tracking acceleration of each vehicle at the next time step

    # Initialzie the assumed state for the first computation: constant speed
    for i in range(numFoll):
        ua[i] = (f_0[i] * (1 - sin_theta**2)**0.5 + f_1[i] * Velocity[i][0] + f_2[i] * Velocity[i][0]**2) / veh_mass[i] + 9.80665 * sin_theta
        Pa[i][0] = Postion[i][0]                # The first point should be interpreted as k = 0 (current state)
        Va[i][0] = Velocity[i][0]
        Aa[i][0] = Acceleration[i][0]
        for j in range(Np):
            Pa[i][j+1], Va[i][j+1], Aa[i][j+1] = vehicle_dynamic(
                ua[i][j],
                Pa[i][j],
                Va[i][j],
                Aa[i][j],
                timeStep,
                sin_theta, 
                f_0[i],
                f_1[i],
                f_2[i],
                phi[i],
                tau_a[i],
                veh_mass[i],
                mfc_array[:, mfc_slice[i]:mfc_slice[i+1]])
    # For debugging
    # Terminal state
    Pend = np.zeros((numFoll, numStep))
    Vend = np.zeros((numFoll, numStep))
    Aend = np.zeros((numFoll, numStep))

    return Postion, Velocity, Acceleration, U, p0, v0, a0, Pa, Va, Aa, ua, \
        Pa_next, Va_next, Aa_next, ua_next, Pend, Vend, Aend, Cost, Exitflg


@njit
def vehicle_dynamic(
    u,  
    Position,
    Velocity,
    Acceleration,
    timeStep, 
    sin_theta,
    f_0, 
    f_1, 
    f_2, 
    phi, 
    tau_a,
    veh_mass,
    mfc_curve
    ):

    accel_next = nonlinear_lon_dyn(
        Velocity, 
        Acceleration, 
        u,
        sin_theta, 
        timeStep,
        phi, 
        tau_a,
        veh_mass,
        f_0,
        f_1,
        f_2)

    # accel_next = linear_lon_dyn(
    #     Velocity,
    #     Acceleration,
    #     u,
    #     timeStep,
    #     tau_a)

    # accel_next = none_lon_dyn(
    #     Velocity, 
    #     u)

    mfc_a_max = utils.interp_binary(mfc_curve[0], mfc_curve[1], Velocity)
    mfc_a_min = utils.interp_binary(mfc_curve[0], mfc_curve[2], Velocity)

    accel_next = utils.clip_min_max(accel_next, mfc_a_min, mfc_a_max)

    ### Integration I ###
    # v_next = Velocity + accel_next * timeStep
    # p_next = Position + v_next * timeStep
    # state_next = np.array([
    #     p_next,
    #     v_next,
    #     accel_next
    # ])

    ### Integration II ###
    state_next = utils.motion_integ(
        Position,
        Velocity,
        accel_next,
        timeStep
    )

    return state_next[0], state_next[1], state_next[2]


def communication_init(
    numFoll,
    TopoType
    ):
    G = nx.DiGraph()
    ###############
    # Vertice set #
    ###############
    G.add_nodes_from(range(1, numFoll+1))
    ############
    # Edge set #
    ############
    if TopoType in ['PF', 'PLF']:
        for i in G.nodes:
            if i+1 in G.nodes:
                G.add_edge(i, i+1)
    elif TopoType in ['BD', 'BDL']:
        for i in G.nodes:
            if i-1 in G.nodes:
                G.add_edge(i, i-1)
            if i+1 in G.nodes:
                G.add_edge(i, i+1)
    elif TopoType in ['TPF', 'TPLF']:
        for i in G.nodes:
            if i+1 in G.nodes:
                G.add_edge(i, i+1)
            if i+2 in G.nodes:
                G.add_edge(i, i+2)
    matA = nx.adjacency_matrix(G).todense().T              # Adjacency matrix
    matI = nx.incidence_matrix(G, oriented=True).todense()  # Incidence matrix
    matDin = np.diag([G.in_degree(i) for i in G.nodes])
    matDout = np.diag([G.out_degree(i) for i in G.nodes])
    matL = matDin - matA                                    # Laplace matrix
    ### In-neighbour set ###
    setPred = [list(G.predecessors(i)) for i in G.nodes]
    ### Out-neighbour set ###
    setSucc = [list(G.successors(i)) for i in G.nodes]
    #######################
    # Pinning information #
    #######################
    if TopoType in ['PF', 'BD']:
        matP = np.zeros((numFoll, numFoll))
        matP[0, 0] = 1
    elif TopoType in ['TPF']:
        matP = np.zeros((numFoll, numFoll))
        matP[0, 0] = 1
        matP[1, 1] = 1
    elif TopoType in ['PLF', 'BDL', 'TPLF']:
        matP = np.eye(numFoll)
    ### Pinning in-neighbour set ###
    setIN = []
    for i in range(len(G.nodes)):
        if matP[i, i] == 1:
            setIN.append([0]+setPred[i]) 
        elif matP[i, i] == 0:
            setIN.append(setPred[i])

    # print(matA)
    # print(matI)
    # print(matDin)
    # print(matDout)
    # print(matL)
    # print(setPred)
    # print(setSucc)
    # print(matP)
    # print(setIN)

    # nx.draw(G, with_labels = True)
    # plt.show()
    return matA, matP


@njit
def mpc_func(
    Np, 
    timeStep, 
    sin_theta,
    X0, 
    u, 
    vehType, 
    mfc_curve
    ):

    f_0 = vehType[0]
    f_1 = vehType[1]
    f_2 = vehType[2]
    phi = vehType[3]
    tau_a = vehType[4]
    veh_mass = vehType[5]

    
    Pp = np.zeros(Np+1)     # Predictive position (m)
    Vp = np.zeros(Np+1)     # Predictive velocity (m/s)
    Ap = np.zeros(Np+1)     # Predictive acceleration (m/s2)

    Pp[0], Vp[0], Ap[0] = X0[0], X0[1], X0[2]

    for i in range(Np):
        Pp[i+1], Vp[i+1], Ap[i+1] = vehicle_dynamic(
            u[i],  
            Pp[i], 
            Vp[i], 
            Ap[i],
            timeStep, 
            sin_theta,
            f_0, 
            f_1, 
            f_2, 
            phi, 
            tau_a,
            veh_mass,
            mfc_curve)

    Xp = np.vstack((Pp, Vp, Ap))    # Predictive state

    return Xp


@njit
def nonlcon_func(u, *args):
    # args = args[0]

    Np = args[0]
    timeStep = args[1]
    sin_theta = args[2]
    X0 = args[3]
    vehType = args[4]
    mfc_curve = args[5]
    Pnp = args[6]
    Vnp = args[7]
    Anp = args[8]

    Xp = mpc_func(
        Np, 
        timeStep, 
        sin_theta,
        X0, 
        u, 
        vehType, 
        mfc_curve
    )
    Pp = Xp[0]
    Vp = Xp[1]
    Ap = Xp[2]
    
    Ceq = np.array([
        Pp[Np]-Pnp,
        Vp[Np]-Vnp,
        Ap[Np]-Anp
    ])

    return Ceq


@njit
def cost_func(u, *args):
    # args = args[0]

    Np = args[0]
    timeStep = args[1]
    sin_theta = args[2]
    X0 = args[3]
    vehType = args[4]
    mfc_curve = args[5]
    Q = args[6]
    Xdes = args[7].transpose()
    R = args[8]
    F = args[9]
    Xa = args[10].transpose()
    G = args[11]
    Xn = args[12].transpose()
    numXn = args[13]

    f_0 = vehType[0]
    f_1 = vehType[1]
    f_2 = vehType[2]
    phi = vehType[3]
    tau_a = vehType[4]
    veh_mass = vehType[5]

    dim = Xa.shape[1]

    Xp = mpc_func(
        Np, 
        timeStep, 
        sin_theta,
        X0, 
        u, 
        vehType, 
        mfc_curve
    )

    Vp = Xp[1]
    Xp = Xp[:dim].transpose()

    Udes = (f_0 * (1 - sin_theta**2)**0.5 + f_1 * Vp + f_2 * Vp**2) / veh_mass + 9.80665 * sin_theta
    
    cost = np.zeros(Np)     # Cost array

    for i in range(Np):
        cost_des = (Xp[i]-Xdes[i]) @ Q @ (Xp[i]-Xdes[i])
        cost_u = (u[i]-Udes[i]) * R * (u[i]-Udes[i])
        cost_self = (Xp[i]-Xa[i]) @ F @ (Xp[i]-Xa[i])
        cost_nbr = 0
        for j in range(numXn):
            cost_nbr += (Xp[i]-Xn[i, dim*j:dim*j+dim]) @ G @ (Xp[i]-Xn[i, dim*j:dim*j+dim])

        cost[i] = cost_des + cost_u + cost_self + cost_nbr

        cost_sum = np.sum(cost)

    return cost_sum


def mpc_simulation(
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
    ):

    numFoll = numVeh-1

    Postion, Velocity, Acceleration, U, p0, v0, a0, Pa, Va, Aa, ua, \
    Pa_next, Va_next, Aa_next, ua_next, Pend, Vend, Aend, Cost, Exitflg = variable_init( \
        numFoll, numStep, timeStep, Np, d, f_0, f_1, f_2, \
            phi, tau_a, veh_mass, mfc_array, mfc_slice)

    vehTypeCol = np.vstack((
        f_0,     # 0
        f_1,     # 1
        f_2,     # 2
        phi,     # 3
        tau_a,   # 4
        veh_mass,# 5
    )).transpose()

    for i in range(1, numStep-Np):
        print('Steps i='+str(i))
        for j in range(numFoll):   # IDX of followers (0, 1, 2, 3 ...)
            start_time = time.time()

            vehType = vehTypeCol[j]
            mfc_curve = mfc_array[:, mfc_slice[j]:mfc_slice[j+1]]

            X0 = np.array([
                Postion[j][i-1], 
                Velocity[j][i-1], 
                Acceleration[j][i-1],
            ])

            # Input-deviation
            R = 0.1

            # Self-deviation
            Xa = np.vstack((Pa[j], Va[j], Aa[j]))
            F = np.diag([5.0, 2.5, 1.0]) * (np.sum(matA[:,j])+1)**2

            # Leader-deviation
            if matP[j, j] == 1:
                numXdes = 1
                Pd = p0[i-1:i+Np] - (j+1)*d
                Vd = v0[i-1:i+Np]
                Ad = a0[i-1:i+Np]
                Xdes = np.vstack((Pd, Vd, Ad))
                Q = np.diag([5.0, 2.5, 1.0])
            else:
                numXdes = 0
                Xdes = np.zeros((3, Np+1))
                Q = np.diag([0.0, 0.0, 0.0])

            # Neighbor-deviation
            numXn = np.sum(matA[j,:])
            if numXn == 0:
                Xn = np.zeros((3, Np+1))
                G = np.diag([0.0, 0.0, 0.0])
            else:
                Xn = np.zeros((0, Np+1))
                for k in range(numFoll):
                    if matA[j,k] == 1:
                        Xn = np.vstack((Xn, Pa[k] - (j-k)*d, Va[k], Aa[k]))
                G = np.diag([5.0, 2.5, 1.0])
            
            lb = a_min[j] * np.ones(Np)
            ub = a_max[j] * np.ones(Np) 
            u0 = ua[j]

            Pnp = Xdes[0][-1]
            Vnp = Xdes[1][-1]
            Anp = Xdes[2][-1]

            for k in range(numXn):
                Pnp += Xn[3*k][-1]
                Vnp += Xn[3*k+1][-1]
                Anp += Xn[3*k+2][-1]
            
            Pnp = Pnp/(numXdes+numXn)
            Vnp = Vnp/(numXdes+numXn)
            Anp = Anp/(numXdes+numXn)

            Pend[j][i] = Pnp
            Vend[j][i] = Vnp
            Aend[j][i] = Anp

            sin_theta = 0
            args_cost=(
                Np,                 # 0
                timeStep,           # 1
                sin_theta,          # 2
                X0,                 # 3
                vehType,            # 4
                mfc_curve,          # 5
                Q,                  # 6
                Xdes,               # 7
                R,                  # 8
                F,                  # 9
                Xa,                 # 10
                G,                  # 11
                Xn,                 # 12
                numXn               # 13
            )
            args_cons=(
                Np,                 # 0
                timeStep,           # 1
                sin_theta,          # 2
                X0,                 # 3
                vehType,            # 4
                mfc_curve,          # 5
                Pnp,                # 6
                Vnp,                # 7
                Anp                 # 8
            )
            # bnds = tuple([(lb[k], ub[k]) for k in range(len(lb))])
            bnds = Bounds(lb, ub)
            cons = ({'type': 'eq',
                'fun': nonlcon_func,
                'args': args_cons
                })
            res = minimize(cost_func, u0, bounds=bnds, args=args_cost, method='SLSQP', constraints=cons, \
                tol = 1e-5,
                options={
                    'ftol': 1e-5,
                    # 'eps': 1e-4,
                    'maxiter': 2000, 
                    # 'disp': True
                    }
                )

            Cost[j][i] = res.fun
            u = res.x
            Exitflg[j][i] = res.status
            nit = res.nit

            U[j][i] = u[0]
            
            Postion[j][i], Velocity[j][i], Acceleration[j][i] = vehicle_dynamic(
                U[j][i],  
                Postion[j][i-1],
                Velocity[j][i-1],
                Acceleration[j][i-1],
                timeStep, 
                sin_theta,
                f_0[j], 
                f_1[j], 
                f_2[j], 
                phi[j], 
                tau_a[j],
                veh_mass[j],
                mfc_curve)

            Temp = np.zeros((3, Np+1))
            Temp[0][0], Temp[1][0], Temp[2][0] = Postion[j][i], Velocity[j][i], Acceleration[j][i]
            ua[j][:Np-1] = u[1:]

            for k in range(Np-1):
                Temp[0][k+1], Temp[1][k+1], Temp[2][k+1] = vehicle_dynamic(
                    ua[j][k],  
                    Temp[0][k], 
                    Temp[1][k], 
                    Temp[2][k],
                    timeStep, 
                    sin_theta,
                    f_0[j], 
                    f_1[j], 
                    f_2[j], 
                    phi[j], 
                    tau_a[j],
                    veh_mass[j],
                    mfc_curve)

            ua[j][Np-1] = (f_0[j] * (1 - sin_theta**2)**0.5 + f_1[j] * Temp[1][Np-1] + f_2[j] * Temp[1][Np-1]**2) / veh_mass[j] + 9.80665 * sin_theta
            
            Temp[0][Np], Temp[1][Np], Temp[2][Np] = vehicle_dynamic(
                ua[j][Np-1],  
                Temp[0][Np-1], 
                Temp[1][Np-1], 
                Temp[2][Np-1],
                timeStep, 
                sin_theta,
                f_0[j], 
                f_1[j], 
                f_2[j], 
                phi[j], 
                tau_a[j],
                veh_mass[j],
                mfc_curve)

            Pa_next[j] = Temp[0]
            Va_next[j] = Temp[1]
            Aa_next[j] = Temp[2]
            
            print(time.time() - start_time)

        Pa = Pa_next
        Va = Va_next
        Aa = Aa_next

    return U, Postion, Velocity, Acceleration, p0, v0, a0, Pend, Vend, Aend

