import os, math, time
import numpy as np
import pandas as pd
from c2art_env import utils
from c2art_env.ctrl_model import spacing_policy as sp
from c2art_env.veh_model.longitudinal_dyn_veh import linear_lon_dyn


def setup_data(pathInput, ExpName):

    labelsData = pd.read_csv(os.path.join(pathInput, 'PyData', 'Labels.csv'), index_col=None)
    experimentData = pd.read_csv(os.path.join(pathInput, 'PyData', ExpName+'.csv'), index_col=None)
    # Sample rate
    t_step_data = experimentData['Time'][1] - experimentData['Time'][0]

    labelsDataExp = labelsData.loc[labelsData['File'] == ExpName]

    K = experimentData.shape[0]

    a_real = experimentData[['Accel'+str(1)]].T.values.tolist()
    v_real = experimentData[['SpeedDoppler'+str(1)]].T.values.tolist()
    x_real = [[None]*K]
    t_real = [[utils.my_round(t_step_data * k, 1) for k in range(K)]]
    for k in range(K):
        if k == 0:
            x_real[0][k] = experimentData['X'+str(1)][0]
        else:
            x_real[0][k] = x_real[0][k-1] + ((v_real[0][k-1]+v_real[0][k])/2) * t_step_data

    return t_real[0], x_real[0], v_real[0], a_real[0]


def acc_step(
    state_pre_veh_p,    # 0-Position, 1-Speed, 2-Acceleration
    state_ego_veh_p,  
    len_pre_veh,        # Length of pre veh
    t_h,
    d_0,
    k_p,
    k_d,
    *args):

    pos_pre_veh = state_pre_veh_p[0]
    speed_pre_veh = state_pre_veh_p[1]
    accel_pre_veh = state_pre_veh_p[2]

    pos_ego_veh = state_ego_veh_p[0]
    speed_ego_veh = state_ego_veh_p[1]
    accel_ego_veh = state_ego_veh_p[2]

    spacing = pos_pre_veh - pos_ego_veh - len_pre_veh

    # Spacing policy: constant-time-headway (CTH)
    spacing_des = sp.constant_time_headway_spacing(
                    speed_ego_veh,
                    d_0,
                    t_h)

    delta_spacing = spacing - spacing_des
    delta_speed = speed_pre_veh - speed_ego_veh

    accel_des = k_d * delta_speed + k_p * delta_spacing
    
    return accel_des


def macc_step(
    state_pre_veh_1_p,   # 0-Position, 1-Speed, 2-Acceleration
    state_pre_veh_p,    # 0-Position, 1-Speed, 2-Acceleration
    state_ego_veh_p,  
    len_pre_veh_1,       # Length of pre1 veh
    len_pre_veh,        # Length of pre veh
    t_h,
    d_0,
    k_p,
    k_d,
    w_macc,
    *args):

    pos_pre_veh_1 = state_pre_veh_1_p[0]
    speed_pre_veh_1 = state_pre_veh_1_p[1]
    accel_pre_veh_1 = state_pre_veh_1_p[2]

    pos_pre_veh = state_pre_veh_p[0]
    speed_pre_veh = state_pre_veh_p[1]
    accel_pre_veh = state_pre_veh_p[2]

    pos_ego_veh = state_ego_veh_p[0]
    speed_ego_veh = state_ego_veh_p[1]
    accel_ego_veh = state_ego_veh_p[2]

    spacing = pos_pre_veh - pos_ego_veh - len_pre_veh
    spacing_1 = pos_pre_veh_1 - pos_ego_veh - len_pre_veh_1

    # Spacing policy: constant-time-headway (CTH)
    spacing_des = sp.constant_time_headway_spacing(
                    speed_ego_veh,
                    d_0,
                    t_h)

    delta_spacing = spacing - spacing_des
    delta_speed = speed_pre_veh - speed_ego_veh

    delta_spacing_1 = spacing_1 - 2*spacing_des
    delta_speed_1 = speed_pre_veh_1 - speed_ego_veh

    accel_des = k_d * (delta_speed + w_macc*delta_speed_1) + \
        k_p * (delta_spacing + w_macc*delta_spacing_1)

    return accel_des


def vehicle_dynamic(
    u,  
    Position,
    Velocity,
    Acceleration,
    timeStep, 
    tau_a,
    a_min,
    a_max,
    ):

    accel_next = linear_lon_dyn(
        Velocity,
        Acceleration,
        u,
        timeStep,
        tau_a)

    accel_next = utils.clip_min_max(accel_next, a_min, a_max)

    state_next = utils.motion_integ(
        Position,
        Velocity,
        accel_next,
        timeStep
    )

    return state_next[0], state_next[1], state_next[2]