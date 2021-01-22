from c2art_env.ctrl_model import spacing_policy as sp
import numpy as np
from numba import njit


@njit(nogil=True)
def linear_acc_d(
        state_pre_veh_p, # State of pre veh
        state_ego_veh_p, # State of ego veh
        spacing_type,
        len_pre_veh,
        k_p,
        k_v,
        k_a,
        k_set,
        v_set,
        d_0, 
        t_h, *args):

    pos_pre_veh = state_pre_veh_p[0]
    speed_pre_veh = state_pre_veh_p[1]
    accel_pre_veh = state_pre_veh_p[2]

    pos_ego_veh = state_ego_veh_p[0]
    speed_ego_veh = state_ego_veh_p[1]
    accel_ego_veh = state_ego_veh_p[2]

    spacing = pos_pre_veh - pos_ego_veh - len_pre_veh

    if spacing_type == 'd_cth':
        accel_max = args[0][0]
        accel_min = args[0][1]
        spacing_des = sp.constant_time_headway_spacing(
                            speed_ego_veh,
                            d_0,
                            t_h)

    if spacing_type == 'd_idm_des':
        accel_max = args[0][0]
        accel_min = args[0][1]
        spacing_des = sp.idm_desired_spacing(
                            speed_pre_veh,
                            speed_ego_veh,
                            d_0,
                            t_h,
                            accel_max,
                            accel_min)
        
    if spacing_type == 'd_gipps_eq':
        teta = args[0][0]
        accel_min = args[0][1]
        accel_min_pre_veh_est = args[0][2]
        spacing_des = sp.gipps_eq_spacing(
                            speed_ego_veh,
                            d_0,
                            t_h,
                            teta,
                            accel_min,
                            accel_min_pre_veh_est)

    delta_spacing = spacing_des - spacing
    delta_speed = speed_pre_veh - speed_ego_veh
    delta_accel = accel_pre_veh - accel_ego_veh

    accel_des = k_a * delta_accel + k_v * delta_speed - k_p * delta_spacing

    if delta_spacing > 0:
        accel_cmd = accel_des
    else:
        accel_cmd = min(k_set * (v_set - speed_ego_veh), accel_des)

    return accel_cmd


@njit(nogil=True)
def linear_acc_v(
        state_pre_veh_p, # State of pre veh
        state_ego_veh_p, # State of ego veh
        spacing_type,
        len_pre_veh,
        k_1,
        k_2,
        v_set,
        d_0, 
        t_h, *args):
    
    flag_scs = 1

    pos_pre_veh = state_pre_veh_p[0]
    speed_pre_veh = state_pre_veh_p[1]
    accel_pre_veh = state_pre_veh_p[2]

    pos_ego_veh = state_ego_veh_p[0]
    speed_ego_veh = state_ego_veh_p[1]
    accel_ego_veh = state_ego_veh_p[2]

    spacing = pos_pre_veh - pos_ego_veh - len_pre_veh

    if spacing_type == 'v_cth':
        accel_max = args[0][0]
        accel_min = args[0][1]
        speed_des = sp.constant_time_headway_speed(
                            speed_ego_veh,
                            spacing,
                            d_0,
                            t_h,
                            v_set)

    if spacing_type == 'v_fvdm':
        accel_max = args[0][0]
        accel_min = args[0][1]
        speed_des = sp.fvdm_speed(
                            speed_ego_veh,
                            spacing,
                            d_0,
                            t_h,
                            v_set)
        
    if spacing_type == 'v_gipps':
        teta = args[0][0]
        accel_min = args[0][1]
        accel_min_pre_veh_est = args[0][2]
        speed_des, flag_scs = sp.gipps_speed(
                            speed_ego_veh,
                            spacing,
                            d_0,
                            t_h,
                            v_set,
                            teta,
                            accel_min,
                            accel_min_pre_veh_est)

    delta_speed_pre = speed_pre_veh - speed_ego_veh
    delta_speed_des = speed_des - speed_ego_veh

    accel_cmd = k_1 * delta_speed_pre + k_2 * delta_speed_des

    return accel_cmd, flag_scs


@njit(nogil=True)
def idm_acc(
        state_pre_veh_p, # State of pre veh
        state_ego_veh_p, # State of ego veh
        spacing_type,
        len_pre_veh,
        delta,
        v_set,
        d_0, 
        t_h,
        accel_max,
        accel_min, *args):

    pos_pre_veh = state_pre_veh_p[0]
    speed_pre_veh = state_pre_veh_p[1]
    pos_ego_veh = state_ego_veh_p[0]
    speed_ego_veh = state_ego_veh_p[1]

    spacing = pos_pre_veh - pos_ego_veh - len_pre_veh


    if spacing_type == 'd_idm_des':
        spacing_des = sp.idm_desired_spacing(
            speed_pre_veh,
            speed_ego_veh,
            d_0,
            t_h,
            accel_max,
            accel_min)


    accel_cmd = accel_max * (1 - np.power(max(speed_ego_veh, 0) / v_set, delta)
                                             - np.power(spacing_des / spacing, 2))

    return accel_cmd


@njit(nogil=True)
def gipps_acc(
        state_pre_veh_p, # State of pre veh
        state_ego_veh_p, # State of ego veh
        speed_ego_veh_current,
        # spacing_type,
        len_pre_veh,
        teta,
        v_set,
        d_0, 
        t_h, 
        accel_max,
        accel_min,
        accel_min_pre_veh_est, *args):

    pos_pre_veh = state_pre_veh_p[0]
    speed_pre_veh = state_pre_veh_p[1]
    pos_ego_veh = state_ego_veh_p[0]
    speed_ego_veh = state_ego_veh_p[1]

    spacing = pos_pre_veh - pos_ego_veh - (len_pre_veh + d_0)

    tau = t_h

    # Detect the negative radical in simulation
    flag_scs = 1
    if accel_min**2 * (0.5 * tau + teta)**2 - accel_min * (2*spacing - speed_ego_veh*tau - speed_pre_veh**2/accel_min_pre_veh_est) < -1e-10:
        flag_scs = 0

    speed_ego_veh_cmd = min(
        speed_ego_veh + 2.5 * accel_max * tau * (1 - speed_ego_veh / v_set) * (0.025 + speed_ego_veh / v_set)**0.5,
        accel_min * (0.5 * tau + teta) + np.sqrt(max(0, accel_min**2 * (0.5 * tau + teta)**2 - 
                                            accel_min * (2*spacing - speed_ego_veh*tau - speed_pre_veh**2/accel_min_pre_veh_est)))
    )

    # speed_ego_veh_cmd = max(0, speed_ego_veh_cmd)         # Causing collisions at start (standstill) phase. Just a desired speed, not the actual!
    
    accel_cmd = (speed_ego_veh_cmd - speed_ego_veh_current) / tau

    return accel_cmd, flag_scs

