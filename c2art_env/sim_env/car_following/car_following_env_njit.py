import numpy as np
import pandas as pd
import c2art_env as env
from c2art_env.sim_env.car_following import car_following_action as cf_act
from numba import njit


@njit(nogil=True)
def run(parm, flow_ctrl, mfc_curve, data_exp, data_track, *args):

    control_type = flow_ctrl[0]
    spacing_type = flow_ctrl[1]
    ctrl_cstr_onoff = flow_ctrl[2]
    veh_dyn_type = flow_ctrl[3]
    veh_cstr_type = flow_ctrl[4]

    parm['veh_mass'] = parm['car_mass'] + parm['veh_load']
    parm['phi'] = parm['car_phi']

    track_len = parm['track_len']
    ########################################
    # Vehicle dynamics model configuration #
    ########################################

    if veh_dyn_type == 'car_none':
        parm_car = np.array([
            parm['veh_a_max'],  # 0
            parm['veh_a_min']   # 1
        ])
    elif veh_dyn_type == 'car_linear':
        parm_car = np.array([
            parm['tau_i'],      # 0 *
            parm['veh_a_max'],  # 1
            parm['veh_a_min']   # 2
        ])
    elif veh_dyn_type == 'car_nonlinear':
        parm_car = np.array([
            parm['tau_i'],      # 0 *
            parm['veh_mass'],   # 1 *
            parm['f_0'],        # 2 *
            parm['f_1'],        # 3 *
            parm['f_2'],        # 4 *
            parm['phi'],        # 5
            parm['veh_a_max'],  # 6
            parm['veh_a_min']   # 7
        ])
    else:
        raise Exception('Please select the vehicle dynamics model!')

    ###################################
    # ACC control model configuration #
    ###################################
    dt = parm['dt']  # Simulation time step (s)
    perception_delay = parm['perception_delay']     # # *
    len_pre_veh = parm['l_i_1']

    if control_type == 'acc_linear_d' and  spacing_type == 'd_cth':
        parm_acc = np.array([
            parm['k_p'],        # 0 *
            parm['k_v'],        # 1 *
            parm['k_a'],        # 2 *
            parm['k_set'],      # 3 *
            parm['v_set'],      # 4 *
            parm['d_0'],        # 5 *
            parm['t_h'],        # 6 *
            parm['acc_a_max'],  # 7
            parm['acc_a_min']   # 8
        ])
    elif control_type == 'acc_linear_d' and spacing_type == 'd_idm_des':
        parm_acc = np.array([
            parm['k_p'],        # 0 *
            parm['k_v'],        # 1 *
            parm['k_a'],        # 2 *
            parm['k_set'],      # 3 *
            parm['v_set'],      # 4 *
            parm['d_0'],        # 5 *
            parm['t_h'],        # 6 *
            parm['acc_a_max'],  # 7 *
            parm['acc_a_min']   # 8 *
        ])
    elif control_type == 'acc_linear_d' and spacing_type == 'd_gipps_eq':
        parm_acc = np.array([
            parm['k_p'],        # 0 *
            parm['k_v'],        # 1 *
            parm['k_a'],        # 2 *
            parm['k_set'],      # 3 *
            parm['v_set'],      # 4 *
            parm['d_0'],        # 5 *
            parm['t_h'],        # 6 *
            parm['teta'],       # 7 *
            parm['acc_a_min'],  # 8 *
            parm['est_a_min_p'],# 9 *
            parm['acc_a_max']   # 10    ADDED
        ])
    elif control_type == 'acc_linear_v' and spacing_type == 'v_cth':
        parm_acc = np.array([
            parm['k_1'],        # 0 *
            parm['k_2'],        # 1 *
            parm['d_0'],        # 2 *
            parm['t_h'],        # 3 *
            parm['v_set'],      # 4 *
            parm['acc_a_max'],  # 5
            parm['acc_a_min']   # 6
        ])
    elif control_type == 'acc_linear_v' and spacing_type == 'v_fvdm':
        parm_acc = np.array([
            parm['k_1'],        # 0 *
            parm['k_2'],        # 1 *
            parm['d_0'],        # 2 *
            parm['t_h'],        # 3 *
            parm['v_set'],      # 4 *
            parm['acc_a_max'],  # 5
            parm['acc_a_min']   # 6
        ])
    elif control_type == 'acc_linear_v' and spacing_type == 'v_gipps':
        parm_acc = np.array([
            parm['k_1'],        # 0 *
            parm['k_2'],        # 1 *
            parm['d_0'],        # 2 *
            parm['t_h'],        # 3 *
            parm['v_set'],      # 4 *
            parm['teta'],       # 5 *
            parm['acc_a_min'],  # 6 *
            parm['est_a_min_p'],# 7 *
            parm['acc_a_max']   # 8    ADDED
        ])
    elif control_type == 'acc_idm' and spacing_type == 'd_idm_des':
        parm_acc = np.array([
            parm['delta'],      # 0 *
            parm['v_set'],      # 1 *
            parm['d_0'],        # 2 *
            parm['t_h'],        # 3 *
            parm['acc_a_max'],  # 4 *
            parm['acc_a_min']   # 5 *
        ])
    elif control_type == 'acc_gipps' and spacing_type == 'd_none':
        parm_acc = np.array([
            parm['teta'],       # 0 *
            parm['v_set'],      # 1 *
            parm['d_0'],        # 2 *
            parm['t_h'],        # 3 *
            parm['acc_a_max'],  # 4 *
            parm['acc_a_min'],  # 5 *
            parm['est_a_min_p'] # 6 *
        ])
    else:
        raise Exception('Please select the ACC control model!')

    time = data_exp[0] - data_exp[0][0]
    # Read trajectories of the preceding vehicle
    pos_pre_veh = data_exp[1]
    speed_pre_veh = data_exp[3]
    accel_pre_veh = np.append((env.utils.numbadiff(speed_pre_veh) / env.utils.numbadiff(time)), 0)
    # 0-Position, 1-Speed, 2-Acceleration
    exp_pre_veh = np.vstack((pos_pre_veh, speed_pre_veh, accel_pre_veh)).transpose()

    # Read trajectories of the following (ego) vehicle
    pos_ego_veh = data_exp[2]
    speed_ego_veh = data_exp[4]
    accel_ego_veh = np.append((env.utils.numbadiff(speed_ego_veh) / env.utils.numbadiff(time)), 0)
    # 0-Position, 1-Speed, 2-Acceleration
    exp_ego_veh = np.vstack((pos_ego_veh, speed_ego_veh, accel_ego_veh)).transpose()
    
    ipd = int(perception_delay / dt)

    ###################
    # Simulation loop #
    ###################
    
    state_ego_veh, state_spacing, state_sin, success, count_cut_acc, count_cut_dec \
        = car_follwing_iterate(ipd, dt, time, exp_pre_veh, exp_ego_veh, len_pre_veh,\
            flow_ctrl, parm_acc, parm_car, mfc_curve, \
                data_track[0], data_track[1], track_len)
    
    ####################
    # Final trajectory #
    ####################

    exp_pre_veh = exp_pre_veh.transpose()
    exp_ego_veh = exp_ego_veh.transpose()
    exp_spacing = exp_pre_veh[0] - exp_ego_veh[0] - len_pre_veh
    state_ego_veh = state_ego_veh.transpose()

    # debug = np.empty(1, np.float64)      # OR    
    debug = np.vstack((
        time,
        exp_pre_veh[0],
        exp_pre_veh[1],
        exp_pre_veh[2],
        exp_ego_veh[0],
        exp_ego_veh[1],
        exp_ego_veh[2],
        exp_spacing,
        state_ego_veh[0], 
        state_ego_veh[1],
        state_ego_veh[2],
        state_spacing,
        state_sin
    ))

    # state_std_a = np.std(state_ego_veh[2][ipd+1:])
    # exp_i_std_a = np.std(exp_ego_veh[2][ipd+1:])
    # print(round(state_std_a/exp_i_std_a, 1))
    ####################################
    # Soft constraints for calibration #
    ####################################
    state_a_sum_abs = np.sum(np.abs(state_ego_veh[2][ipd+1:]))
    exp_i_a_sum_abs = np.sum(np.abs(exp_ego_veh[2][ipd+1:]))
    # print(state_a_sum_abs/exp_i_a_sum_abs-1)
    comfort = 1
    f_comf = 0.2
    if state_a_sum_abs > (1+f_comf) * exp_i_a_sum_abs:
        comfort = 0

    terminal = 1
    f_term = 1
    if np.abs(state_spacing[-1] - exp_spacing[-1]) > f_term:
        terminal = 0
    
    # Note: Comfort and terminal don't make sense when success=0
    
    #################################################
    # Hard constraints for calibration & validation #
    #################################################
    if success == 0:
        gof = [float(100000)]*17
        errors_d = np.empty(1, np.float64)
        errors_v = np.empty(1, np.float64)
        errors_a = np.empty(1, np.float64)
        return gof, count_cut_acc, count_cut_dec, success, comfort, terminal, \
            errors_d, errors_v, errors_a, debug

    state_std_v = np.std(state_ego_veh[1][ipd+1:])
    exp_i_std_v = np.std(exp_ego_veh[1][ipd+1:])
    errors_d = (state_spacing - exp_spacing)[ipd+1:]
    errors_v = (exp_ego_veh[1] - state_ego_veh[1])[ipd+1:]
    errors_a = (exp_ego_veh[2] - state_ego_veh[2])[ipd+1:]
    errors_std_v = state_std_v - exp_i_std_v
    errors_percentage_d = np.divide(errors_d, exp_spacing[ipd+1:])
    errors_percentage_v = np.divide(errors_v, env.utils.not_zero_array(exp_ego_veh[1][ipd+1:]))
    errors_percentage_std_v = np.divide(errors_std_v, exp_i_std_v)
    rmse_d = np.sqrt((errors_d**2).mean())
    rmse_v = np.sqrt((errors_v**2).mean())
    rmse_a = np.sqrt((errors_a**2).mean())
    rmspe_d = np.sqrt((errors_percentage_d**2).mean())
    rmspe_v = np.sqrt((errors_percentage_v**2).mean())
    rmspe_d_v = 0.5*rmspe_d + 0.5*rmspe_v
    rmspe_std_v = np.sqrt(errors_percentage_std_v**2)
    rmspe_std_v_d = 0.5*rmspe_std_v + 0.5*rmspe_d

    nrmse_d = rmse_d / np.sqrt((exp_spacing[ipd+1:]**2).mean())
    nrmse_v = rmse_v / np.sqrt((exp_ego_veh[1][ipd+1:]**2).mean())
    nrmse_a = rmse_a / np.sqrt((exp_ego_veh[2][ipd+1:]**2).mean())
    nrmse_d_v = nrmse_d + nrmse_v
    nrmse_d_v_a = nrmse_d + nrmse_v + nrmse_a

    u_d = rmse_d / (np.sqrt((state_spacing[ipd+1:]**2).mean()) + np.sqrt((exp_spacing[ipd+1:]**2).mean()))
    u_v = rmse_v / (np.sqrt((exp_ego_veh[1][ipd+1:]**2).mean()) + np.sqrt((state_ego_veh[1][ipd+1:]**2).mean()))
    u_a = rmse_a / (np.sqrt((exp_ego_veh[2][ipd+1:]**2).mean()) + np.sqrt((state_ego_veh[2][ipd+1:]**2).mean()))
    u_d_v = u_d + u_v
    u_d_v_a = u_d + u_v + u_a

    errors_dn = env.utils.min_max_normalize(state_spacing[ipd+1:], exp_spacing[ipd+1:]) - env.utils.min_max_normalize(exp_spacing[ipd+1:], exp_spacing[ipd+1:])
    errors_vn = env.utils.min_max_normalize(exp_ego_veh[1][ipd+1:], exp_ego_veh[1][ipd+1:]) - env.utils.min_max_normalize(state_ego_veh[1][ipd+1:], exp_ego_veh[1][ipd+1:])
    errors_an = env.utils.min_max_normalize(exp_ego_veh[2][ipd+1:], exp_ego_veh[2][ipd+1:]) - env.utils.min_max_normalize(state_ego_veh[2][ipd+1:], exp_ego_veh[2][ipd+1:])
    rmse_dn = np.sqrt((errors_dn**2).mean())
    rmse_vn = np.sqrt((errors_vn**2).mean())
    rmse_an = np.sqrt((errors_an**2).mean())

    rmse_dn_vn = rmse_dn + rmse_vn
    rmse_dn_vn_an = rmse_dn + rmse_vn + rmse_an

    gof = [
        float(rmse_d),          # 0
        float(rmse_v),          # 1
        float(rmse_a),          # 2
        float(rmspe_d),         # 3
        float(rmspe_v),         # 4
        float(nrmse_d),         # 5
        float(nrmse_v),         # 6
        float(nrmse_a),         # 7
        float(nrmse_d_v),       # 8
        float(nrmse_d_v_a),     # 9
        float(u_d),             # 10
        float(u_v),             # 11
        float(u_a),             # 12
        float(u_d_v),           # 13
        float(u_d_v_a),         # 14
        float(rmse_dn_vn),      # 15
        float(rmse_dn_vn_an),   # 16
    ]

    return gof, count_cut_acc, count_cut_dec, success, comfort, terminal, \
        errors_d, errors_v, errors_a, debug


@njit(nogil=True)
def car_follwing_iterate(ipd, dt, time, exp_pre_veh, exp_ego_veh, len_pre_veh,\
    flow_ctrl, parm_acc, parm_car, mfc_curve, \
        data_track_X, data_track_Y, track_len):

    state_ego_veh = np.zeros_like(exp_ego_veh)
    state_spacing = np.zeros_like(time)
    state_sin = np.zeros_like(time)

    count_cut_acc = 0
    count_cut_dec = 0
    flag_scs = 1
    success = 1

    for i in range(len(time)):
        if i <= ipd:
            state_ego_veh[i] = exp_ego_veh[i]

            # Detect negative radical and modify ego vehicle position 
            # at initial steps in Gipps models (M19-M36)
            if flow_ctrl[0] == 'acc_gipps' and True:
                teta = parm_acc[0]
                v_set = parm_acc[1]
                d_0 = parm_acc[2]
                t_h = parm_acc[3]
                ctrl_a_max = parm_acc[4]
                ctrl_a_min = parm_acc[5]
                accel_min_pre_veh_est = parm_acc[6]

                d_i = exp_pre_veh[i][0] - exp_ego_veh[i][0] - (len_pre_veh + d_0)
                sqrt_item = ctrl_a_min**2 * (0.5 * t_h + teta)**2 - ctrl_a_min * (2*d_i - exp_ego_veh[i][1]*t_h - exp_pre_veh[i][1]**2/accel_min_pre_veh_est)

                if sqrt_item < 0:
                    d_i = (ctrl_a_min * (0.5 * t_h + teta)**2 + exp_ego_veh[i][1]*t_h + exp_pre_veh[i][1]**2/accel_min_pre_veh_est) / 2
                    p_i = exp_pre_veh[i][0] - (len_pre_veh + d_0) - d_i

                    state_ego_veh[i] = np.array([p_i, exp_ego_veh[i][1], exp_ego_veh[i][2]])
        else:
            state_next, count_cut_acc_local, count_cut_dec_local, flag_scs = cf_act.step(
                exp_pre_veh[i-ipd-1],   # State of pre veh
                state_ego_veh[i-ipd-1],
                state_ego_veh[i-1],      # State of ego veh
                state_spacing[i-ipd-1],
                state_sin[i-1],          # Road slope
                len_pre_veh,        # Length of pre veh
                flow_ctrl,
                parm_acc, 
                parm_car,           # Model parameters
                mfc_curve,          # MFC constraints
                dt                 # Time interval (s)
            )  

            count_cut_acc += count_cut_acc_local
            count_cut_dec += count_cut_dec_local
    
            state_ego_veh[i] = state_next

        ############################
        # Update and check spacing #
        ############################
        state_spacing[i] =  exp_pre_veh[i][0] - state_ego_veh[i][0] - len_pre_veh
        if state_spacing[i] <= 0 or flag_scs == 0:
            success = 0
            break
        #####################
        # Update road slope #
        #####################
        state_sin[i] = env.utils.interp_grid_slope(data_track_X, data_track_Y, state_ego_veh[i][0] % track_len) # 0-Position
        # print(state_ego_veh[i])
    return state_ego_veh, state_spacing, state_sin, success, count_cut_acc, count_cut_dec