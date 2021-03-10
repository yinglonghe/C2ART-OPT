import numpy as np
import pandas as pd
import c2art_env as env
from c2art_env.sim_env.free_flow_stable import free_flow_action as ff_act
from numba import njit


@njit(nogil=True)
def run(parm, flow_ctrl, mfc_curve, data_exp, data_track, *args):

    control_type = flow_ctrl[0]
    ctrl_cstr_onoff = flow_ctrl[1]
    turning_effect = flow_ctrl[2]
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
    v_set = parm['v_set']   # Set speed of ACC (m/s)
    perception_delay = parm['perception_delay']

    if control_type == 'acc_free_flow_pid' and turning_effect == 'off':
        parm_acc = np.array([
            parm['k_p'],        # 0 *
            parm['k_i'],        # 1 *
            parm['k_d'],        # 2 *
            parm['v_set'],      # 3
            parm['acc_a_max'],  # 4
            parm['acc_a_min']   # 5
        ])
    elif control_type == 'acc_free_flow_pid' and turning_effect == 'on':
        parm_acc = np.array([
            parm['k_p'],        # 0 *
            parm['k_i'],        # 1 *
            parm['k_d'],        # 2 *
            parm['a_turning'],  # 3 *
            parm['v_set'],      # 4
            parm['acc_a_max'],  # 5
            parm['acc_a_min']   # 6
        ])
    else:
        raise Exception('Please select the ACC control model!')

    time_veh = data_exp[0] - data_exp[0][0]

    # Read trajectories of the ego vehicle
    pos_ego_veh = data_exp[1]
    speed_ego_veh = data_exp[2]
    accel_ego_veh = np.append((env.utils.numbadiff(speed_ego_veh) / env.utils.numbadiff(time_veh)), 0)
    # 0-Position, 1-Speed, 2-Acceleration
    exp_ego_veh = np.vstack((pos_ego_veh, speed_ego_veh, accel_ego_veh)).transpose()
    
    ipd = int(perception_delay / dt)

    ###################
    # Simulation loop #
    ###################
    
    state_ego_veh, state_v_error, state_sin, state_curv, success, count_cut_acc, count_cut_dec \
        = car_follwing_iterate(ipd, dt, v_set, time_veh, exp_ego_veh,\
             flow_ctrl, parm_acc, parm_car, mfc_curve, data_track, track_len)
    
    ####################
    # Final trajectory #
    ####################

    exp_ego_veh = exp_ego_veh.transpose()
    exp_v_error = v_set - exp_ego_veh[1]
    state_ego_veh = state_ego_veh.transpose()

    # debug = np.empty(1, np.float64)      # OR    
    debug = np.vstack((
        time_veh,           # 0
        exp_ego_veh[0],     # 1 - Exp position (m)
        exp_ego_veh[1],     # 2 - Exp speed (m/s)
        exp_ego_veh[2],     # 3 - Exp acceleration (m/s2)
        exp_v_error,        # 4 - Exp speed deviation (m/s)
        state_ego_veh[0],   # 5 - Sim position (m)
        state_ego_veh[1],   # 6 - Sim speed (m/s)
        state_ego_veh[2],   # 7 - Sim acceleration (m/s2)
        state_v_error,      # 8 - Sim speed deviation (m/s)
        state_sin,          # 9
        state_curv,         # 10
    ))

    ####################################
    # Soft constraints for calibration #
    ####################################
    state_a_sum_abs = np.sum(np.abs(state_ego_veh[2][ipd+1:]))
    exp_i_a_sum_abs = np.sum(np.abs(exp_ego_veh[2][ipd+1:]))

    comfort = 1
    f_comf = 0.2
    if state_a_sum_abs > (1+f_comf) * exp_i_a_sum_abs:
        comfort = 0
    
    #################################################
    # Hard constraints for calibration & validation #
    #################################################
    if success == 0:
        gof = np.array([float(100000)]*13)
        return gof, count_cut_acc, count_cut_dec, success, comfort, debug

    state_std_v = np.std(state_ego_veh[1][ipd+1:])
    exp_i_std_v = np.std(exp_ego_veh[1][ipd+1:])
    errors_v = (exp_ego_veh[1] - state_ego_veh[1])[ipd+1:]
    errors_a = (exp_ego_veh[2] - state_ego_veh[2])[ipd+1:]
    errors_std_v = state_std_v - exp_i_std_v

    errors_percentage_v = np.divide(errors_v, env.utils.not_zero_array(exp_ego_veh[1][ipd+1:]))
    errors_percentage_std_v = np.divide(errors_std_v, exp_i_std_v)

    rmse_v = np.sqrt((errors_v**2).mean())
    rmse_a = np.sqrt((errors_a**2).mean())

    rmspe_v = np.sqrt((errors_percentage_v**2).mean())
    rmspe_std_v = np.sqrt(errors_percentage_std_v**2)

    nrmse_v = rmse_v / np.sqrt((exp_ego_veh[1][ipd+1:]**2).mean())
    nrmse_a = rmse_a / np.sqrt((exp_ego_veh[2][ipd+1:]**2).mean())
    nrmse_v_a = nrmse_v + nrmse_a

    u_v = rmse_v / (np.sqrt((exp_ego_veh[1][ipd+1:]**2).mean()) + np.sqrt((state_ego_veh[1][ipd+1:]**2).mean()))
    u_a = rmse_a / (np.sqrt((exp_ego_veh[2][ipd+1:]**2).mean()) + np.sqrt((state_ego_veh[2][ipd+1:]**2).mean()))
    u_v_a = u_v + u_a

    errors_vn = env.utils.min_max_normalize(exp_ego_veh[1][ipd+1:], exp_ego_veh[1][ipd+1:]) - env.utils.min_max_normalize(state_ego_veh[1][ipd+1:], exp_ego_veh[1][ipd+1:])
    errors_an = env.utils.min_max_normalize(exp_ego_veh[2][ipd+1:], exp_ego_veh[2][ipd+1:]) - env.utils.min_max_normalize(state_ego_veh[2][ipd+1:], exp_ego_veh[2][ipd+1:])
    rmse_vn = np.sqrt((errors_vn**2).mean())
    rmse_an = np.sqrt((errors_an**2).mean())
    rmse_vn_an = rmse_vn + rmse_an

    gof = np.array([
        float(rmse_v),          # 0
        float(rmse_a),          # 1
        float(rmspe_v),         # 2
        float(rmspe_std_v),     # 3
        float(nrmse_v),         # 4
        float(nrmse_a),         # 5
        float(nrmse_v_a),       # 6
        float(u_v),             # 7
        float(u_a),             # 8
        float(u_v_a),           # 9
        float(rmse_vn),         # 10
        float(rmse_an),         # 11
        float(rmse_vn_an),      # 12
    ])

    return gof, count_cut_acc, count_cut_dec, success, comfort, debug


@njit(nogil=True)
def car_follwing_iterate(ipd, dt, v_set, time_veh, exp_ego_veh, \
    flow_ctrl, parm_acc, parm_car, mfc_curve, data_track, track_len):

    data_track_X = data_track[0]
    data_track_SlopeSin = data_track[1]
    data_track_Curv = data_track[2]

    state_ego_veh = np.zeros_like(exp_ego_veh)
    state_v_error = np.zeros_like(time_veh)
    state_sin = np.zeros_like(time_veh)
    state_curv = np.zeros_like(time_veh)

    count_cut_acc = 0
    count_cut_dec = 0
    flag_scs = 1
    success = 1

    for i in range(len(time_veh)):
        if i <= ipd:
            state_ego_veh[i] = exp_ego_veh[i]
        else:
            state_next, count_cut_acc_local, count_cut_dec_local, flag_scs = ff_act.step(
                state_ego_veh[i-ipd-1],
                state_ego_veh[i-1],      # State of ego veh
                state_v_error[:i-1+1],
                state_sin[i-1],         # Road slope
                state_curv[i-1],        # Road curvature
                flow_ctrl,
                parm_acc, 
                parm_car,           # Model parameters
                mfc_curve,          # MFC constraints
                dt                 # Time interval (s)
            )

            count_cut_acc += count_cut_acc_local
            count_cut_dec += count_cut_dec_local
    
            state_ego_veh[i] = state_next

        ################################
        # Update and check speed error #
        ################################
        state_v_error[i] = v_set - state_ego_veh[i][1]

        if flag_scs == 0:
            success = 0
            break
        #####################
        # Update road slope #
        #####################
        state_sin[i] = env.utils.interp_binary(data_track_X, data_track_SlopeSin, state_ego_veh[i][0] % track_len) # 0-Position
        state_curv[i] = env.utils.interp_binary(data_track_X, data_track_Curv, state_ego_veh[i][0] % track_len) # 0-Position

        # print(state_ego_veh[i])
    return state_ego_veh, state_v_error, state_sin, state_curv, success, count_cut_acc, count_cut_dec
