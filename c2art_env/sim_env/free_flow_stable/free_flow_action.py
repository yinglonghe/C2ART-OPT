import numpy as np
from numba import njit
import c2art_env.ctrl_model.acceleration_ctrl as ctrl
import c2art_env.veh_model.longitudinal_dyn_veh as veh_dyn
import c2art_env as env


@njit(nogil=True)
def step(
        state_ego_veh_p,  
        state_ego_veh,      # 0-Position, 1-Speed, 2-Acceleration
        state_v_error,
        sin_theta,          # Road slope
        curvature,
        flow_ctrl,
        parm_ctrl, 
        parm_veh,           # Model parameters
        mfc_curve,
        dt, *args):         # Time interval (s)

    count_cut_acc = 0
    count_cut_dec = 0
    flag_scs = 1
    ########################################
    # Stop and go control (state override) #
    ########################################
    # if state_pre_veh_p[1] < 1 and state_spacing_p < 6:
    #     accel_next = -10
    #     state_next = env.utils.motion_integ(
    #         state_ego_veh[0],   # 0-Position
    #         state_ego_veh[1],   # 1-Speed
    #         accel_next,
    #         dt
    #     )
    #     return state_next, count_cut_acc, count_cut_dec

    ################
    # Flow control #
    ################
    control_type = flow_ctrl[0]
    ctrl_cstr_onoff = flow_ctrl[1]
    turning_effect = flow_ctrl[2]
    veh_dyn_type = flow_ctrl[3]
    veh_cstr_type = flow_ctrl[4]

    ##################
    # ACC controller #
    ##################
    if control_type == 'acc_free_flow_pid':
        v_err = state_v_error[-1]
        v_err_i = np.sum(state_v_error * dt)
        if len(state_v_error) == 1:
            v_err_d = 0
        else:
            v_err_d = (state_v_error[-1] - state_v_error[-2]) / dt

        if turning_effect == 'off':
            # Calibrating
            k_p = parm_ctrl[0]
            k_i = parm_ctrl[1]
            k_d = parm_ctrl[2]
            # Not calibrating
            v_set = parm_ctrl[3]
            ctrl_a_max = parm_ctrl[4]
            ctrl_a_min = parm_ctrl[5]

            parm_args = np.array([0.0])

        elif turning_effect == 'on':
            # Calibrating
            k_p = parm_ctrl[0]
            k_i = parm_ctrl[1]
            k_d = parm_ctrl[2]
            a_turning = parm_ctrl[3]
            # Not calibrating
            v_set = parm_ctrl[4]
            ctrl_a_max = parm_ctrl[5]
            ctrl_a_min = parm_ctrl[6]

            parm_args = np.array([a_turning])

        accel_cmd = ctrl.pid_free_flow_acc(
            k_p,
            k_i,
            k_d,
            v_err,
            v_err_i,
            v_err_d,
            turning_effect,
            state_ego_veh_p,
            curvature,
            parm_args           # *args[0]
        )

    ##########################
    # Controller constraints #
    ##########################
    if ctrl_cstr_onoff == 'on':
        accel_cmd = env.utils.clip_min_max(accel_cmd, ctrl_a_min, ctrl_a_max)
    elif ctrl_cstr_onoff == 'off':
        pass

    ####################
    # Vehicle dynamics #
    ####################
    if veh_dyn_type == 'car_none':
        # Calibrating

        # Not calibrating
        veh_a_max = parm_veh[0]
        veh_a_min = parm_veh[1]

        accel_next = veh_dyn.none_lon_dyn(
                    state_ego_veh[1],       # 1-Speed
                    accel_cmd
        )

    elif veh_dyn_type == 'car_linear':
        # Calibrating
        tau_a = parm_veh[0]
        # Not calibrating
        veh_a_max = parm_veh[1]
        veh_a_min = parm_veh[2]
        
        accel_next = veh_dyn.linear_lon_dyn(
                    state_ego_veh[1],       # 1-Speed
                    state_ego_veh[2],       # 2-Acceleration
                    accel_cmd,
                    dt, 
                    tau_a
        )

    elif veh_dyn_type == 'car_nonlinear':
        # Calibrating
        tau_a = parm_veh[0]
        veh_mass = parm_veh[1]
        f_0 = parm_veh[2]
        f_1 = parm_veh[3]
        f_2 = parm_veh[4]
        # Not calibrating
        phi = parm_veh[5]
        veh_a_max = parm_veh[6]
        veh_a_min = parm_veh[7]

        accel_next = veh_dyn.nonlinear_lon_dyn(
                    state_ego_veh[1],       # 1-Speed
                    state_ego_veh[2],       # 2-Acceleration
                    accel_cmd,
                    sin_theta,
                    dt,
                    phi,
                    tau_a,
                    veh_mass,
                    f_0,
                    f_1,
                    f_2,
        )

    #######################
    # Vehicle constraints #
    #######################
    if veh_cstr_type == 'mfc':
        mfc_a_max = env.utils.interp_binary(mfc_curve[0], mfc_curve[1], state_ego_veh[1])
        mfc_a_min = env.utils.interp_binary(mfc_curve[0], mfc_curve[2], state_ego_veh[1])

        accel_next = env.utils.clip_min_max(accel_next, mfc_a_min, mfc_a_max)

        if accel_next == mfc_a_max:
            count_cut_acc = 1
        elif accel_next == mfc_a_min:
            count_cut_dec = 1
    elif veh_cstr_type == 'constant':
        accel_next = env.utils.clip_min_max(accel_next, veh_a_min, veh_a_max)

        if accel_next == veh_a_max:
            count_cut_acc = 1
        elif accel_next == veh_a_min:
            count_cut_dec = 1
    elif veh_cstr_type == 'off':
        pass

    #####################################
    # Integration of speed and position #
    #####################################
    state_next = env.utils.motion_integ(
        state_ego_veh[0],
        state_ego_veh[1],
        accel_next,
        dt
    )
    return state_next, count_cut_acc, count_cut_dec, flag_scs
