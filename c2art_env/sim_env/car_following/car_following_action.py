import numpy as np
from numba import njit
import c2art_env.ctrl_model.acceleration_ctrl as ctrl
import c2art_env.veh_model.longitudinal_dyn_veh as veh_dyn
import c2art_env as env


@njit(nogil=True)
def step(
        state_pre_veh_p,    # 0-Position, 1-Speed, 2-Acceleration
        state_ego_veh_p,  
        state_ego_veh,      # 0-Position, 1-Speed, 2-Acceleration
        state_spacing_p,  
        sin_theta,          # Road slope
        len_pre_veh,        # Length of pre veh
        flow_ctrl,
        parm_ctrl, 
        parm_veh,           # Model parameters
        mfc_curve,
        dt, *args):         # Time interval (s)

    count_cut_acc = 0
    count_cut_dec = 0
    gipps_sus = 1
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
    spacing_type = flow_ctrl[1]
    ctrl_cstr_onoff = flow_ctrl[2]           
    veh_dyn_type = flow_ctrl[3]
    veh_cstr_type = flow_ctrl[4] 

    ##################
    # ACC controller #
    ##################
    if control_type == 'acc_linear_d':
        if spacing_type == 'd_cth':
            # Calibrating
            k_p = parm_ctrl[0]
            k_v = parm_ctrl[1]
            k_a = parm_ctrl[2]
            k_set = parm_ctrl[3]
            v_set = parm_ctrl[4]
            d_0 = parm_ctrl[5]
            t_h = parm_ctrl[6]
            # Not calibrating
            ctrl_a_max = parm_ctrl[7]
            ctrl_a_min = parm_ctrl[8]

            # parm_args = np.empty(1, np.float64)
            parm_args = np.array([ctrl_a_max, ctrl_a_min])  
        elif spacing_type == 'd_idm_des':
            # Calibrating
            k_p = parm_ctrl[0]
            k_v = parm_ctrl[1]
            k_a = parm_ctrl[2]
            k_set = parm_ctrl[3]
            v_set = parm_ctrl[4]
            d_0 = parm_ctrl[5]
            t_h = parm_ctrl[6]
            ctrl_a_max = parm_ctrl[7]
            ctrl_a_min = parm_ctrl[8]
            # Not calibrating

            parm_args = np.array([ctrl_a_max, ctrl_a_min])            
        elif spacing_type == 'd_gipps_eq':
            # Calibrating
            k_p = parm_ctrl[0]
            k_v = parm_ctrl[1]
            k_a = parm_ctrl[2]
            k_set = parm_ctrl[3]
            v_set = parm_ctrl[4]
            d_0 = parm_ctrl[5]
            t_h = parm_ctrl[6]
            teta = parm_ctrl[7]
            ctrl_a_min = parm_ctrl[8]
            accel_min_pre_veh_est = parm_ctrl[9]
            # Not calibrating
            ctrl_a_max = parm_ctrl[10]

            parm_args = np.array([teta, ctrl_a_min, accel_min_pre_veh_est])

        accel_cmd = ctrl.linear_acc_d(
                    state_pre_veh_p,    # State of pre veh
                    state_ego_veh_p,    # State of ego veh
                    spacing_type,
                    len_pre_veh,        # Length of pre veh
                    k_p,
                    k_v,
                    k_a,
                    k_set,
                    v_set,
                    d_0,
                    t_h,
                    parm_args           # *args[0]
        )

    elif control_type == 'acc_linear_v':
        if spacing_type == 'v_cth':
            # Calibrating
            k_1 = parm_ctrl[0]
            k_2 = parm_ctrl[1]
            d_0 = parm_ctrl[2]
            t_h = parm_ctrl[3]
            v_set = parm_ctrl[4]
            # Not calibrating
            ctrl_a_max = parm_ctrl[5]
            ctrl_a_min = parm_ctrl[6]

            # parm_args = np.empty(1, np.float64)
            parm_args = np.array([ctrl_a_max, ctrl_a_min])  
        elif spacing_type == 'v_fvdm':
            # Calibrating
            k_1 = parm_ctrl[0]
            k_2 = parm_ctrl[1]
            d_0 = parm_ctrl[2]
            t_h = parm_ctrl[3]
            v_set = parm_ctrl[4]
            # Not calibrating
            ctrl_a_max = parm_ctrl[5]
            ctrl_a_min = parm_ctrl[6]

            parm_args = np.array([ctrl_a_max, ctrl_a_min])            
        elif spacing_type == 'v_gipps':
            # Calibrating
            k_1 = parm_ctrl[0]
            k_2 = parm_ctrl[1]
            d_0 = parm_ctrl[2]
            t_h = parm_ctrl[3]
            v_set = parm_ctrl[4]
            teta = parm_ctrl[5]
            ctrl_a_min = parm_ctrl[6]
            accel_min_pre_veh_est = parm_ctrl[7]
            # Not calibrating
            ctrl_a_max = parm_ctrl[8]

            parm_args = np.array([teta, ctrl_a_min, accel_min_pre_veh_est])

        accel_cmd = ctrl.linear_acc_v(
                    state_pre_veh_p,    # State of pre veh
                    state_ego_veh_p,    # State of ego veh
                    spacing_type,
                    len_pre_veh,        # Length of pre veh
                    k_1,
                    k_2,
                    v_set,
                    d_0,
                    t_h,
                    parm_args           # *args[0]
        )

    elif control_type == 'acc_idm':
        # Calibrating
        delta = parm_ctrl[0]
        v_set = parm_ctrl[1]
        d_0 = parm_ctrl[2]
        t_h = parm_ctrl[3]
        ctrl_a_max = parm_ctrl[4]
        ctrl_a_min = parm_ctrl[5]
        # Not calibrating

        accel_cmd = ctrl.idm_acc(
                    state_pre_veh_p,     # State of pre veh
                    state_ego_veh_p,     # State of ego veh
                    spacing_type,
                    len_pre_veh,                # Length of pre veh
                    delta,
                    v_set,
                    d_0,
                    t_h,
                    ctrl_a_max,
                    ctrl_a_min,
        )
    
    elif control_type == 'acc_gipps':
        # Calibrating
        teta = parm_ctrl[0]
        v_set = parm_ctrl[1]
        d_0 = parm_ctrl[2]
        t_h = parm_ctrl[3]
        ctrl_a_max = parm_ctrl[4]
        ctrl_a_min = parm_ctrl[5]
        accel_min_pre_veh_est = parm_ctrl[6]
        # Not calibrating

        accel_cmd, gipps_sus = ctrl.gipps_acc(
                    state_pre_veh_p, # State of pre veh
                    state_ego_veh_p, # State of ego veh
                    state_ego_veh[1],
                    # spacing_type,
                    len_pre_veh,                # Length of pre veh
                    teta,
                    v_set,
                    d_0,
                    t_h,
                    ctrl_a_max,
                    ctrl_a_min,
                    accel_min_pre_veh_est
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
    return state_next, count_cut_acc, count_cut_dec, gipps_sus