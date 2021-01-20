import numpy as np
from numba import njit


@njit(nogil=True)
def constant_time_headway_spacing(speed_ego_veh, d_0, t_h, *args):

    spacing_des = d_0 + t_h * speed_ego_veh

    return spacing_des


@njit(nogil=True)
def idm_desired_spacing(speed_pre_veh, speed_ego_veh, d_0, t_h, a, b, *args):

    spacing_des = d_0 + max(t_h * speed_ego_veh + speed_ego_veh * (speed_ego_veh - speed_pre_veh) / (2 * np.sqrt(-a*b)), 0)

    return spacing_des


@njit(nogil=True)
def gipps_eq_spacing(speed_ego_veh, d_0, t_h, teta, accel_min_ego_veh, accel_min_pre_veh, *args):

    spacing_des = d_0 + (t_h + teta) * speed_ego_veh + 0.5 * speed_ego_veh**2 * (1/(-accel_min_ego_veh) - 1/(-accel_min_pre_veh))

    return spacing_des


@njit(nogil=True)
def constant_time_headway_speed(speed_ego_veh, spacing, d_0, t_h, v_set, *args):

    if spacing < d_0:
        speed_des = 0
    elif spacing >= d_0 and spacing <= d_0 + t_h*v_set:
        speed_des = (spacing - d_0) / t_h
    elif spacing > d_0 + t_h*v_set:
        speed_des = v_set

    return speed_des


@njit(nogil=True)
def fvdm_speed(speed_ego_veh, spacing, d_0, t_h, v_set, *args):

    if spacing < d_0:
        speed_des = 0
    elif spacing >= d_0 and spacing <= d_0 + t_h*v_set:
        speed_des = 0.5 * v_set * (1 - np.cos(np.pi * (spacing-d_0) / (t_h*v_set)))
    elif spacing > d_0 + t_h*v_set:
        speed_des = v_set

    return speed_des


@njit(nogil=True)
def gipps_speed(
    speed_ego_veh,
    spacing,
    d_0,
    t_h,
    v_set,
    teta,
    accel_min,
    accel_min_pre_veh_est,
    *args):

    if accel_min < accel_min_pre_veh_est:       # Both are negative
        symAxis = (t_h + teta) / (1/(-accel_min_pre_veh_est) - 1/(-accel_min))
        radicand = 1 - 2*(spacing - d_0)*(1/(-accel_min_pre_veh_est) - 1/(-accel_min)) / (t_h + teta)**2
        if v_set > symAxis or radicand < 0:
            speed_des = 0
            gipps_sus = 0
        else:
            speed_des = symAxis * (1 - np.sqrt(radicand))
            gipps_sus = 1
    elif accel_min > accel_min_pre_veh_est:     # Both are negative
        symAxis = (t_h + teta) / (1/(-accel_min_pre_veh_est) - 1/(-accel_min))
        radicand = 1 - 2*(spacing - d_0)*(1/(-accel_min_pre_veh_est) - 1/(-accel_min)) / (t_h + teta)**2
        if radicand < 0:
            speed_des = 0
            gipps_sus = 0
        else:
            speed_des = symAxis * (1 - np.sqrt(radicand))
            gipps_sus = 1
    elif accel_min == accel_min_pre_veh_est:    # Both are negative
        speed_des = (spacing - d_0) / (t_h + teta)
        gipps_sus = 1

    if speed_des < 0:
        speed_des = 0

    if speed_des > v_set:
        speed_des = v_set

    return speed_des, gipps_sus


# if __name__ == "__main__":
#     print(constant_time_headway(10, 5, 1.8))
#     print(idm_desired_spacing(12, 10, 5, 1.5, 4.2, -5))
#     print(gipps_eq_spacing(12, 5, 1.5, 1.5, -5.5, -6))