import numpy as np
from numba import njit


@njit(nogil=True)
def constant_time_headway(speed_ego_veh, d_0, t_h, *args):

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


# if __name__ == "__main__":
#     print(constant_time_headway(10, 5, 1.8))
#     print(idm_desired_spacing(12, 10, 5, 1.5, 4.2, -5))
#     print(gipps_eq_spacing(12, 5, 1.5, 1.5, -5.5, -6))