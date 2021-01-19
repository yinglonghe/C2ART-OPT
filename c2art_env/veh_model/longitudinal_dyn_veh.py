from numba import njit


@njit(nogil=True)
def none_lon_dyn(
        speed_ego_veh, 
        accel_cmd, *args):

    accel_next = accel_cmd

    return accel_next


@njit(nogil=True)
def linear_lon_dyn(
        speed_ego_veh, 
        accel_ego_veh, 
        accel_cmd, 
        dt, 
        tau_a, *args):

    tau_a_dt = tau_a / dt

    accel_next = (tau_a_dt * accel_ego_veh + accel_cmd) / (1 + tau_a_dt)

    return accel_next


@njit(nogil=True)
def nonlinear_lon_dyn(
        speed_ego_veh, 
        accel_ego_veh, 
        accel_cmd, 
        sin_theta, 
        dt,
        phi, 
        tau_a,
        veh_mass,
        f_0,
        f_1,
        f_2, *args):

    tau_a_dt = tau_a / dt

    accel_tractive_current = accel_ego_veh * phi + (f_0 * (1 - sin_theta**2)**0.5 + f_1 * speed_ego_veh + f_2 * speed_ego_veh**2) / veh_mass + 9.80665 * sin_theta

    accel_tractive_next = (tau_a_dt * accel_tractive_current + accel_cmd) / (1 + tau_a_dt)

    accel_next = (accel_tractive_next - (f_0 * (1 - sin_theta**2)**0.5 + f_1 * speed_ego_veh + f_2 * speed_ego_veh**2) / veh_mass - 9.80665 * sin_theta) / phi

    return accel_next


