import functools as functools
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from c2art_env.veh_model.mfc.road_load_coefficients import compute_f_coefficients
import c2art_env.veh_model.mfc.defaults as defaults


def get_full_load(ignition_type):
    """
    Returns vehicle full load curve.

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :return:
        Vehicle normalized full load curve.
    :rtype: scipy.interpolate.InterpolatedUnivariateSpline
    """

    xp, fp = defaults.dfl.functions.get_full_load.FULL_LOAD[ignition_type]
    func = functools.partial(
        np.interp, xp=xp, fp=fp, left=fp[0], right=fp[-1]
    )
    return func


def calculate_full_load_speeds_and_powers(
        full_load_curve, engine_max_power, engine_max_speed_at_max_power,
        idle_engine_speed):
    """
    Calculates the full load speeds and powers [RPM, kW].

    :param full_load_curve:
        Vehicle normalized full load curve.
    :type full_load_curve: scipy.interpolate.InterpolatedUnivariateSpline

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :param engine_max_speed_at_max_power:
        Engine nominal speed at engine nominal power [RPM].
    :type engine_max_speed_at_max_power: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
         T1 map speed [RPM] and power [kW] vectors.
    :rtype: (numpy.array, numpy.array)
    """
    n_norm = np.arange(0.0, 1.21, 0.1)
    full_load_powers = full_load_curve(n_norm) * engine_max_power
    idle = idle_engine_speed[0]
    full_load_speeds = n_norm * (engine_max_speed_at_max_power - idle) + idle

    return full_load_speeds, full_load_powers


# Find where spline is zero
# def spline_find_zero(myspline1, myspline2, top_speed):
#     cutoff = top_speed
#     for i in np.arange(0, top_speed, 0.1):
#         acc = myspline1(i) - myspline2(i)
#         if acc < 0:
#             cutoff = i - 1
#             break
#     return cutoff


# The maximum force that the vehicle can have on the road
def Armax(total_mass, car_power, mh_base, road_type=1, car_type=1):
    if car_type == 1:  # forward-wheel drive vehicles
        fmass = 0.6 * total_mass
    elif car_type == 2:  # rear-wheel drive vehicles
        fmass = 0.45 * total_mass
    else:  # all-wheel drive vehicles, 4x4
        fmass = 1 * total_mass

    # Optimal values:
    # 0.8 dry, 0.6 light rain, 0.4 heavy rain, 0.1 icy
    # if road_type == 1:
    #     mh_base = 0.75  # for normal road
    # elif road_type == 2:
    #     mh_base = 0.25  # for wet road
    # else:
    #     mh_base = 0.1  # for icy road

    alpha = 43.398
    beta = 5.1549
    mh = mh_base * (alpha * np.log(car_power) + beta) / 190

    Frmax = fmass * 9.80665 * mh  # * cos(f) for the gradient of the road. Here we consider as 0

    return Frmax / total_mass


# Calculates a spline with the resistances
def veh_resistances(f0, f1, f2, sp, total_mass):
    # f0 + f1 * v + f2 * v2
    Fresistance = []
    for i in range(len(sp)):
        Fresistance.append(f0 + f1 * sp[i] * 3.6 + f2 * pow(sp[i] * 3.6, 2))
        # Facc = Fmax @ wheel - f0 * cos(a) - f1 * v - f2 * v2 - m * g * sin(a)

    aprx_mass = int(total_mass)
    Aresistance = [x / aprx_mass for x in Fresistance]
    a = int(np.floor(sp[0]))
    b = int(np.floor(sp[-1]))
    resistance_spline_curve = CubicSpline(
        [k for k in range(a - 10, a)] + sp + [k for k in range(b + 1, b + 11)],
        [Aresistance[0]] * 10 + Aresistance + [Aresistance[-1]] * 10
    )
    resistance_spline_curve_f = CubicSpline(
        [k for k in range(a - 10, a)] + sp + [k for k in range(b + 1, b + 11)],
        [Fresistance[0]] * 10 + Fresistance + [Fresistance[-1]] * 10
    )

    return resistance_spline_curve, resistance_spline_curve_f


# Acceleration curve for electric vehicles
def ev_curves(
        car,
        veh_mass,
        f0,
        f1,
        f2,
        mh_base):
    phi = car.phi
    motor_max_power = car.engine_max_power  # kW
    motor_max_torque = car.motor_max_torque  # Nm
    gr = car.gr
    engine_type = car.powertrain
    ignition_type = car.ignition_type
    tire_radius = car.tire_radius
    driveline_slippage = car.driveline_slippage
    driveline_efficiency = car.driveline_efficiency
    final_drive = car.final_drive
    top_speed = round(car.top_speed, 2)  # m/s
    car_type = car.car_type
    motor_base_speed = motor_max_power * 1000 * (motor_max_torque / 60 * 2 * np.pi) ** -1  # rpm
    motor_max_speed = top_speed * (60 * final_drive * gr[0]) / (1 - driveline_slippage) / (2 * np.pi * tire_radius)  # rpm
    veh_base_speed = 2 * np.pi * tire_radius * motor_base_speed * (1 - driveline_slippage) / (60 * final_drive * gr[0])  # m/s
    veh_max_acc = motor_max_torque * (final_drive * gr[0]) * driveline_efficiency / (tire_radius * veh_mass)  # m/s2
    veh_speed = list(np.arange(0, top_speed + 0.1, 0.1))  # m/s
    veh_acc = []
    for k in range(len(veh_speed)):
        if 0 <= veh_speed[k] <= veh_base_speed:
            veh_acc.append(veh_max_acc)
        elif veh_speed[k] > veh_base_speed:
            veh_acc.append(motor_max_power * 1000 * driveline_efficiency / (veh_speed[k] * veh_mass))
        else:
            print("You can't move backward!")
            exit()
    # plt.plot(veh_speed, veh_acc)
    a = np.round((veh_speed[0]), 2) - 0.01
    b = np.round((veh_speed[-1]), 2) + 0.01
    prefix_list = [a - k * 0.1 for k in range(10, -1, -1)]
    suffix_list = [b + k * 0.1 for k in range(0, 11, 1)]
    cs_acc_ev = CubicSpline(
        prefix_list + veh_speed + suffix_list,
        [veh_acc[0]] * len(prefix_list) + veh_acc + [veh_acc[-1]] * len(suffix_list)
    )
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2, list(veh_speed), veh_mass)
    Alimit = Armax(veh_mass, motor_max_power, mh_base, car_type=car_type)
    final_acc = cs_acc_ev(veh_speed) - car_res_curve(veh_speed)
    final_acc[final_acc > Alimit] = Alimit
    final_acc = final_acc / phi
    ### make 0 the acc outside gear possible use
    start = veh_speed[0]
    stop = veh_speed[-1]
    final_acc[(veh_speed < start)] = 0
    final_acc[(veh_speed > stop)] = 0
    final_acc[final_acc < 0] = 0
    Res = interp1d(veh_speed, final_acc)
    return Res, cs_acc_ev, (start, stop)


def gear_curves(
        car,
        veh_mass,
        f0,
        f1,
        f2,
        mh_base):
    phi = car.phi
    engine_max_power = int(car.engine_max_power)  # kW
    engine_max_speed_at_max_power = int(car.engine_max_speed_at_max_power)  # rpm
    gr = car.gr
    ignition_type = car.ignition_type
    tire_radius = car.tire_radius
    driveline_slippage = car.driveline_slippage
    driveline_efficiency = car.driveline_efficiency  # 0.90 0.93
    final_drive = car.final_drive
    top_speed = round(car.top_speed, 2) # m/s
    car_type = car.car_type
    idle_engine_speed = car.idle_engine_speed
    full_load = get_full_load(ignition_type)
    full_load_speeds, full_load_powers = calculate_full_load_speeds_and_powers(
        full_load,
        engine_max_power,
        engine_max_speed_at_max_power,
        idle_engine_speed
    )
    full_load_torque = full_load_powers * 1000 * (full_load_speeds / 60 * 2 * np.pi) ** -1
    speed_per_gear, acc_per_gear = [], []
    for j in range(len(gr)):
        speed_per_gear.append([])
        acc_per_gear.append([])
        for i in range(len(full_load_speeds)):
            # below 1.25 * idle the vehicle cannot be driven.
            if full_load_speeds[i] > 1.25 * idle_engine_speed[0]:
                speed_per_gear[j].append(
                    2 * np.pi * tire_radius * full_load_speeds[i] * (1 - driveline_slippage) / (60 * final_drive * gr[j])
                )
                acc_per_gear[j].append(
                    full_load_torque[i] * (final_drive * gr[j]) * driveline_efficiency / (tire_radius * veh_mass)
                )
    cs_acc_per_gear = []
    for j in range(len(gr)):
        # cs_acc_per_gear.append([])
        a = np.round((speed_per_gear[j][0]), 2) - 0.01
        b = np.round((speed_per_gear[j][-1]), 2) + 0.01
        prefix_list = [a - k * 0.1 for k in range(10, -1, -1)]
        suffix_list = [b + k * 0.1 for k in range(0, 11, 1)]
        cs_acc_per_gear.append(
            CubicSpline(
                prefix_list + speed_per_gear[j] + suffix_list,
                [acc_per_gear[j][0]] * len(prefix_list) + acc_per_gear[j] + [acc_per_gear[j][-1]] * len(suffix_list)
            )
        )
    Start = []
    Stop = []
    for i in speed_per_gear:
        Start.append(i[0])
        Stop.append(i[-1])
    sp_bins = np.arange(0,top_speed+0.1,0.1)
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2, list(sp_bins), veh_mass)
    Alimit = Armax(veh_mass, engine_max_power, mh_base, car_type=car_type)
    Res = []
    speed_per_gear[0][0] = 0
    for gear, acc in enumerate(cs_acc_per_gear):
        final_acc = acc(sp_bins)- car_res_curve(sp_bins)
        final_acc[final_acc > Alimit] = Alimit
        final_acc = final_acc / phi
        ### make 0 the acc outside gear possible use
        start = speed_per_gear[gear][0]
        stop = speed_per_gear[gear][-1]
        final_acc[(sp_bins < start)] = 0
        final_acc[(sp_bins > stop)] = 0
        final_acc[final_acc < 0] = 0
        Res.append(interp1d(sp_bins, final_acc))
    return Res, cs_acc_per_gear, (Start, Stop)
