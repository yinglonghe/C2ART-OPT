import re, time
import sys, os, math
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from lmfit import Parameters, minimize
import c2art_env.utils as utils
import c2art_env.sim_env.hybrid_mfc.utils as sim_utils
from c2art_env.sim_env.hybrid_mfc.utils import PID
from c2art_env.sim_env.hybrid_mfc.road_load_coefficients import compute_f_coefficients
import c2art_env.sim_env.hybrid_mfc.mfc_acc as driver
from c2art_env.sim_env.hybrid_mfc.driver_charact import driver_charact

pid_aecms_ef = PID(Kp=125, Ki=0.16, Kd=0.004)


def energy_consumption_cd(
    car,
    a,
    v,
    gear, 
    soc_p,
    dt=0.1,
):
    a_r = (car.f0 + car.f1 * v * 3.6 + car.f2 * pow(v * 3.6, 2)) / car.total_mass
    a_t = a + a_r
    trq_req = a_t * car.total_mass * car.tire_radius / car.driveline_efficiency / (car.final_drive * car.gr[gear])      # Debug=50
    w_em = v / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gear]) # rad/s, Debug=315

    # Motor and Generator Configuration
    n_machine = 0.9     # Efficiency of electrical machine
    # T_em_max = car.motor_max_torque # 400 # Maximum Electric machine Torque [Nm]
    T_em_max = np.interp(w_em*60/(2*np.pi), car.em_full_load[0], car.em_full_load[1])
    # T_em_max = utils.interp_binary(car.em_full_load[0], car.em_full_load[1], w_ice*60/(2*np.pi))
    P_em_max = car.motor_max_power # 50 # Power of Electric Machine [kW]
    m_em = 1.5          # Electric motor weight [kg/Kw]

    # Battery Configuration
    Q_0 = car.battery_capacity*1000/car.battery_voltage # 6.5 # Battery charging capacity [Ah]
    U_oc = car.battery_voltage # 300 # Open circuit voltage [V]
    R_i = 0.65       # Inner resistance [ohm]
    # P_batt_max = P_em_max / n_machine # [kW]
    # I_batt_max = (U_oc - np.sqrt(max(0, (U_oc**2-4*R_i*P_batt_max*1000))))/(2*R_i)
    I_max = 200     # Maximum dis-/charging current [A]
    I_min = -200    # Maximum dis-/charging current [A]
    m_batt = 45      # [kg]
    
    T_em = max(trq_req, -T_em_max)
    Power_electric_machine = T_em*w_em
    Power_battery = T_em * w_em / (n_machine**np.sign(T_em))
    I = (U_oc - np.sqrt(max(0, (U_oc**2-4*R_i*Power_battery))))/(2*R_i)
    soc = -I*dt / (Q_0*3600) + soc_p
    soc = sim_utils._clamp(soc, (0.0, 1.0))
    P_ech = I*U_oc          # Electro-chemical Power for Battery

    return P_ech, soc, T_em, w_em, I

# @njit(nogil=True)
def energy_consumption_cs_aecms(
    car,
    a,
    v,
    v_p,
    gear, 
    soc_r_p,
    soc_p,
    dt=0.1,
    ):
    ef = pid_aecms_ef(soc_r_p, soc_p)       # Debug= 1.5 Equivalence Factor (constant or variable)
    # ef = 1.5
    a_r = (car.f0 + car.f1 * v * 3.6 + car.f2 * pow(v * 3.6, 2)) / car.total_mass
    a_t = a + a_r
    trq_req = a_t * car.total_mass * car.tire_radius / car.driveline_efficiency / (car.final_drive * car.gr[gear])      # Debug=50
    # w_ice_rpm = v / (2 * np.pi * car.tire_radius * (1 - car.driveline_slippage)) * (60 * car.final_drive * car.gr[gear]) # rpm
    # https://www.engineeringtoolbox.com/angular-velocity-acceleration-power-torque-d_1397.html
    # https://www.convertunits.com/from/RPM/to/rad/sec
    w_ice = v / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gear]) # rad/s, Debug=315
    w_ice_p = v_p / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gear]) # rad/s
    dw_ice = (w_ice - w_ice_p) / dt     # Debug=300
    
    # Fuel and Air Parameters
    H_l = 44.6e6     # Lower Heating Value [J/kg]
    roha_petrol = 737.2   # Fuel Density [Kg/m3]
    roha_air = 1.18   # Air Density [kg/m3] 

    # Engine Configuration
    J_e = 0.2   # Engine Inertia [kgm2]
    T_ice_max = car.engine_max_torque # 115     # Engine Maximum Torque [Nm]
    # T_ice_max = np.interp(w_ice*60/(2*np.pi), car.ice_full_load[0], car.ice_full_load[1])
    # T_ice_max = utils.interp_binary(car.ice_full_load[0], car.ice_full_load[1], w_ice*60/(2*np.pi))
    P_ice_max = car.engine_max_power    # Engine Maximum Power [kW]
    V_d = car.fuel_eng_capacity*1e-6    # 1.497e-3        # Engine Displacment [m3]
    n_r = 2
    m_e = 1.2   # [kg/KW]
    
    # Willans approximation of engine efficiency
    e = 0.4     # Willans approximation of engine efficiency
    p_me_0 = 0.1e6      # Willans approximation of engine pressure [MPa]

    # Motor and Generator Configuration
    n_machine = 0.9     # Efficiency of electrical machine
    T_em_max = car.motor_max_torque # 400 # Maximum Electric machine Torque [Nm]
    # T_em_max = np.interp(w_ice*60/(2*np.pi), car.em_full_load[0], car.em_full_load[1])
    # T_em_max = utils.interp_binary(car.em_full_load[0], car.em_full_load[1], w_ice*60/(2*np.pi))
    P_em_max = car.motor_max_power # 50 # Power of Electric Machine [kW]
    m_em = 1.5          # Electric motor weight [kg/Kw]

    # Battery Configuration
    Q_0 = car.battery_capacity*1000/car.battery_voltage # 6.5 # Battery charging capacity [Ah]
    U_oc = car.battery_voltage # 300 # Open circuit voltage [V]
    R_i = 0.65       # Inner resistance [ohm] 0.2 (dimitris)
    # P_batt_max = P_em_max / n_machine # [kW]
    # I_batt_max = (U_oc - np.sqrt(max(0, (U_oc**2-4*R_i*P_batt_max*1000))))/(2*R_i)
    I_max = 200     # Maximum dis-/charging current [A]
    I_min = -200    # Maximum dis-/charging current [A]
    m_batt = 45      # [kg]
    
    # Total Powertrain Power
    P_total_max = car.max_power # 90.8 # Total powertrain power[kW]
    
    # Logitudinal Vehicle Model Parameters
    g = 9.81    # Gravity [m/s2]
    c_D = 0.32  # Drag coefficient [-]
    c_R = 0.015 # Rolling resistance coefficient [-]
    A_f = 2.31  # Frontal area [m2]
    m = car.total_mass # 1500 # Vehicle mass [kg]
    r_w = car.tire_radius # 0.3   # Wheel radius [m]
    J_w = 0.6   # Inertia of the wheels [kgm2]
    m_wheel = J_w/(r_w**2)   # Mass of Wheel [kg]
    n_gearbox = 0.98        # Efficiency of Transmission

    # Range
    N_e = np.arange(0, 5000, 800)    # Engine RPM range
    w_e = 300   # [rad/s2] % Maximum accelration of the engine

    # -----------------------Cost Vector Calculations--------------------------
    # Battery Model
    
    # I = np.linspace(I_min, I_max, 50000)
    # Power_battery = (U_oc*I) - (R_i*(I**2))   # Power of Battery

    # Electric Machine Model
    # Power_electric_machine = Power_battery*(n_machine**np.sign(I))  # Power of Electric Machine

    # T_em = (Power_battery*(n_machine**np.sign(I))) / w_ice  # Torque of Electric Machine
    # T_em[T_em > T_em_max] = T_em_max
    # T_em[T_em < -T_em_max] = -T_em_max
    
    T_em = np.linspace(-T_em_max, T_em_max, 50000)
    Power_electric_machine = T_em*w_ice
    Power_battery = T_em * w_ice / (n_machine**np.sign(T_em))
    tmp_ = U_oc**2-4*R_i*Power_battery
    tmp_[tmp_<0]=0
    I = (U_oc - np.sqrt(tmp_))/(2*R_i)
    # Engine Model
    # trq_req = max(trq_req, -T_em_max)
    T_ice = trq_req - T_em  # Torque of Engine
    # T_ice[T_ice<0] = 0

    x = ((p_me_0*V_d)/(4*np.pi))     # First Part to calculate Fuel consumption

    y = J_e*dw_ice        # Second Part to calculate Fuel consumption

    costVector = (w_ice/(e*H_l))*(T_ice+x+y)  # Fuel Consumption Cost Vector for given t_vec

    # Hamiltonian Calculation [Torque Split]

    P_f = H_l*costVector    # Fuel Power
    P_f[costVector<0] = 0     # Constraints
    P_ech = I*U_oc          # Electro-chemical Power for Battery

    # If condition for torque split
    if w_ice<=0.1:
        T_ice_cmd = 0
        T_em_cmd = 0
        u = [T_ice_cmd, T_em_cmd]
        Batt_I=0
        soc = soc_p
        p_ice = 0
        p_em = 0
        p_batt = 0
        p_fuel = 0
        fc = 0
    else:
        H = P_f + (ef * P_ech)
        H[(T_ice > T_ice_max)] = np.inf
        H[(I > I_max) | (I < I_min)] = np.inf
        # H[(T_em > T_em_max) | (T_em < -T_em_max)] = np.inf
        H[(Power_electric_machine < -P_em_max*1000) | (Power_electric_machine > P_em_max*1000)] = np.inf
        i = np.argmin(H)
        T_ice_cmd = max(0, T_ice[i])
        T_em_cmd = T_em[i]
        u = [T_ice_cmd, T_em_cmd]
        # New battery state of charge
        Batt_I = I[i]
        soc = -Batt_I*dt / (Q_0*3600) + soc_p
        soc = sim_utils._clamp(soc, (0.0, 1.0))
        # Power and fuel economy
        p_ice = T_ice_cmd*w_ice
        p_em = T_em_cmd*w_ice
        p_batt = P_ech[i]   # Energy consumption of battery [W]
        p_fuel = P_f[i]     # Energy consumption of ice [W]
        fc = p_fuel*dt/H_l/roha_petrol*1000     # Fuel consumption [L]

    return fc, p_fuel, p_batt, p_ice, p_em, T_ice_cmd, T_em_cmd, soc, w_ice, trq_req, ef, Batt_I


def powertrain_hyd_cs_aecms(
    car,
    soc_p,
    soc_ref_p,
    w_gr_in,
    trq_gr_in,
    dw_ice,
    dt=0.1,
    ):
    ef = pid_aecms_ef(soc_ref_p, soc_p)       # Debug= 1.5 Equivalence Factor (constant or variable)
        
    # Fuel and Air Parameters
    H_l = 44.6e6     # Lower Heating Value [J/kg]
    roha_petrol = 737.2   # Fuel Density [Kg/m3]
    roha_air = 1.18   # Air Density [kg/m3] 

    # Engine Configuration
    J_e = 0.2   # Engine Inertia [kgm2]
    T_ice_max = car.engine_max_torque # 115     # Engine Maximum Torque [Nm]
    # T_ice_max = np.interp(w_ice*60/(2*np.pi), car.ice_full_load[0], car.ice_full_load[1])
    # T_ice_max = utils.interp_binary(car.ice_full_load[0], car.ice_full_load[1], w_ice*60/(2*np.pi))
    P_ice_max = car.engine_max_power    # Engine Maximum Power [kW]
    V_d = car.fuel_eng_capacity*1e-6    # 1.497e-3        # Engine Displacment [m3]
    n_r = 2
    m_e = 1.2   # [kg/KW]
    
    # Willans approximation of engine efficiency
    e = 0.4     # Willans approximation of engine efficiency
    p_me_0 = 0.1e6      # Willans approximation of engine pressure [MPa]

    # Motor and Generator Configuration
    n_machine = 0.9     # Efficiency of electrical machine
    T_em_max = car.motor_max_torque # 400 # Maximum Electric machine Torque [Nm]
    # T_em_max = np.interp(w_ice*60/(2*np.pi), car.em_full_load[0], car.em_full_load[1])
    # T_em_max = utils.interp_binary(car.em_full_load[0], car.em_full_load[1], w_ice*60/(2*np.pi))
    P_em_max = car.motor_max_power # 50 # Power of Electric Machine [kW]
    m_em = 1.5          # Electric motor weight [kg/Kw]

    # Battery Configuration
    Q_0 = car.battery_capacity*1000/car.battery_voltage # 6.5 # Battery charging capacity [Ah]
    U_oc = car.battery_voltage # 300 # Open circuit voltage [V]
    R_i = 0.65       # Inner resistance [ohm] 0.2 (dimitris)
    # P_batt_max = P_em_max / n_machine # [kW]
    # I_batt_max = (U_oc - np.sqrt(max(0, (U_oc**2-4*R_i*P_batt_max*1000))))/(2*R_i)
    I_max = 200     # Maximum dis-/charging current [A]
    I_min = -200    # Maximum dis-/charging current [A]
    m_batt = 45      # [kg]
    
    # Total Powertrain Power
    P_total_max = car.max_power # 90.8 # Total powertrain power[kW]
    
    # Logitudinal Vehicle Model Parameters
    g = 9.81    # Gravity [m/s2]
    c_D = 0.32  # Drag coefficient [-]
    c_R = 0.015 # Rolling resistance coefficient [-]
    A_f = 2.31  # Frontal area [m2]
    m = car.total_mass # 1500 # Vehicle mass [kg]
    r_w = car.tire_radius # 0.3   # Wheel radius [m]
    J_w = 0.6   # Inertia of the wheels [kgm2]
    m_wheel = J_w/(r_w**2)   # Mass of Wheel [kg]
    n_gearbox = 0.98        # Efficiency of Transmission

    # Range
    N_e = np.arange(0, 5000, 800)    # Engine RPM range
    w_e = 300   # [rad/s2] % Maximum accelration of the engine

    # -----------------------Cost Vector Calculations--------------------------
    # Battery Model
    
    # I = np.linspace(I_min, I_max, 50000)
    # Power_battery = (U_oc*I) - (R_i*(I**2))   # Power of Battery

    # Electric Machine Model
    # Power_electric_machine = Power_battery*(n_machine**np.sign(I))  # Power of Electric Machine

    # T_em = (Power_battery*(n_machine**np.sign(I))) / w_ice  # Torque of Electric Machine
    # T_em[T_em > T_em_max] = T_em_max
    # T_em[T_em < -T_em_max] = -T_em_max
    
    T_em = np.linspace(-T_em_max, T_em_max, 50000)
    Power_electric_machine = T_em*w_gr_in
    Power_battery = T_em * w_gr_in / (n_machine**np.sign(T_em))
    tmp_ = U_oc**2-4*R_i*Power_battery
    tmp_[tmp_<0]=0
    I = (U_oc - np.sqrt(tmp_))/(2*R_i)
    # Engine Model
    # trq_req = max(trq_req, -T_em_max)
    T_ice = trq_gr_in - T_em  # Torque of Engine
    # T_ice[T_ice<0] = 0

    x = ((p_me_0*V_d)/(4*np.pi))     # First Part to calculate Fuel consumption

    y = J_e*dw_ice        # Second Part to calculate Fuel consumption

    costVector = (w_gr_in/(e*H_l))*(T_ice+x+y)  # Fuel Consumption Cost Vector for given t_vec

    # Hamiltonian Calculation [Torque Split]

    P_f = H_l*costVector    # Fuel Power
    P_f[costVector<0] = 0     # Constraints
    P_ech = I*U_oc          # Electro-chemical Power for Battery

    # If condition for torque split
    if w_gr_in<=0.1:
        T_ice_cmd = 0
        T_em_cmd = 0
        u = [T_ice_cmd, T_em_cmd]
        Batt_I=0
        soc = soc_p
        p_ice = 0
        p_em = 0
        p_batt = 0
        p_fuel = 0
        fc = 0
    else:
        H = P_f + (ef * P_ech)
        H[(T_ice > T_ice_max)] = np.inf
        H[(I > I_max) | (I < I_min)] = np.inf
        # H[(T_em > T_em_max) | (T_em < -T_em_max)] = np.inf
        H[(Power_electric_machine < -P_em_max*1000) | (Power_electric_machine > P_em_max*1000)] = np.inf
        i = np.argmin(H)
        T_ice_cmd = max(0, T_ice[i])
        T_em_cmd = T_em[i]
        u = [T_ice_cmd, T_em_cmd]
        # New battery state of charge
        Batt_I = I[i]
        soc = -Batt_I*dt / (Q_0*3600) + soc_p
        soc = sim_utils._clamp(soc, (0.0, 1.0))
        # Power and fuel economy
        p_ice = T_ice_cmd*w_gr_in
        p_em = T_em_cmd*w_gr_in
        p_batt = P_ech[i]   # Energy consumption of battery [W]
        p_fuel = P_f[i]     # Energy consumption of ice [W]
        fc = p_fuel*dt/H_l/roha_petrol*1000     # Fuel consumption [L]

    return soc, fc, T_ice_cmd, p_ice, p_fuel, T_em_cmd, p_em, p_batt, Batt_I, ef 


def powertrain_hyd_cd(
    car,
    soc_p,
    w_gr_in,
    trq_gr_in,
    dt=0.1,
    ):
    # Motor and Generator Configuration
    n_machine = 0.9     # Efficiency of electrical machine
    # T_em_max = car.motor_max_torque # 400 # Maximum Electric machine Torque [Nm]
    T_em_max = np.interp(w_gr_in*60/(2*np.pi), car.em_full_load[0], car.em_full_load[1])
    # T_em_max = utils.interp_binary(car.em_full_load[0], car.em_full_load[1], w_ice*60/(2*np.pi))
    P_em_max = car.motor_max_power # 50 # Power of Electric Machine [kW]
    m_em = 1.5          # Electric motor weight [kg/Kw]

    # Battery Configuration
    Q_0 = car.battery_capacity*1000/car.battery_voltage # 6.5 # Battery charging capacity [Ah]
    U_oc = car.battery_voltage # 300 # Open circuit voltage [V]
    R_i = 0.65       # Inner resistance [ohm]
    # P_batt_max = P_em_max / n_machine # [kW]
    # I_batt_max = (U_oc - np.sqrt(max(0, (U_oc**2-4*R_i*P_batt_max*1000))))/(2*R_i)
    I_max = 200     # Maximum dis-/charging current [A]
    I_min = -200    # Maximum dis-/charging current [A]
    m_batt = 45      # [kg]
    
    T_em = max(trq_gr_in, -T_em_max)
    p_em = T_em*w_gr_in
    Power_battery = T_em * w_gr_in / (n_machine**np.sign(T_em))
    I = (U_oc - np.sqrt(max(0, (U_oc**2-4*R_i*Power_battery))))/(2*R_i)
    soc = -I*dt / (Q_0*3600) + soc_p
    soc = sim_utils._clamp(soc, (0.0, 1.0))
    P_ech = I*U_oc          # Electro-chemical Power for Battery

    return soc, T_em, p_em, P_ech, I


def variable_speed_sim(
    car,
    hyd_mode,
    driver_style, 
    gs_th,
    ap_curve, 
    dp_curve, 
    will_acc_model, 
    veh_max_speed, 
    overshoot
    ):

    dt = 0.1
    t_f = 270
    t = np.arange(0, t_f + dt, dt)
    X, V, A, V_n = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    V_n[0] = 20

    FC, Gear, SOC, SOC_R = np.zeros(len(t)), np.zeros(len(t)).astype(int), np.zeros(len(t)), np.ones(len(t))*0.5
    P_fuel, P_batt, P_ice, P_em, T_ice, T_em, w_e, trq_req, ef, I = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    
    Gear[0] = int(gs_th(V[0])) 
    SOC[0] = 0.8 if hyd_mode=='CD' else 0.5
    gs_cnt = 10

    dd = 700
    for i in tqdm(range(1, len(t))):
        if X[i-1] <= 0.3 * dd:
            V_n[i] = V_n[0]
        elif X[i-1] <= 1 * dd:
            V_n[i] = 30
        elif X[i-1] <= 1.5 * dd:
            V_n[i] = 5
        elif X[i-1] <= 2 * dd:
            V_n[i] = 25
        elif X[i-1] <= 4.5 * dd:
            V_n[i] = veh_max_speed
        elif X[i-1] <= 6.5 * dd:
            V_n[i] = 15
        else:
            V_n[i] = 0

        gs_cnt += 1
        if gs_cnt < 10:
            Gear[i] = Gear[i-1]
        else:
            if int(gs_th(V[i-1]))>Gear[i-1]:
                Gear[i] = Gear[i-1]+1
                gs_cnt = 0
            elif int(gs_th(V[i-1]))<Gear[i-1]:
                Gear[i] = Gear[i-1]-1
                gs_cnt = 0
            else:
                Gear[i] = Gear[i-1]

        A[i] = accMFC(V[i-1], V_n[i], driver_style, ap_curve, dp_curve, will_acc_model, overshoot)
        if gs_cnt<3 and A[i]>0:
            A[i] = 0
        V[i] = max(V[i-1] + A[i] * dt, 0)
        X[i] = X[i-1] + (V[i] + V[i-1]) / 2 * dt

        if hyd_mode=="CS":
            FC[i], P_fuel[i], P_batt[i], P_ice[i], P_em[i], \
                T_ice[i], T_em[i], SOC[i], w_e[i], trq_req[i], ef[i], I[i] \
                = energy_consumption_cs_aecms(
                    car, A[i], V[i], V[i-1], Gear[i], \
                    SOC_R[i-1], SOC[i-1])
        elif hyd_mode=="CD":
            P_batt[i], SOC[i], T_em[i], w_e[i], I[i] = energy_consumption_cd(car, A[i], V[i], Gear[i], SOC[i-1])

    A[0] = A[1]

    data_plot = [
        V, A, Gear, w_e*60/(2*np.pi), \
        P_batt*1e-3, SOC, T_em, I, \
        FC*1e3, P_ice*1e-3, T_ice,
    ]
    data_label = [
        'v(m/s)', 'a(m/s2)', 'gear', 'w_b4_gear\n(rpm)', \
        'P_batt(kW)', 'soc', 'T_em(Nm)', 'I(A)', \
        'fc(ml)', 'P_ice(kW)', 'T_ice(Nm)',
    ]

    fig, axs = plt.subplots(len(data_plot), sharex=True)
    fig.suptitle('MFC hybrid energy consumption - '+hyd_mode)

    for i in range(len(data_plot)):
        axs[i].plot(t, data_plot[i])
        if data_label[i]=='v(m/s)':
            axs[i].plot(t, V_n, 'r--')
        else:
            axs[i].plot(t, np.zeros_like(t), 'g--')
        axs[i].set_ylabel(data_label[i], rotation=0, labelpad=30)
        if data_label[i]=='fc(ml)':
            axs[i].text(200, 0.2, 'avg. fc (l/100km): '+str(round(np.sum(FC)/(X[-1]*1e-5), 1)))
    axs[i].set_xlabel('time(s)')
    axs[i].set_xlim(0, t[-1])
    plt.show()

    return t, V_n, X, V, A


def variable_speed_sim_dc(
    driver_style, 
    ap_curve, 
    dp_curve, 
    will_acc_model, 
    veh_max_speed, 
    overshoot
    ):

    dt = 0.1
    t_f = 270
    t = np.arange(0, t_f + dt, dt)
    X, V, A, V_n = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    V_n[0] = 20

    dd = 700
    dc = driver_charact(driver_style)
    for i in range(1, len(t)):
        if X[i-1] <= 0.3 * dd:
            V_n[i] = V_n[0]
        elif X[i-1] <= 1 * dd:
            V_n[i] = 30
        elif X[i-1] <= 1.5 * dd:
            V_n[i] = 5
        elif X[i-1] <= 2 * dd:
            V_n[i] = 25
        elif X[i-1] <= 4.5 * dd:
            V_n[i] = veh_max_speed
        elif X[i-1] <= 6.5 * dd:
            V_n[i] = 15
        else:
            V_n[i] = 0

        if i == 1 or i % 10 == 0:
            ids = dc.gen_ids()
            ds = dc.get_ds(V[i-1], V_n[i])
        
        A[i] = accMFC(V[i-1], V_n[i], ds, ap_curve, dp_curve, will_acc_model, overshoot)

        V[i] = max(V[i-1] + A[i] * dt, 0)

        X[i] = X[i-1] + (V[i] + V[i-1]) / 2 * dt

    A[0] = A[1]

    return t, V_n, X, V, A


def accel_sim(
    car,
    driver_style,
    gs_th, 
    ap_curve, 
    dp_curve, 
    will_acc_model, 
    veh_max_speed, 
    overshoot,
    t_f,
    acc_or_dec
    ):

    dt = 0.1
    t = np.arange(0, t_f + dt, dt)
    X, V, A = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))

    if acc_or_dec == 'acc':
        V_n = np.ones(len(t)) * veh_max_speed
        V[0] = 0
    elif acc_or_dec == 'dec':
        V_n = np.zeros(len(t))
        V[0] = veh_max_speed

    Gear = [int(gs_th(V[0]))]
    gs_cnt = 0

    for i in range(1, len(t)):

        A[i] = accMFC(V[i-1], V_n[i], driver_style, ap_curve, dp_curve, will_acc_model, overshoot)
        V[i] = max(V[i-1] + A[i] * dt, 0)
        X[i] = X[i-1] + (V[i] + V[i-1]) / 2 * dt
    
        gs_cnt += 1
        if gs_cnt < 10:
            Gear.append(Gear[-1])
        else:
            if gs_th(V[i])>Gear[-1]:
                Gear.append(Gear[-1]+1)
                gs_cnt = 0
            if gs_th(V[i])<Gear[-1]:
                Gear.append(Gear[-1]-1)
                gs_cnt = 0
            else:
                Gear.append(Gear[-1])

    A[0] = A[1]

    return t, V_n, X, V, A


def accel_sim_dc(
    driver_style, 
    ap_curve, 
    dp_curve, 
    will_acc_model, 
    veh_max_speed, 
    overshoot,
    t_f,
    acc_or_dec
    ):

    dt = 0.1
    t = np.arange(0, t_f + dt, dt)
    X, V, A = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))

    if acc_or_dec == 'acc':
        V_n = np.ones(len(t)) * veh_max_speed
        V[0] = 0
    elif acc_or_dec == 'dec':
        V_n = np.zeros(len(t))
        V[0] = veh_max_speed

    dc = driver_charact(driver_style)
    for i in range(1, len(t)):
        if i == 1 or i % 10 == 0:
            ids = dc.gen_ids()
            ds = dc.get_ds(V[i-1], V_n[i])

        A[i] = accMFC(V[i-1], V_n[i], ds, ap_curve, dp_curve, will_acc_model, overshoot)

        V[i] = max(V[i-1] + A[i] * dt, 0)

        X[i] = X[i-1] + (V[i] + V[i-1]) / 2 * dt

    A[0] = A[1]

    return t, V_n, X, V, A


def accMFC(
    v, 
    vdes, 
    driver_style, 
    ap_curve, 
    dp_curve, 
    will_acc_model='horizontal_b', 
    overshoot=0,
    ):

    if will_acc_model == 'gipps':
        acc = ap_curve(v) * 2.5 * driver_style * (1 - v / (vdes + overshoot)) * np.sqrt(0.025 + v / (vdes + overshoot))
    elif will_acc_model == 'idm':
        acc = ap_curve(v) * driver_style * (1 - pow(v / (vdes + overshoot), 4))
    elif will_acc_model == 'horizontal_a':
        if v >= vdes:
            onoff = (1 - pow(1 - (v - vdes - overshoot) / 50, 100))
            acc = dp_curve(v) * driver_style * onoff
        elif v >= 0.5 * vdes:
            onoff = (1 - pow(1 + (v - vdes - overshoot) / 50, 100))
            acc = ap_curve(v) * driver_style * onoff
        else:
            onoff = (1 - 0.8 * pow(1 - v / (vdes + overshoot), 60))
            acc = ap_curve(v) * driver_style * onoff
    elif will_acc_model == 'horizontal_b':
        onoff = max(1 - pow(1 + 2 * (v - vdes) / (vdes + 0.1), 30), \
            1 - pow(1 - (v - vdes) / 50, 100))
        if v <= vdes:
            acc = ap_curve(v) * driver_style * onoff
        else:
            acc = dp_curve(v) * driver_style * onoff
    else:
        print('wrong willing to accelerate model selection')
        acc = -1

    return acc


def mfc_curves(
    car, 
    car_id, 
    hyd_mode, 
    veh_load,
    gs_style,
    ):

    veh_mass = car.veh_mass + veh_load

    f0, f1, f2 = compute_f_coefficients(
        car.powertrain,
        car.type_of_car,
        car.car_width,
        car.car_height,
        veh_mass
    )
    car.total_mass = veh_mass
    car.f0, car.f1, car.f2 = f0, f1, f2

    veh_max_speed = int(car.top_speed)

    if car.powertrain in ['fuel engine']:
        if car_id == 26966:
            ppar0, ppar1, ppar2 = 0.0045, -0.1710, -1.8835
        elif car_id == 7565:
            ppar0, ppar1, ppar2 = 0.0045, -0.1710, -1.8835

        curves = driver.gear_curves(
            car,
            veh_mass,
            f0,
            f1,
            f2
        )

        car_info = 'ICEV: ' + car.model

    elif car.powertrain in ['electric engine']:
        if car_id == 26687:
            ppar0, ppar1, ppar2 = 0.0045, -0.1710, -1.8835

        curves = driver.ev_curves(
            car,
            veh_mass,
            f0,
            f1,
            f2
        )

        car_info = 'EV: ' + car.model

    elif car.powertrain in ['hybrid', 'plug-in hybrid']:
        if car_id == 26712:
            car.motor_max_power = 44.5
            car.final_drive = 4.438
            car.driveline_efficiency = 0.93
            ppar0, ppar1, ppar2 = 0.0058, -0.2700, -1.8835
        else:
            ppar0, ppar1, ppar2 = 0.0058, -0.2700, -1.8835

        # if hyd_mode == 'CD':
        #     curves = driver.ev_curves(
        #         car,
        #         veh_mass,
        #         f0,
        #         f1,
        #         f2
        #     )

        #     car_info = 'HEV (CD): ' + car.model

        # elif hyd_mode == 'CS':
        curves = driver.hybrid_curves(
            car,
            veh_mass,
            f0,
            f1,
            f2,
            hyd_mode
        )

        car_info = 'HEV_'+hyd_mode+': '+ car.model
            

    veh_model_speed = list(np.arange(0, veh_max_speed + 0.1, 0.1))  # m/s
    veh_model_acc = []      # Considering gearshift
    veh_model_acc_max = []  # Not considering gearshift, use the maximum across different gears
    veh_model_dec = []
    ppar = [ppar0, ppar1, ppar2]
    dec_curve = np.poly1d(ppar)

    (start, stop) = curves[2]
    gr_ice_lim = np.array([start, stop]).T.tolist()
    # gr_overlap = [max(gr_ice_lim[i-1][1]-max(gr_ice_lim[i][0], gr_ice_lim[i-1][0]),0) for i in np.arange(1,len(gr_ice_lim))]
    # gs_th = [gr_ice_lim[i][1]-(1-gs_style)*gr_overlap[i] for i in range(len(gr_ice_lim)-1)]
    gs_up_th = [gr_ice_lim[i][0]+gs_style*(gr_ice_lim[i][1]-gr_ice_lim[i][0]) for i in range(len(gr_ice_lim)-1)]
    gs_down_th = [gr_ice_lim[i][1]-gs_style*(gr_ice_lim[i][1]-gr_ice_lim[i][0]) for i in np.arange(1, len(gr_ice_lim))]
    gs_up, gs_down = np.zeros(len(veh_model_speed)).astype(int), np.zeros(len(veh_model_speed)).astype(int)
    for i in range(len(veh_model_speed)):
        for j in range(len(gs_up_th)):
            if veh_model_speed[i]>gs_up_th[j]:
                gs_up[i] += 1
            if veh_model_speed[i]>gs_down_th[j]:
                gs_down[i] += 1  
    gs_th = interp1d(veh_model_speed, gs_up, kind='nearest', fill_value="extrapolate")

    for k in range(len(veh_model_speed)):
        acc_temp = []
        for i in range(len(curves[0])):
            acc_temp.append(float(curves[0][i](veh_model_speed[k])))
        veh_model_acc_max.append(max(acc_temp))
        veh_model_acc.append(float(curves[0][int(gs_th(veh_model_speed[k]))](veh_model_speed[k])))
        veh_model_dec.append(min(dec_curve(veh_model_speed[k]), -1))
    acc_curve = interpolate.CubicSpline(
            veh_model_speed, veh_model_acc)
    dec_curve = interpolate.CubicSpline(
            veh_model_speed, veh_model_dec)

    return car_info, acc_curve, dec_curve, veh_model_speed, veh_model_acc, veh_model_dec, veh_model_acc_max, veh_max_speed, gs_th


def plt_exp_val_spd_accel(
    pathRoot, 
    car_id, 
    car_info, 
    hyd_mode, 
    veh_model_speed, 
    veh_model_acc, 
    veh_model_dec, 
    veh_max_speed,
    res_path
    ):

    if car_id == 26712:
        data_name = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                            'datasets', 'kia_niro')
        if hyd_mode == 'CD':
            df = pd.read_csv(os.path.join(data_name, '20190424_02_WarmUpCD_ElasticityCD_SailingCD_ConstSpeedCS_aligned.csv'))
            df['spd_mps'] = df['Current speed [km/h]'] / 3.6
            df['accel_mps2'] = np.append((utils.numbadiff(df['spd_mps'].to_numpy()) / utils.numbadiff(df['Time [s]'].to_numpy())), 0)
            df_exp = df.loc[df['EngineRPM [rpm]'] == 0]
        elif hyd_mode == 'CS':
            df = pd.read_csv(os.path.join(data_name, '20190424_03_ElasticityCS_SailingCS_aligned.csv'))
            df['spd_mps'] = df['Current speed [km/h]'] / 3.6
            df['accel_mps2'] = np.append((utils.numbadiff(df['spd_mps'].to_numpy()) / utils.numbadiff(df['Time [s]'].to_numpy())), 0)
            df_exp = df.loc[df['EngineRPM [rpm]'] != 0]
        plt.scatter(df_exp['spd_mps'], df_exp['accel_mps2'], s=6, label='Experimental points')
        plt.xlim(0, df_exp['spd_mps'].max()+1)

    if car_id == 26687:
        data_name = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                    'datasets', 'hyundai_ioniq')
        df = pd.read_csv(os.path.join(data_name, 'Hyundai Ioniq Electric_final.csv'))
        df['spd_mps'] = df['v Dyno [km/h]'] / 3.6
        df['accel_mps2'] = df['a Dyno [m/s2]']
        df_exp = df
        plt.scatter(df_exp['spd_mps'], df_exp['accel_mps2'], s=6, label='Experimental points')
        plt.xlim(0, df_exp['spd_mps'].max()+1)

    if car_id == 26966:
        data_name = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                    'datasets', 'jrc_vicolungo_dataset')
        file_list = [
            'JRC_Vicolungo_26_2_2019.csv',
            'JRC_Vicolungo_28_2_2019.csv',
            'Vicolungo_JRC_26_2_2019.csv',
            'Vicolungo_JRC_28_2_2019.csv',
        ]

        v, a = [], []
        for i in range(len(file_list)):
            df = pd.read_csv(os.path.join(data_name, file_list[i]))
            # df['spd_mps_0'] = df['v1s']
            # df['accel_mps2_0'] = df['v1a']
            df['t'] = df['Time'] - df['Time'][0]
            df['spd_mps'] = signal.savgol_filter(df['v1s'], 201, 3)
            df['spd_mps'] = df['spd_mps'].apply(lambda x : x if x > 0 else 0)
            df['accel_mps2'] = np.append((utils.numbadiff(df['spd_mps'].to_numpy()) / utils.numbadiff(df['t'].to_numpy())), 0)
            
            df = df.loc[(df['accel_mps2'] <= 5) & (df['accel_mps2'] >= -5) & (df['spd_mps'] < veh_max_speed)]
            v = v + df['spd_mps'].tolist()
            a = a + df['accel_mps2'].tolist()

        plt.scatter(v, a, s=6, label='Experimental points')
        plt.xlim(0, max(v)+1)

    plt.plot(veh_model_speed, veh_model_acc, 'k-', label='Accel potential ($a_{ap}$)')
    plt.plot(veh_model_speed, veh_model_dec, 'b-', label='Decel potential ($a_{dp}$)')
    plt.xlabel(r'Veh. spd ($m/s$)')
    plt.ylabel(r'Veh. accel ($m/s^2$)')
    plt.title(car_info)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, 'ap_dp_curves_exp_val.png'))
    # plt.show()
    plt.close()


def plt_accel_scenario(
    driver_style,
    gs_th,
    acc_curve,
    dec_curve,
    veh_max_speed,
    car_info,
    car,
    hyd_mode,
    res_path
    ):
    """
    t, V_n, X, V, A = [], [], [], [], []
    for k in range(len(driver_style)):
        t_, V_n_, X_, V_, A_ = accel_sim(car, driver_style[k], gs_th, acc_curve, dec_curve, 'horizontal_b', veh_max_speed, 0, 200, 'acc')
        t.append(t_)
        V_n.append(V_n_) 
        X.append(X_)
        V.append(V_)
        A.append(A_)
    temp_path = os.path.join(res_path, 'accel_sim')
    plot_traj(driver_style, t, V_n, X, V, A, temp_path)

    t, V_n, X, V, A = [], [], [], [], []
    for k in range(len(driver_style)):
        t_, V_n_, X_, V_, A_ = accel_sim(car, driver_style[k], gs_th, acc_curve, dec_curve, 'horizontal_b', veh_max_speed, 0, 100, 'dec')
        t.append(t_)
        V_n.append(V_n_)
        X.append(X_)
        V.append(V_)
        A.append(A_)

    temp_path = os.path.join(res_path, 'decel_sim')
    plot_traj(driver_style, t, V_n, X, V, A, temp_path)
    """
    t, V_n, X, V, A = [], [], [], [], []
    for k in range(len(driver_style)):
        t_, V_n_, X_, V_, A_ = variable_speed_sim(car, hyd_mode, driver_style[k], gs_th, acc_curve, dec_curve, 'horizontal_b', veh_max_speed, 0)
        t.append(t_)
        V_n.append(V_n_) 
        X.append(X_)
        V.append(V_)
        A.append(A_)

    temp_path = os.path.join(res_path, 'variable_speed_sim')
    plot_traj(driver_style, t, V_n, X, V, A, temp_path)


def plt_accel_scenario_dc(
    driver_style,
    acc_curve,
    dec_curve,
    veh_max_speed,
    car_info,
    res_path
    ):

    t, V_n, X, V, A = [], [], [], [], []
    for k in range(len(driver_style)):
        t_, V_n_, X_, V_, A_ = accel_sim_dc(driver_style[k], acc_curve, dec_curve, 'horizontal_b', veh_max_speed, 0, 200, 'acc')
        t.append(t_)
        V_n.append(V_n_) 
        X.append(X_)
        V.append(V_)
        A.append(A_)
    temp_path = os.path.join(res_path, 'accel_sim_dc')
    plot_traj(driver_style, t, V_n, X, V, A, temp_path)

    t, V_n, X, V, A = [], [], [], [], []
    for k in range(len(driver_style)):
        t_, V_n_, X_, V_, A_ = accel_sim_dc(driver_style[k], acc_curve, dec_curve, 'horizontal_b', veh_max_speed, 0, 100, 'dec')
        t.append(t_)
        V_n.append(V_n_)
        X.append(X_)
        V.append(V_)
        A.append(A_)

    temp_path = os.path.join(res_path, 'decel_sim_dc')
    plot_traj(driver_style, t, V_n, X, V, A, temp_path)

    t, V_n, X, V, A = [], [], [], [], []
    for k in range(len(driver_style)):
        t_, V_n_, X_, V_, A_ = variable_speed_sim_dc(driver_style[k], acc_curve, dec_curve, 'horizontal_b', veh_max_speed, 0)
        t.append(t_)
        V_n.append(V_n_) 
        X.append(X_)
        V.append(V_)
        A.append(A_)

    temp_path = os.path.join(res_path, 'variable_speed_sim_dc')
    plot_traj(driver_style, t, V_n, X, V, A, temp_path)


def plot_traj(driver_style, t, V_n, X, V, A, temp_path):
    f, axs = plt.subplots(3, 2, figsize=(13,8))

    for k in range(len(driver_style)):
        axs[0, 0].plot(X[k], V[k], label='DS = ' + str(driver_style[k]))
        axs[1, 0].plot(X[k], A[k], alpha=0.5)
        axs[2, 0].plot(V[k], A[k], alpha=0.5)

        axs[0, 1].plot(t[k], V[k])
        axs[1, 1].plot(t[k], A[k], alpha=0.5)

    axs[0, 0].plot(X[k], V_n[k], '--', label='Desired speed ($V_D$)')
    axs[0, 0].set_xlabel('Travel distance, $x_n$ ($m$)')
    axs[0, 0].set_ylabel('Speed, $v_n$ ($m/s$)')
    axs[0, 0].legend()

    axs[1, 0].set_xlabel('Travel distance, $x_n$ ($m$)')
    axs[1, 0].set_ylabel('Acceleration, $a_n$ ($m/s^2$)')

    axs[2, 0].set_xlabel('Speed, $v_n$ ($m/s$)')
    axs[2, 0].set_ylabel('Acceleration, $a_n$ ($m/s^2$)')

    axs[0, 1].set_xlabel('Time, t ($s$)')
    axs[0, 1].set_ylabel('Speed, $v_n$ ($m/s$)')

    axs[1, 1].set_xlabel('Time, t ($s$)')
    axs[1, 1].set_ylabel('Acceleration, $a_n$ ($m/s^2$)')

    plt.tight_layout()
    plt.savefig(temp_path+'.png')
    # plt.show()

    df = pd.DataFrame()
    for k in range(len(driver_style)):
        df['t'+str(k+1)] = t[k]
        df['V_n'+str(k+1)] = V_n[k]
        df['X'+str(k+1)] = X[k]
        df['V'+str(k+1)] = V[k]
        df['A'+str(k+1)] = A[k]
    df.to_csv(temp_path+'.csv', index=False)


def get_sp_MFC(parameters, tp, sstart, sdes, rt, freq):
    driver_style = parameters['driver_style']
    mfc_acc_curve = parameters['mfc_acc_curve']
    mfc_dec_curve = parameters['mfc_dec_curve']
    will_acc_model = parameters['will_acc_model']
    overshoot = parameters['overshoot']
    gs_th = parameters['gs_th']
    max_time = tp[-1] - tp[0]
    dt = 1 / freq
    n = int(np.ceil(max_time / dt) + 1)
    xp = np.arange(tp[0], tp[0] + n * dt, dt)
    sp = sp_MFC(len(xp), sstart, sdes, rt, freq, mfc_acc_curve, mfc_dec_curve, will_acc_model, overshoot, driver_style, gs_th)

    sp_mfc = np.interp(tp, xp, sp)
    return sp_mfc


def get_sp_Gipps(parameters,tp,sstart,sdes, rt, freq, **kwargs):
    if 'beta' in kwargs and 'gamma' in kwargs:
        beta = kwargs['beta']
        gamma = kwargs['gamma']
    else:
        beta = 0.025
        gamma = 0.5
    AlimitGipps = parameters['AlimitGipps']
    max_time = tp[-1] - tp[0]
    dt = 1 / freq
    n = int(np.ceil(max_time / dt) + 1)
    xp = np.arange(tp[0], tp[0] + n * dt, dt)
    sp = sp_Gipps(len(xp), sstart, sdes, rt, freq, AlimitGipps, beta = beta, gamma = gamma)
    sp_gipps = np.interp(tp, xp, sp)
    return sp_gipps


def get_sp_IDM(parameters,tp,sstart,sdes, rt, freq, **kwargs):
    if 'delta' in kwargs:
        delta = kwargs['delta']
    else:
        delta = 4
    AlimitIDM = parameters['AlimitIDM']
    sp = sp_IDM(tp, sstart, sdes, AlimitIDM, delta = delta)
    return sp


def sp_MFC(n, sstart, sdes, rt, freq, acc_p_curve, dec_p_curve, will_acc_model, overshoot, driver_style, gs_th):
    # rt = 0.5
    steps = int((rt / (1 / freq)) - 1)
    if steps > 0:
        delay_map = np.zeros(steps)
    dt = 1 / freq
    sp = [sstart]
    curr_speed = sstart

    gs_cnt = 10
    gear_prev = int(gs_th(curr_speed))
    gear_curr = gear_prev

    for i in range(1, n):
        gs_cnt += 1
        if gs_cnt < 10:
            gear_curr = gear_prev
        else:
            if int(gs_th(curr_speed))>gear_prev:
                gear_curr = gear_prev+1
                gs_cnt = 0
            elif int(gs_th(curr_speed))<gear_prev:
                gear_curr = gear_prev-1
                gs_cnt = 0
            else:
                gear_curr = gear_prev

        if steps == 0:
            a0 = accMFC(curr_speed, sdes, driver_style, acc_p_curve, dec_p_curve, will_acc_model, overshoot)

            if curr_speed > sdes:
                sp.append(curr_speed)
            else:
                curr_speed += a0 * dt
                sp.append(curr_speed)
        else:
            a0 = accMFC(curr_speed, sdes, driver_style, acc_p_curve, dec_p_curve, will_acc_model, overshoot)
            pos_mod = (i - 1) % steps
            if i > steps:
                a_upd = delay_map[pos_mod]
            else:
                # for the first second due to RT acceleration is set to zero
                a_upd = 0

            delay_map[pos_mod] = a0
            if curr_speed > sdes:
                sp.append(curr_speed)
            else:
                curr_speed += a_upd * dt
                sp.append(curr_speed)

        gear_prev = gear_curr

    return sp


def sp_Gipps(n, sstart, sdes, rt, freq, amax, **kwargs):
    if ('beta' in kwargs) and ('gamma' in kwargs):
        gamma = kwargs['gamma']
        beta = kwargs['beta']
    else:
        beta = 0.025
        gamma = 0.5
    steps = int((rt / (1 / freq)) - 1)
    if steps > 0:
        delay_map = np.zeros(steps)
    dt = 1 / freq
    sp = [sstart]
    curr_speed = sstart
    for i in range(1, n):
        if steps == 0:
            a0 = accGipps(curr_speed, amax, sdes, beta=beta, gamma=gamma)
            curr_speed += a0 * dt
            sp.append(curr_speed)
        else:
            a0 = accGipps(curr_speed, amax, sdes, beta=beta, gamma=gamma)
            pos_mod = (i - 1) % steps
            if i > steps:
                a_upd = delay_map[pos_mod]
                curr_speed += a_upd * dt
            else:
                a_upd = 0
                curr_speed += a_upd * dt

            delay_map[pos_mod] = a0
            sp.append(curr_speed)
    return sp


# Calculation of IDM speed ndarray.
def sp_IDM(tp, sstart, sdes, amax, **kwargs):
    if 'delta' in kwargs:
        delta = kwargs['delta']
    else:
        delta = 4
    sp = [sstart]
    curr_speed = sstart
    for dt in np.diff(tp):

        a0 = accIDM(curr_speed, amax, sdes, delta=delta)
        curr_speed += a0 * dt
        # ensure th
        curr_speed = max(0, curr_speed)
        sp.append(curr_speed)
    return sp


def accGipps(s, amax, sdes, **kwargs):
    alpha = 2.5
    beta = 0.025
    gamma = 0.5
    if ('beta' in kwargs) and ('gamma' in kwargs):
        gamma = kwargs['gamma']
        beta = kwargs['beta']
        upt = np.power((1 + gamma), 1 + gamma)
        downt = np.power(gamma, gamma) * np.power((1 + beta), 1 + gamma)
        alpha = upt / downt
    return alpha * amax * (1 - s / sdes) * np.power(max(0, beta + s / sdes), gamma)


def accIDM(s, amax, sdes, **kwargs):
    delta = 4
    if 'delta' in kwargs:
        delta = kwargs['delta']
    res = amax * (1 - pow(s / sdes, delta))
    return res


def follow_leader_gipps(instance, tp, lsp, sim_step, start_dist, start_speed):
    if instance['followDistance'] == True:
        cycle = True
        cycle_dist_interp = instance['distance_cycle']
        dist_travelled = instance['distTravelLimit']
        sdes = 0
    else:
        cycle = False
        sdes = instance['vn']
    amax = instance['an']
    rt = instance['rt']
    bn = instance['bn']

    if 'beta' in instance and 'gamma' in instance:
        beta = instance['beta']
        gamma = instance['gamma']
    else:
        beta = 0.025
        gamma = 0.5
    # res_speed_profile = [lsp[0]]
    # distance traveled for leader/follower
    dtl = start_dist
    dtf = 0
    # speed at the previous moment for leader/follower
    sl_prev = lsp[0]
    sf_prev = start_speed
    #speed history leader/follower
    shl = [sl_prev] * 50
    shf = [sf_prev] * 50
    dhf = [start_dist] * 50

    ceil_RT = int(np.ceil(rt))
    floor_RT = int(np.floor(rt))
    
    # shortcuts
    myinterp = np.interp
    mysp = gipps_calc_vel
    round_RT = round(rt, 2)
    yield sf_prev
    # yield start_dist

    for i in range(1,len(tp)):
        if cycle == True:
            if dtf > dist_travelled:
                break
            else:
                sdes = cycle_dist_interp(dtf)

        sl_curr = lsp[i]
        shl_interp = myinterp(round_RT, [ceil_RT, floor_RT], [shl[-ceil_RT], shl[-floor_RT]])
        dhf_interp = myinterp(round_RT, [ceil_RT, floor_RT], [dhf[-ceil_RT], dhf[-floor_RT]])
        shf_interp = myinterp(round_RT, [ceil_RT, floor_RT], [shf[-ceil_RT], shf[-floor_RT]])
        sf_curr = mysp(sf_prev, shl_interp, sdes, dhf_interp,
                       sim_step, rt, bn, amax, shf_interp, beta = beta, gamma = gamma)

        sl_avg = (sl_prev + sl_curr) / 2
        sf_avg = (sf_prev + sf_curr) / 2
        dtl += sl_avg * (tp[i] - tp[i-1])
        dtf += sf_avg * (tp[i] - tp[i-1])

        if dtl - dtf < 0:
            dtf -= sf_avg * (tp[i] - tp[i - 1])  # (sf_prev + 0.5 * sf_avg * (tp[i] - tp[i-1]))
            sf_curr = sl_prev + sl_curr - sf_prev
            sf_avg = (sf_prev + sf_curr) / 2
            dtf += sf_avg * (tp[i] - tp[i - 1])  # (sf_prev + 0.5 * sf_avg * (tp[i] - tp[i-1]))

        dhf = dhf[1:]
        dhf.append(dtl - dtf)

        if dhf[-1] < 0:
            print("negative distance!")

        yield sf_curr
        # yield dtl - dtf
        # res_speed_profile.append(sf_curr)
        sl_prev = sl_curr
        sf_prev = sf_curr

        shl = shl[1:]
        shl.append(sl_curr)
        shf = shf[1:]
        shf.append(sf_curr)


def follow_leader_idm(instance, tp, lsp, sim_step, start_dist, start_speed):
    if instance['followDistance'] == True:
        cycle = True
        cycle_dist_interp = instance['distance_cycle']
        dist_travelled = instance['distTravelLimit']
        sdes = 0
    else:
        cycle = False
        sdes = instance['vn']
    amax = instance['an']
    bn = instance['bn']
    if 'delta' in instance:
        delta = instance['delta']
    else:
        delta = 4
    # distance traveled for leader/follower
    dtl = start_dist
    dtf = 0
    # speed at the previous moment for leader/follower
    sl_prev = lsp[0]
    sf_prev = start_speed
    dist = start_dist
    yield sf_prev

    for i in range(1,len(tp)):
        if cycle == True:
            if dtf > dist_travelled:
                break
            else:
                sdes = cycle_dist_interp(dtf)
        sl_curr = lsp[i]
        sf_curr = idm_calc_vel(sf_prev, sl_prev, sdes, dist, bn, amax, sim_step)

        sl_avg = (sl_prev + sl_curr) / 2
        sf_avg = (sf_prev + sf_curr) / 2
        dtl += sl_avg * (tp[i] - tp[i - 1])  # (sl_prev + 0.5 * sl_avg * (tp[i] - tp[i-1]))
        dtf += sf_avg * (tp[i] - tp[i - 1])  # (sf_prev + 0.5 * sf_avg * (tp[i] - tp[i-1]))
        dist = dtl - dtf

        yield sf_curr
        sl_prev = sl_curr
        sf_prev = sf_curr
    # return res_speed_profile


def follow_leader_mfc(instance, tp, lsp, sim_step, start_dist, start_speed):
    if instance['followDistance'] == True:
        cycle = True
        cycle_dist_interp = instance['distance_cycle']
        dist_travelled = instance['distTravelLimit']
        sdes = 0
    else:
        cycle = False
        sdes = instance['vn']

    rt = instance['rt']
    # bn = instance['bn']
    will_acc_model = instance['will_acc_model']
    overshoot = instance['overshoot']
    if instance['mfc_curve'] == False:
        _, acc_p_curve, dec_p_curve, _, _, _, _, _, gs_th \
            = mfc_curves(instance['car'], instance['car_id'], 
            instance['hyd_mode'], instance['veh_load'], instance['gs_style'])
    else:
        acc_p_curve = instance['mfc_curve'][0]
        dec_p_curve = instance['mfc_curve'][1]

    driver_style = instance['driver_style']

    if 'beta' in instance and 'gamma' in instance:
        beta = instance['beta']
        gamma = instance['gamma']
    else:
        beta = 0.025
        gamma = 0.5
    # distance traveled for leader/follower
    dtl = start_dist
    dtf = 0
    # speed at the previous moment for leader/follower
    sl_prev = lsp[0]
    sf_prev = start_speed
    sf_curr = sf_prev
    # speed history leader/follower
    shl = [sl_prev] * 50
    shf = [sf_prev] * 50
    dhf = [start_dist] * 50

    ceil_RT = int(np.ceil(rt))
    floor_RT = int(np.floor(rt))
    round_RT = np.round(rt, 2)
    yield sf_curr


    gs_cnt = 10
    gear_prev = int(gs_th(sf_prev))
    gear_curr = gear_prev

    for i in range(1, len(tp)):
        if cycle == True:
            if dtf > dist_travelled:
                break
            else:
                sdes = cycle_dist_interp(dtf)
        sl_curr = lsp[i]
        shl_interp = np.interp(round_RT, [ceil_RT, floor_RT], [shl[-ceil_RT], shl[-floor_RT]])
        dhf_interp = np.interp(round_RT, [ceil_RT, floor_RT], [dhf[-ceil_RT], dhf[-floor_RT]])
        shf_interp = np.interp(round_RT, [ceil_RT, floor_RT], [shf[-ceil_RT], shf[-floor_RT]])

        gs_cnt += 1
        if gs_cnt < 10:
            gear_curr = gear_prev
        else:
            if int(gs_th(sf_prev))>gear_prev:
                gear_curr = gear_prev+1
                gs_cnt = 0
            elif int(gs_th(sf_prev))<gear_prev:
                gear_curr = gear_prev-1
                gs_cnt = 0
            else:
                gear_curr = gear_prev

        sf_curr = mfc_calc_vel(sf_prev, shl_interp, sdes, dhf_interp, \
                                        sim_step, rt, shf_interp, driver_style, acc_p_curve, dec_p_curve, will_acc_model, \
                                        overshoot, gs_cnt)

        sl_avg = (sl_prev + sl_curr) / 2
        sf_avg = (sf_prev + sf_curr) / 2
        dtl += sl_avg * (tp[i] - tp[i - 1])  # (sl_prev + 0.5 * sl_avg * (tp[i] - tp[i-1]))
        dtf += sf_avg * (tp[i] - tp[i - 1])  # (sf_prev + 0.5 * sf_avg * (tp[i] - tp[i-1]))

        if dtl - dtf < 0:
            dtf -= sf_avg * (tp[i] - tp[i - 1])  # (sf_prev + 0.5 * sf_avg * (tp[i] - tp[i-1]))
            sf_curr = sl_prev + sl_curr - sf_prev
            sf_avg = (sf_prev + sf_curr) / 2
            dtf += sf_avg * (tp[i] - tp[i - 1])  # (sf_prev + 0.5 * sf_avg * (tp[i] - tp[i-1]))

        dhf = dhf[1:]
        dhf.append(dtl - dtf)

        if dhf[-1] < 0:
            print("negative distance!")

        yield sf_curr
        sl_prev = sl_curr
        sf_prev = sf_curr
        gear_prev = gear_curr

        shl = shl[1:]
        shl.append(sl_curr)
        shf = shf[1:]
        shf.append(sf_curr)


def gipps_calc_vel(fol_v, lead_v, vn, dist, sim_step, RT, bn, an, prev_v, **kwargs):
    alpha = 2.5
    beta = 0.025
    gamma = 0.5
    if ('beta' in kwargs) and ('gamma' in kwargs):
        gamma = kwargs['gamma']
        beta = kwargs['beta']
        upt = np.power((1 + gamma), 1 + gamma)
        downt = np.power(gamma, gamma) * np.power((1 + beta), 1 + gamma)
        alpha = upt / downt

    min_dist = 2
    fol_v_vn = fol_v / vn
    sim_step_RT = sim_step * RT
    msqrt = math.sqrt

    # sqa=(0.025 + fol_v / vn)
    sqb = (math.pow(bn * sim_step * RT, 2) - bn * (2 * (dist - min_dist) - prev_v * sim_step * RT - math.pow(lead_v, 2) / bn))
    if sqb < 0:
        # print("Gipps - sqrt term negative!")
        sqb = 0
    #
    # ff = fol_v + 2.5 * an * (1 - fol_v / vn) * m.sqrt(sqa) * sim_step  # sim_step * rt is the reaction time
    # fc = (bn * sim_step*RT + m.sqrt(sqb))
    new_vel = min(fol_v + max(bn, alpha * an * (1 - fol_v_vn) * np.power(max(0, beta + fol_v_vn), gamma)) * sim_step, \
                  (bn * sim_step_RT + msqrt(sqb)))
    if new_vel < 0:
        # print("Gipps - new_vel negative!")
        new_vel = 0
    return new_vel


def idm_calc_vel(fol_v, lead_v, vn, dist, bn, an, sim_step, **kwargs):
    if 'delta' in kwargs:
        delta = kwargs['delta']
    else:
        delta = 4
    ds = 2  # intervehicle spacing at a stop
    Th = 1.5  # the desired time headway to the vehicle in front
    bn_idm = abs(bn)  # braking deceleration  b - a positive number

    return max(0, fol_v + max(bn, (an * (1 - pow(fol_v / vn, delta)) + \
                                   -an * (pow(max(0, (ds + fol_v * Th) / dist + (fol_v * (fol_v - lead_v)) / (
                                       2 * math.sqrt(an * bn_idm) * dist)), 2)))) * sim_step)


def mfc_calc_vel(fol_v, lead_v, vn, dist, sim_step, RT, prev_v, driver_style, acc_p_curve, dec_p_curve, will_acc_model,
                       overshoot, gs_cnt):
    min_dist = 2
    sim_step_RT = sim_step * RT
    msqrt = math.sqrt

    potential_acc = accMFC(prev_v, vn, driver_style, acc_p_curve, dec_p_curve, will_acc_model, overshoot)
    # if vn < fol_v:
    #     # potential_acc = acc_p_curve(fol_v) * driver_style * (1 - fol_v / vn) * np.power(max(0, 0.025 + fol_v / vn), 0.5)
    #     potential_acc = 2.5 * 6 * (1 - fol_v / vn) * np.power(max(0, 0.025 + fol_v / vn), 0.5)

    if gs_cnt<3 and potential_acc>0:
        potential_acc = 0

    new_vel = fol_v + potential_acc * sim_step
    if new_vel < 0:
        # print("GippsMFC - new_vel negative!")
        new_vel = 0

    return new_vel


def fcn2minGA_freeflow(params, instance, model, tp, lsp, start_dist, start_speed):
    if isinstance(params, Parameters):
        new_instance = {**instance, **params.valuesdict()}
    else:
        new_instance = {**instance, **params}
    sim_step = 0.1
    m_sp = list(model(new_instance, tp, lsp, sim_step, start_dist, start_speed))
    m_ap = np.append(np.diff(m_sp), 0) * 10
    m_dt = list(np.cumsum([i * 0.1 + 0.005 * j for i, j in zip(m_sp, m_ap)]))

    interp_m = interpolate.interp1d(m_dt, m_sp)
    dist_travelled = instance['distTravelLimit']
    gt_dist = np.arange(50, int(round(dist_travelled - 50)), 2)

    if m_dt[-1] < int(round(dist_travelled - 50)):
        return 1000000

    m_sp = interp_m(gt_dist)

    # Comparison speed/acceleration every 1 meter - window 10meters
    interp_gt = instance['interp_gt']
    gt_sp = interp_gt(gt_dist)

    rel_speed = [k / j for k, j in zip(m_sp, gt_sp)]

    # return rmse_speed
    opt_metric = np.sum([np.log(k) ** 2 for k in rel_speed])
    return opt_metric

