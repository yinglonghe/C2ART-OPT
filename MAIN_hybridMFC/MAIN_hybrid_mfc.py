# %%
# Packages & paths
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from itertools import product
from functools import partial
from IPython.core import display as ICD
from scipy.interpolate import interp1d

pathRoot = os.path.abspath(__file__ + "/../../")
sys.path.append(pathRoot)

import c2art_env.sim_env.hybrid_mfc.reading_n_organizing as rno
import c2art_env.sim_env.hybrid_mfc.microsim as msim


def mfc_main(
    car,
    car_id,
    hyd_mode,
    veh_load,
    driver_style,
    gs_style,
    res_path
    ):

    car_info, ap_curve, dp_curve, veh_model_speed, veh_model_acc, veh_model_dec, veh_model_acc_max, veh_max_speed, gs_th \
        = msim.mfc_curves(
            car,
            car_id,
            hyd_mode,
            veh_load,
            gs_style,
            )

    plt.plot(veh_model_speed, veh_model_acc, '--', label='($a_{ap, gs}$)')
    plt.plot(veh_model_speed, veh_model_acc_max, label='($a_{ap, max}$)')
    plt.legend()
    plt.xlabel(r'Veh. spd ($m/s$)')
    plt.ylabel(r'Veh. accel ($m/s^2$)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, 'ap_gs_max_curves_comp.png'))
    plt.close()
    
    df = pd.DataFrame()
    df['Spd [m/s]'] = veh_model_speed
    df['AccelMax [m/s2]'] = veh_model_acc_max
    df['Accel [m/s2]'] = veh_model_acc
    df['Decel [m/s2]'] = veh_model_dec
    df.to_csv(os.path.join(res_path, 'ap_dp_curves.csv'), index=False)

    msim.plt_exp_val_spd_accel(
        pathRoot, 
        car_id, 
        car_info, 
        hyd_mode,
        veh_model_speed, 
        veh_model_acc,
        veh_model_dec, 
        veh_max_speed,
        res_path
        )

    (f_des, v0, duration, cycle_des) = [
        (interp1d(       # Varying speed limits scenario
            [0, 210, 700, 1050, 1400, 3150, 4550, 5250],  # Position (m)
            [20, 20, 30, 5, 25, veh_max_speed, 15, 0],  # Desired speed (m/s)
            kind="next", fill_value="extrapolate",), 0, 250, 'var'),
        (interp1d(       # Acceleration scenario
            [0, 5250],  # Position (m)
            [veh_max_speed, veh_max_speed],  # Desired speed (m/s)
            kind="next", fill_value="extrapolate",), 0, 100, 'acc'),
        (interp1d(
            [0, 5250],  # Position (m)
            [0, 0],  # Desired speed (m/s)
            kind="next", fill_value="extrapolate",), veh_max_speed, 20, 'dec'),
        ][2]        ### Changable
    
    dt = 0.1        # The simulation step in seconds
    t = np.arange(0, duration + dt, dt) # sample time series
    x, v, a, v_des, gr_idx, trq_gr_in, w_gr_in, p_gr_in = \
        np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)).astype(int), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    soc_ref, soc, fc, trq_ice, p_ice, p_fuel, trq_em, p_em, p_batt, i_batt, ef = \
        np.ones(len(t))*0.3, np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    if hyd_mode == 'CD':
        soc0 = 0.8
    else:
        soc0 = 0.3
    dw_gr_in = np.zeros(len(t))

    gs_cnt = 10
    for i in range(len(t)):
        if i == 0:
            v[i] = v0
            soc[i] = soc0
            v_des[i] = f_des(x[i])
            gr_idx[i] = int(gs_th(v[i]))
            w_gr_in[i] = v[i] / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gr_idx[i]]) # rad/s
        else:
            gs_cnt += 1
            if gs_cnt < 10:
                gr_idx[i] = gr_idx[i-1]
            else:
                if int(gs_th(v[i-1]))>gr_idx[i-1]:
                    gr_idx[i] = gr_idx[i-1]+1
                    gs_cnt = 0
                elif int(gs_th(v[i-1]))<gr_idx[i-1]:
                    gr_idx[i] = gr_idx[i-1]-1
                    gs_cnt = 0
                else:
                    gr_idx[i] = gr_idx[i-1]

            a[i] = msim.accMFC(v[i-1], v_des[i-1], driver_style, ap_curve, dp_curve, will_acc_model='horizontal_b', overshoot=0)
            if gs_cnt<3 and a[i]>0:
                a[i] = 0
            v[i] = max(v[i-1] + a[i] * dt, 0)
            a[i] = (v[i] - v[i-1]) / dt
            x[i] = x[i-1] + (v[i] + v[i-1]) / 2 * dt
            v_des[i] = f_des(x[i])
            
            a_r = (car.f0 + car.f1*v[i]*3.6 + car.f2*pow(v[i]*3.6, 2)) / car.total_mass
            a_t = a[i] + a_r
            trq_gr_in[i] = a_t*car.total_mass*car.tire_radius / car.driveline_efficiency / (car.final_drive * car.gr[gr_idx[i]]) # Nm, Debug=50
            # https://www.engineeringtoolbox.com/angular-velocity-acceleration-power-torque-d_1397.html
            # https://www.convertunits.com/from/RPM/to/rad/sec
            w_gr_in[i] = v[i] / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gr_idx[i]]) # rad/s, Debug=315
            p_gr_in[i] = trq_gr_in[i] * w_gr_in[i]
            # dw_gr_in[i] = a[i] / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gr_idx[i]])
            dw_gr_in[i] = a_t / (car.tire_radius * (1 - car.driveline_slippage)) * (car.final_drive * car.gr[gr_idx[i]])

            if car.powertrain in ['hybrid', 'plug-in hybrid']:
                if hyd_mode == 'CS':
                    soc[i], fc[i], trq_ice[i], p_ice[i], p_fuel[i], trq_em[i], p_em[i], p_batt[i], i_batt[i], ef[i] = \
                        msim.powertrain_hyd_cs_aecms(
                            car,
                            soc[i-1],
                            soc_ref[i-1],
                            w_gr_in[i],  
                            trq_gr_in[i],   
                            dw_gr_in[i],         
                        )
                if hyd_mode == 'CD':
                    soc[i], trq_em[i], p_em[i], p_batt[i], i_batt[i] = \
                        msim.powertrain_hyd_cd(
                            car,
                            soc[i-1],
                            w_gr_in[i],  
                            trq_gr_in[i],
                        )


    trq_gr_in[0] = trq_gr_in[1]
    p_gr_in[0] = trq_gr_in[0]*w_gr_in[0]
    a[0] = a[1]

    data1_plot = [
        t, v_des, x, v, a, gr_idx, w_gr_in*60/(2*np.pi), trq_gr_in, p_gr_in*1e-3, dw_gr_in]
    data1_label = [
        'time(s)', 'vn(m/s)', 'x(m)', 'v(m/s)', 'a(m/s2)', 'gr_idx', 'w_gr_in\n(rpm)', 'trq_gr_in\n(Nm)', 'p_gr_in\n(kW)', 'dw_gr_in']
    fig, axs = plt.subplots(len(data1_plot), sharex=True, figsize=(11,6))
    for i in range(len(data1_plot)):
        axs[i].plot(t, data1_plot[i])
        if data1_label[i]=='v(m/s)':
            axs[i].plot(t, v_des, 'r--')
        if data1_label[i] in ['time(s)', 'vn(m/s)']:
            continue
        axs[i].set_ylabel(data1_label[i], rotation=0, labelpad=30)
        axs[i].grid()
    axs[i].set_xlabel('time(s)')
    axs[i].set_xlim(0, t[-1])
    plt.savefig(os.path.join(res_path, 'mfc_driving_cycle_'+cycle_des+'_vehdyn.png'))
    # plt.show()

    if car.powertrain in ['hybrid', 'plug-in hybrid']:
        if hyd_mode == 'CS':
            data2_plot = [
                soc, fc*1e3, trq_ice, p_ice*1e-3, trq_em, p_em*1e-3, p_batt*1e-3, i_batt, ef,]
            data2_label = [
                'soc', 'fc(ml)', 'trq_ice(Nm)', 'p_ice(kW)', 'trq_em(Nm)', 'p_em(kW)', 'p_batt(kW)', 'i_batt(A)', 'ef',]
        if hyd_mode == 'CD':
            data2_plot = [soc, trq_em, p_em*1e-3, p_batt*1e-3, i_batt,]
            data2_label = ['soc', 'trq_em(Nm)', 'p_em(kW)', 'p_batt(kW)', 'i_batt(A)',]
    fig, axs = plt.subplots(len(data2_plot), sharex=True, figsize=(11,6))
    for i in range(len(data2_plot)):
        axs[i].plot(t, data2_plot[i])
        axs[i].set_ylabel(data2_label[i], rotation=0, labelpad=30)
        axs[i].grid()
        if data2_label[i]=='fc(ml)':
            axs[i].text(200, 0.2, 'avg. fc (l/100km): '+str(round(np.sum(fc)/(x[-1]*1e-5), 1)))
    axs[i].set_xlabel('time(s)')
    axs[i].set_xlim(0, t[-1])
    plt.savefig(os.path.join(res_path, 'mfc_driving_cycle_'+cycle_des+'_powdyn.png'))
    plt.show()
            
    df = pd.DataFrame.from_dict(dict(zip(data1_label+data2_label, data1_plot+data2_plot)))
    df.to_csv(os.path.join(res_path, \
        'ds'+str(round(driver_style,1)).replace('.', '')+'_gs'+str(round(gs_style,1)).replace('.', '')+'_'+hyd_mode+'_'+cycle_des+'.csv'), index=False)


if __name__ == "__main__":
    # Select valid hybrid specs from the car database
    db_name = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                            'car_database', '2019_07_03_car_db_full')
    # %%
    # Select the car and model its driving capabilities
    # 26966 - Space Star, https://www.cars-data.com/en/mitsubishi-space-star-1.2-life-specs/73285
    # 7565 - Golf, https://www.cars-data.com/en/volkswagen-golf-2.0-tdi-150hp-highline-specs/59579
    # 26687 - Ioniq, https://www.cars-data.com/en/hyundai-ioniq-electric-comfort-specs/76078
    # 26712 - Niro, https://www.cars-data.com/en/kia-niro-1.6-gdi-hybrid-executiveline-specs/76106
    # 55559 - Golf 8, https://www.cars-data.com/en/volkswagen-golf-1-4-ehybrid-204hp-style-specs/166618/tech, [f0, f1, f2] = [115.5, 0.106, 0.03217]

    cars = [
        # 0 - 'ICEV'
        [26966, 7565],      
        # 1 - 'EV'
        [26687],
        # 2 - 'HEV'
        [26712, 55559],
    ]

    car_id = cars[2][1]         ### Changable, SELECT THE CAR 
    if car_id in cars[2]:
        hyd_mode = ['CD', 'CS'][1]  ### Changable, CD/CS, ONLY VALID FOR HEV, Na = not applicable
    else:
        hyd_mode = 'Na'

    driver_style = [1.0, 0.8, 0.6][2]   ### Changable
    gs_style = [1.0, 0.8, 0.6][2]       ### Changable
    veh_load = 75 * 0

    db = rno.load_db_to_dictionary(db_name)
    car = rno.get_vehicle_from_db(db, car_id)

    res_path = os.path.join(pathRoot, 'PostProcessing', 'HybridMFC', 'Outputs', car.model, 'ds'+str(round(driver_style,1)).replace('.', '')+'_gs'+str(round(gs_style,1)).replace('.', ''))
    if hyd_mode != 'Na':
        res_path = os.path.join(res_path, hyd_mode)
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)

    mfc_main(
        car,
        car_id,
        hyd_mode,
        veh_load,
        driver_style,
        gs_style,
        res_path
        )
