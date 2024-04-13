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
    driver_style_dc,
    gs_style,
    res_path
    ):

    car_info, acc_curve, dec_curve, veh_model_speed, veh_model_acc, veh_model_dec, veh_max_speed, gs_th \
        = msim.mfc_curves(
            car,
            car_id,
            hyd_mode,
            veh_load,
            gs_style,
            res_path,
            )
    
    df = pd.DataFrame()
    df['Spd [m/s]'] = veh_model_speed
    df['Accel [m/s2]'] = veh_model_acc
    df['Decel [m/s2]'] = veh_model_dec
    df.to_csv(os.path.join(res_path, hyd_mode+'_MFC_curves.csv'), index=False)
    # """
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
    # """

    msim.plt_accel_scenario(
        driver_style,
        gs_th,
        acc_curve,
        dec_curve,
        veh_max_speed,
        car_info,
        car,
        hyd_mode,
        res_path
        )
    
    msim.plt_accel_scenario_dc(
        driver_style_dc,
        acc_curve,
        dec_curve,
        veh_max_speed,
        car_info,
        res_path
        )


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

    cars = [
        # 0 - 'ICEV'
        [26966, 7565],      
        # 1 - 'EV'
        [26687],
        # 2 - 'HEV'
        [26712],
    ]

    car_id = cars[2][0]         # SELECT THE CAR
    hyd_mode = ['Na', 'CD', 'CS'][2]  # CD/CS, ONLY VALID FOR HEV, Na = not applicable

    driver_style = [1.0, 0.8, 0.6]
    driver_style_dc = [0, 1, 2] # Class driver_charact() in c2art_env/sim_env/hybrid_mfc/driver_charact.py
    gs_style = [0.9, 0.7, 0.5]
    veh_load = 75 * 0

    db = rno.load_db_to_dictionary(db_name)
    car = rno.get_vehicle_from_db(db, car_id)
    if hyd_mode == 'Na':
        res_path = os.path.join(pathRoot, 'PostProcessing', 'HybridMFC', 'Outputs', car.model, 'curves')
    else:
        res_path = os.path.join(pathRoot, 'PostProcessing', 'HybridMFC', 'Outputs', car.model, hyd_mode, 'curves')
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)

    mfc_main(
        car,
        car_id,
        hyd_mode,
        veh_load,
        driver_style,
        driver_style_dc,
        gs_style[2],
        res_path
        )
