# %%
# Packages & paths
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product
from functools import partial
import collections
from IPython.core import display as ICD

pathRoot = os.path.abspath(__file__ + "/../../")
sys.path.append(pathRoot)

import c2art_env.sim_env.hybrid_mfc.reading_n_organizing as rno
import c2art_env.sim_env.hybrid_mfc.microsim as msim
import c2art_env.sim_env.hybrid_mfc.vehicle_specs_class as vcc


def process_0_100_accelerations(selected_car, Models):
    carid = int(selected_car["Unnamed: 0"])  # Category-Item
    results_list_item = []
    error_car_list_item = 'ok'
    print("carid: %d" % carid)
    hyd_mode = None

    if selected_car["Drive-Drive system"] in ['fuel engine']:
        powertrain = 'conventional'
    elif selected_car["Drive-Drive system"] in ['electric engine']:
        powertrain = 'electric'
    elif selected_car["Drive-Drive system"] in ['hybrid', 'plug-in hybrid']:
        powertrain = 'hybrid'
        hyd_mode = 'CS'
    my_car = vcc.veh_specs(selected_car, powertrain)

    max_power_gt = my_car.max_power

    rt = 0.1
    will_acc_model = 'horizontal_b'  # gipps, idm, horizontal
    driver_style = 1
    overshoot = 0  # m/s NOT valid for horizontal wTa line
    over_f = 1.2  # surpassing (%) of the estimated acceleration curve of the vehicle
    freq = 10  # frequency
    try:
        mfc_acc_curve = 0
        mfc_dec_curve = 0

        for model_name in Models:
            if model_name == 'MFC':
                veh_load = 75 * 0
                _ , mfc_acc_curve, mfc_dec_curve, _, _, _, _ \
                    = msim.mfc_curves(
                        my_car,
                        carid,
                        hyd_mode,
                        veh_load
                        )

        sstart = 0  # start speed
        sdes = round(my_car.top_speed, 2)  # target speed in m/s
        time0_100_gt = my_car.time_0_100

        tp = np.arange(0, 200, 1 / freq)

        parameters = {
            'driver_style': driver_style,  ### Driver style for MFC [0,1]
            'mfc_acc_curve': mfc_acc_curve,  ### Acceleration curve MFC
            'mfc_dec_curve': mfc_dec_curve,  ### Deceleration curve MFC
            'will_acc_model': will_acc_model,  ### Will to accelerate
            'overshoot': overshoot,  ### overshoot - not used
            'AlimitGipps': max(mfc_acc_curve.c[3]), # mfc_acc_curve(0.32 * sdes / 3.6),  # Max acceleration Gipps
            'AlimitIDM': max(mfc_acc_curve.c[3]) # mfc_acc_curve(0)  # Max acceleration IDM
        }

        results_list_item.append(carid)
        results_list_item.append(max_power_gt)
        results_list_item.append(time0_100_gt)

        for model_name in Models:
            model = Models[model_name]
            sp = model(parameters, tp, sstart, sdes, rt, freq)
            ap = np.append(np.diff(sp), 0) * 10

            cnt_car_ok = 0
            cnt_car_over = 0
            for j in range(len(sp)):
                if sp[j] >= 100 / 3.6 and sp[j - 1] < 100 / 3.6:
                    time0_100_sim = tp[j]
                    break
                if round(ap[j], 2) > round(over_f * mfc_acc_curve(sp[j]), 2):
                    cnt_car_over += 1
                else:
                    cnt_car_ok += 1

            results_list_item.append(time0_100_sim)
            results_list_item.append(cnt_car_ok)
            results_list_item.append(cnt_car_over)

    except:
        error_car_list_item = carid
        print("Not able to compute %s curve for carid: %d" % (model_name, carid))
        return results_list_item, error_car_list_item

    return results_list_item, error_car_list_item


def multi_process_accelerations(car_list):
    Models = collections.OrderedDict()
    Models['MFC'] = msim.get_sp_MFC
    Models['Gipps'] = msim.get_sp_Gipps
    Models['IDM'] = msim.get_sp_IDM

    # p = Pool(processes=4)
    # success = p.map(partial(process_0_100_accelerations, Models=Models), car_list)
    success = [process_0_100_accelerations(car, Models) for car in car_list]
    # for i in range(len(success)):
    #     print(success[i])

    labels = []
    labels.append('carid')
    labels.append('power')
    labels.append('time_gt')
    for model_name in Models:
        labels.append('time_%s' % model_name)
        labels.append('eng_ok_%s' % model_name)
        labels.append('eng_over_%s' % model_name)
    results_list = []
    for i in range(3):
        results_list.append([])
    for model_name in Models:
        results_list.append([])
        results_list.append([])
        results_list.append([])

    error_car_list = []

    for i in range(len(success)):
        print(i)
        if success[i][1] == 'ok':
            try:
                results_list[0].append(success[i][0][0])
            except:
                print('please check')
            results_list[1].append(success[i][0][1])
            results_list[2].append(success[i][0][2])
            cnt = 3
            for model_name in Models:
                model = Models[model_name]
                results_list[cnt].append(success[i][0][cnt])
                results_list[cnt + 1].append(success[i][0][cnt + 1])
                results_list[cnt + 2].append(success[i][0][cnt + 2])
                cnt += 3
        else:
            error_car_list.append(success[i][1])
    return results_list, error_car_list, labels


if __name__ == "__main__":
    # Select valid hybrid specs from the car database
    db_name = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                            'car_database', '2019_07_03_car_db_full')

    df = pd.read_csv(db_name+'.csv')
    df_hyd = df[
        (df['Drive-Drive system'].isin(['hybrid', 'plug-in hybrid'])) &
        (df['Transmission  / Gear ratio-Gear Box Ratios'].notnull()) &
        (df['Transmission  / Gear ratio-Final drive'].notnull()) &
        (df['Fuel Engine-Max power'].notnull()) &
        (df['Fuel Engine-Max power RPM'].notnull()) &
        (df['Electric Engine-Total max power'].notnull()) &
        (df['Electric Engine-Max torque'].notnull()) &
        (df['Drive-Total max power']<=500)
    ]
    car_list = df_hyd.to_dict('records')

    results_list, error_car_list, labels = multi_process_accelerations(car_list)

    res_dict = {label: results_list[i] for i, label in enumerate(labels)}

    res_path = os.path.join(pathRoot, 'PostProcessing', 'HybridMFC', 'Outputs', 'Hyd_0_100_val')
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    df = pd.DataFrame.from_dict(res_dict)[
        labels]
    df.to_csv(os.path.join(res_path, 'res_0_100.csv'), index=None, float_format='%.10f')

