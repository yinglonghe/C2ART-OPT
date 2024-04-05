# %%
# Packages & paths
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal
import collections
import multiprocessing
from lmfit import Parameters, minimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from functools import partial
from IPython.core import display as ICD

pathRoot = os.path.abspath(__file__ + "/../../")
sys.path.append(pathRoot)

import c2art_env.sim_env.hybrid_mfc.reading_n_organizing as rno
import c2art_env.sim_env.hybrid_mfc.microsim as msim
import c2art_env.utils as utils


# %% Define functions
def data_preprocess(data_path, Dataset):
    df1 = pd.read_excel(os.path.join(data_path, Dataset))
    cols_ = df1.columns

    t = np.round(np.arange(round(df1['Time'].iloc[0]*10), round(df1['Time'].iloc[-1]*10+1), 1)*0.1, 1)
    df2 = pd.DataFrame()
    df2['Time [s]'] = t
    for i in range(1, len(df1.columns)):
        if cols_[i] == 'Engaged gear [-]':
            f = interpolate.interp1d(df1['Time'], df1[cols_[i]], kind='nearest', fill_value='extrapolate')
            df2[cols_[i]] = f(t)
        else:
            f = interpolate.interp1d(df1['Time'], df1[cols_[i]], fill_value='extrapolate')
            df2[cols_[i]] = f(t)
    
    df2['Speed [m/s]'] = df2['Vehicle speed [km/h]'] / 3.6
    df2['Accel [m/s2]'] = np.append((utils.numbadiff(df2['Speed [m/s]'].to_numpy()) / utils.numbadiff(df2['Time [s]'].to_numpy())), 0)

    return df2


def plt_driving_traj(df, fig_path):
    f, axs = plt.subplots(len(df.columns)-1, 1, figsize=(10,8), sharex=True)
    for i, col in enumerate(df.columns[1:]):
        axs[i].plot(df['Time [s]'], df[col])
        axs[i].set_ylabel(col)
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(fig_path)
    # plt.show()


def cal_val_traj(df1, df2, car, hyd_mode, res_path):
    cal_res_path = os.path.join(res_path, 'calibration')
    val_res_path = os.path.join(res_path, 'validation')
    for xpath in [cal_res_path, val_res_path]:
        if not os.path.exists(xpath):
            os.makedirs(xpath, exist_ok=True)

    Models = collections.OrderedDict()
    Models['Gipps'] = msim.follow_leader_gipps
    Models['IDM'] = msim.follow_leader_idm
    Models['MFC'] = msim.follow_leader_mfc

    Models_results = collections.OrderedDict()
    for model_name in Models:
        Models_results[model_name] = {}

    results_list = []
    for lap_num, df in enumerate([df1, df2]):
        testDataset = hyd_mode + '_' + str(lap_num+1)

        ltp = np.array(df['Time [s]']-df['Time [s]'].iloc[0])
        lsp = np.array(df['Speed [m/s]'])
        lap = np.array(df['Accel [m/s2]'])

        ltp_long = np.arange(0, round(2 * len(lsp) / 10, 1), 0.1)
        freq = 10

        # This test is about car following
        start_dist = 10000
        start_speed = lsp[0]
        bn = -3
        rt = 1
        beta = 0.025
        gamma = 0.5
        delta = 4

        # ldp = [i * 0.1 + 0.005 * j for i, j in zip(lsp, lap)]
        ldp = [0] + [(lsp[i]+lsp[i-1]) / 2 * 0.1 for i in range(1, len(lsp))]
        ldt = np.cumsum(ldp)
        dist_travelled = ldt[-1]
        interp_gt = interpolate.interp1d(ldt, lsp)

        # Creating a distance traveled with equally spaced points - 2 meters distance
        gt_dist = np.arange(50, int(round(dist_travelled - 50)), 2)
        lsp_dist = interp_gt(gt_dist)

        # Compute acceleration based on the speed diff and the distance traveled
        lap_dist, ltp_dist = [], [0]
        for i in range(1, len(lsp_dist)):
            lap_dist.append((lsp_dist[i] - lsp_dist[i - 1]) * (lsp_dist[i - 1] + lsp_dist[i]) / 4)
            ltp_dist.append(4 / (lsp_dist[i - 1] + lsp_dist[i]))
        lap_dist.append(0)
        lap_dist = np.array(lap_dist)
        ltp_dist = np.cumsum(np.array(ltp_dist))
        ldt_dist = gt_dist

        # Interpolation to get the distance cycle at each point
        df_ = pd.DataFrame.from_dict(
            {'ldt': ldt[1:],
             'lsp': lsp[:-1]}
        )
        df_.drop_duplicates(subset=['ldt'], inplace=True)
        cycle_dist_cubic = interpolate.CubicSpline(df_['ldt'], df_['lsp'])

        # Dictionary for the lmfit optimization
        min_lmfit = {}
        tmpResDict = collections.OrderedDict()
        tmpResDict['GroundTruth'] = {}
        for model_name in Models:
            model = Models[model_name]
            instance = {
                'vn': 0,  ### Desired speed of the follower in m/s
                'bn': bn,  ### Maximum deceleration of the follower in m/s2
                'an': np.random.uniform(0, 8),  ### Maximum acceleration of the follower in m/s2
                's0': 1,  ### Minimum distance in meters
                'rt': rt,  ### Reaction time in sec
                'car_id': car_id,
                'car': car,  ### The car for the MFC
                'hyd_mode': hyd_mode,
                'veh_load': 75 * 0,
                'will_acc_model': 'horizontal_b',  ### Will to accelerate
                'overshoot': 0,  ### overshoot - not used
                'automatic': car.transmission,  ### Gear box
                'mfc_curve': False,
                'gs': False,
                'driver_style': np.random.uniform(0, 1),
                'gs_style': np.random.uniform(0, 1),
                'followDistance': True,
                'distance_cycle': cycle_dist_cubic,
                'distTravelLimit': dist_travelled,
                'interp_gt': interp_gt,
            }

            ficticious_leader = [30] * len(ltp_long)

            # Start of the lmfit optimization
            params_lmfit = Parameters()
            if model == msim.follow_leader_gipps:
                # params_lmfit.add('an', value=1, vary=True, min=0.5, max=4.0)
                params_lmfit.add('an', value=1, vary=True, min=0.5, max=5.0)
                params_lmfit.add('bn', value=bn, vary=False, min=-4.0, max=-1)
                params_lmfit.add('rt', value=rt, vary=False, min=1, max=20)
                params_lmfit.add('beta', value=beta, vary=True, min=0.001, max=5.0)
                params_lmfit.add('gamma', value=gamma, vary=True, min=0.1, max=2.0)
            if model == msim.follow_leader_idm:
                # params_lmfit.add('an', value=1, vary=True, min=0.5, max=4.0)
                params_lmfit.add('an', value=1, vary=True, min=0.5, max=5.0)
                params_lmfit.add('delta', value=delta, vary=True, min=0.1, max=10.0)
                params_lmfit.add('bn', value=bn, vary=False, min=-4.0, max=-1)
            if model == msim.follow_leader_mfc:
                # We need first to run Gipps to calibrate an, bn and rt
                params_lmfit.add('driver_style', value=1, vary=True, min=0.1, max=1)
                params_lmfit.add('gs_style', value=1, vary=True, min=0.1, max=1)
                params_lmfit.add('bn', value=bn, vary=False, min=-4.0, max=-1)
                params_lmfit.add('rt', value=rt, vary=False, min=1, max=20)

            min_lmfit[model_name] = minimize(msim.fcn2minGA_freeflow, params_lmfit,
                                             args=(
                                                 instance, model, ltp_long, ficticious_leader, start_dist, start_speed),
                                             method='Powell', tol=10e-5)

        lsp_dist_cycle = cycle_dist_cubic(gt_dist)
        # Plot the fitted models - LMFit
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ldt_dist, lsp_dist, label='ref. vehicle')
        plt.plot(ldt_dist, lsp_dist_cycle, label='cycle')
        plt.title('Distance - Speed (part of a lap)', fontsize=18)
        df = pd.DataFrame()
        df['ldt_dist'] = ldt_dist
        df['lsp_dist'] = lsp_dist
        df['lsp_dist_cycle'] = lsp_dist_cycle
        df['ltp_dist'] = ltp_dist
        
        for model_name in Models:
            mfit = min_lmfit[model_name]
            model = Models[model_name]
            Params = collections.OrderedDict()
            if model == msim.follow_leader_gipps:
                Params = {
                    'an': mfit.params['an'].value,
                    'rt': mfit.params['rt'].value,
                    'bn': mfit.params['bn'].value,
                    'beta': mfit.params['beta'].value,
                    'gamma': mfit.params['gamma'].value,
                }
            if model == msim.follow_leader_idm:
                Params = {
                    'an': mfit.params['an'].value,
                    'bn': mfit.params['bn'].value,
                    'delta': mfit.params['delta'].value,
                }
            if model == msim.follow_leader_mfc:
                Params = {
                    'rt': mfit.params['rt'].value,
                    # 'bn': mfit.params['bn'].value,
                    'driver_style': mfit.params['driver_style'].value,
                    'gs_style': mfit.params['gs_style'].value,
                }

            Params = collections.OrderedDict(sorted(Params.items()))

            tmpResDict[model_name] = Params

            new_instance = {**instance, **Params}

            sim_step = 0.1
            ficticious_leader = [30] * len(ltp_long)
            m_sp = list(model(new_instance, ltp_long, ficticious_leader, sim_step, start_dist, start_speed))
            m_ap = np.append(np.diff(m_sp), 0) * 10
            # m_dt = list(np.cumsum([i * 0.1 + 0.005 * j for i, j in zip(m_sp, m_ap)]))
            m_dt = list(np.cumsum([0] + [(m_sp[i]+m_sp[i-1]) / 2 * 0.1 for i in range(1, len(m_sp))]))

            interp_m = interpolate.interp1d(m_dt, m_sp, fill_value='extrapolate')
            
            m_sp_1 = np.asarray(m_sp)
            m_dt_1 = np.asarray(m_dt)

            # Store interpolation
            Models_results[model_name]['interp'] = interp_m

            m_sp_dist = interp_m(gt_dist)
            # m_ap_dist = np.append(np.diff(m_sp_dist), 0) * 10
            m_ap_dist, m_tp_dist = [], [0]
            for i in range(1, len(m_sp_dist)):
                m_ap_dist.append((m_sp_dist[i] - m_sp_dist[i - 1]) * ((m_sp_dist[i - 1] + m_sp_dist[i]) / 4))
                m_tp_dist.append(4 / (m_sp_dist[i - 1] + m_sp_dist[i]))
            m_ap_dist.append(0)
            m_ap_dist = np.array(m_ap_dist)
            m_tp_dist = np.cumsum(np.array(m_tp_dist))
            m_dt_dist = gt_dist

            rmse_sp = np.sqrt(mean_squared_error(lsp_dist, m_sp_dist))
            rmse_ap = np.sqrt(mean_squared_error(lap_dist, m_ap_dist))
            rmse_tp = np.sqrt(mean_squared_error(ltp_dist, m_tp_dist))

            similarity_sp = cosine_similarity(lsp_dist.reshape(1,-1), m_sp_dist.reshape(1,-1))

            # Save rmsn errors
            tmpResDict[model_name]['file'] = testDataset
            tmpResDict[model_name]['lap'] = (lap_num + 1)
            tmpResDict[model_name]['model'] = model_name
            tmpResDict[model_name]['rmse_sp'] = rmse_sp
            tmpResDict[model_name]['std_sp'] = np.std(np.array(lsp_dist) - np.array(m_sp_dist))
            tmpResDict[model_name]['rmse_ap'] = rmse_ap
            tmpResDict[model_name]['std_ap'] = np.std(np.array(lap_dist) - np.array(m_ap_dist))
            tmpResDict[model_name]['rmse_tp'] = rmse_tp
            tmpResDict[model_name]['std_tp'] = np.std(np.array(ltp_dist) - np.array(m_tp_dist))
            tmpResDict[model_name]['fit'] = mfit.residual[0]

            print(mfit.params.valuesdict())
            print(mfit.residual[0])
            print('rmse_speed: %.3f' % rmse_sp)
            print('std_sp: %.3f' % np.std(np.array(lsp_dist) - np.array(m_sp_dist)))
            print('rmse_acc: %.3f' % rmse_ap)
            print('std_ap: %.3f' % np.std(np.array(lap_dist) - np.array(m_ap_dist)))
            print('rmse_dist: %.3f' % rmse_tp)
            print('std_tp: %.3f' % np.std(np.array(ltp_dist) - np.array(m_tp_dist)))

            Models_results[model_name]['sp'] = list(m_sp)
            Models_results[model_name]['sp_dist'] = list(m_sp_dist)
            Models_results[model_name]['ap_dist'] = list(m_ap_dist)
            Models_results[model_name]['dt_dist'] = list(m_dt_dist)
            Models_results[model_name]['tp_dist'] = list(m_tp_dist)

            df[model_name+'_sp_dist'] = Models_results[model_name]['sp_dist']
            df[model_name+'_ap_dist'] = Models_results[model_name]['ap_dist']
            df[model_name+'_dt_dist'] = Models_results[model_name]['dt_dist']
            df[model_name+'_tp_dist'] = Models_results[model_name]['tp_dist']

            if model_name == 'Gipps':
                linest = '-.'
            elif model_name == 'IDM':
                linest = ':'
            else:
                linest = '--'

            plt.plot(m_dt_dist, m_sp_dist, label=model_name, linestyle=linest)
        plt.legend(loc='best')
        plt.ylabel('Speed $(m/s)$', fontsize=16)
        plt.xlabel('Distance $(m)$', fontsize=16)

        plt.savefig(os.path.join(cal_res_path, '%s_models_distance_speed_part.png' % testDataset), dpi=150)
        plt.close(fig1)
        df.to_csv(os.path.join(cal_res_path, '%s_models_part.csv' % testDataset), index=False)

        # Macro stats
        print('Average speed GT:%.2f' % np.mean(lsp_dist))

        print('Average absolute acceleration GT:%.2f' % (np.mean(np.absolute(lap_dist))))

        # Save macro results
        tmpResDict['GroundTruth']['avg_speed'] = np.mean(lsp_dist)
        tmpResDict['GroundTruth']['avg_absolute_acceleration'] = np.mean(np.absolute(lap_dist))
        tmpResDict['GroundTruth']['duration'] = len(lsp) / freq
        tmpResDict['GroundTruth']['file'] = testDataset
        tmpResDict['GroundTruth']['model'] = 'GroundTruth'

        for model_name in Models:
            m_sp_dist = Models_results[model_name]['sp_dist']
            print('Average speed %s:%.2f' % (model_name, np.mean(m_sp_dist)))
            m_ap_dist = Models_results[model_name]['ap_dist']
            print('Average absolute acceleration %s:%.2f' % (model_name, np.mean(np.absolute(m_ap_dist))))
            m_sp = Models_results[model_name]['sp']
            print('Time dev %s:%d sec' % (model_name, (len(m_sp) - len(lsp)) / freq))

            # Save macro results
            tmpResDict[model_name]['avg_speed'] = np.mean(m_sp_dist)
            tmpResDict[model_name]['avg_absolute_acceleration'] = np.mean(np.absolute(m_ap_dist))
            tmpResDict[model_name]['time_difference'] = (len(m_sp) - len(lsp)) / freq

        # Append the results for the specific file to the total results' list.
        results_list.append(tmpResDict['GroundTruth'])
        for model_name in Models:
            results_list.append(tmpResDict[model_name])

        # Plot of cycle - Acceleration
        fig1, ax = plt.subplots(figsize=(8, 6))
        plt.plot(ldt_dist, lap_dist, label='ref. vehicle')

        # plt.plot(ldt, cycle_sp, label='cycle')
        for model_name in Models:
            m_dt_dist = Models_results[model_name]['dt_dist']
            m_ap_dist = Models_results[model_name]['ap_dist']
            if model_name == 'Gipps':
                linest = '-.'
            elif model_name == 'IDM':
                linest = ':'
            else:
                linest = '--'
            plt.plot(m_dt_dist, m_ap_dist, label=model_name, linestyle=linest)

        plt.ylabel('Acceleration $(m/s^2)$', fontsize=16)
        plt.xlabel('Distance $(m)$', fontsize=16)
        plt.title('Distance - Acceleration (part of a lap)', fontsize=18)
        plt.legend(loc='best')

        plt.savefig(os.path.join(cal_res_path, '%s_models_distance_acceleration_part.png' % testDataset), dpi=150)

        plt.close(fig1)

        ####################################################
        # Plot of cycle - Speed over Time
        fig1, ax = plt.subplots()
        plt.plot(ltp_dist, lsp_dist, label='ref. vehicle')
        for model_name in Models:
            m_tp_dist = Models_results[model_name]['tp_dist']
            m_sp_dist = Models_results[model_name]['sp_dist']
            plt.plot(m_tp_dist, m_sp_dist, label=model_name)

        plt.ylabel('Speed $(m/s)$')
        plt.xlabel('Time $(s)$')
        plt.title('Time - Speed (1 lap)')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(os.path.join(cal_res_path, '%s_models_time_speed.png' % testDataset), dpi=300)
        plt.close(fig1)

        # Plot of cycle - Acceleration over Time
        fig1, ax = plt.subplots()
        plt.plot(ltp_dist, lap_dist, label='ref. vehicle')
        for model_name in Models:
            m_tp_dist = Models_results[model_name]['tp_dist']
            m_ap_dist = Models_results[model_name]['ap_dist']
            plt.plot(m_tp_dist, m_ap_dist, label=model_name)

        plt.ylabel('Acceleration $(m/s^2)$')
        plt.xlabel('Time $(s)$')
        plt.title('Time - Acceleration (1 lap)')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(os.path.join(cal_res_path, '%s_models_time_acceleration.png' % testDataset), dpi=300)
        plt.close(fig1)

    if not results_list:
        print('empty list')
    else:
        cal_res_df = pd.DataFrame(results_list)
        cal_res_df.to_csv(os.path.join(cal_res_path, 'free_flow_calibration.csv'),
                      index=False)

    # Validation 
    results_list = []
    for lap_num, df in enumerate([df1, df2]):
        testDataset = hyd_mode + '_' + str(lap_num+1)
        if lap_num+1 == 1:
            CalibrateDataset = hyd_mode + '_2'
        elif lap_num+1 == 2:
            CalibrateDataset = hyd_mode + '_1'

        ltp = np.array(df['Time [s]']-df['Time [s]'].iloc[0])
        lsp = np.array(df['Speed [m/s]'])
        lap = np.array(df['Accel [m/s2]'])

        ltp_long = np.arange(0, round(2 * len(lsp) / 10, 1), 0.1)
        freq = 10

        # This test is about car following
        start_dist = 10000
        start_speed = lsp[0]
        bn = -3
        rt = 1
        beta = 0.025
        gamma = 0.5
        delta = 4

        # ldp = [i * 0.1 + 0.005 * j for i, j in zip(lsp, lap)]
        ldp = [0] + [(lsp[i]+lsp[i-1]) / 2 * 0.1 for i in range(1, len(lsp))]
        ldt = np.cumsum(ldp)
        dist_travelled = ldt[-1]
        interp_gt = interpolate.interp1d(ldt, lsp)

        # Creating a distance traveled with equally spaced points - 2 meters distance
        gt_dist = np.arange(50, int(round(dist_travelled - 50)), 2)
        lsp_dist = interp_gt(gt_dist)

        # Compute acceleration based on the speed diff and the distance traveled
        lap_dist, ltp_dist = [], [0]
        for i in range(1, len(lsp_dist)):
            lap_dist.append((lsp_dist[i] - lsp_dist[i - 1]) * (lsp_dist[i - 1] + lsp_dist[i]) / 4)
            ltp_dist.append(4 / (lsp_dist[i - 1] + lsp_dist[i]))
        lap_dist.append(0)
        lap_dist = np.array(lap_dist)
        ltp_dist = np.cumsum(np.array(ltp_dist))
        ldt_dist = gt_dist

        # Interpolation to get the distance cycle at each point
        cycle_dist_cubic = interpolate.CubicSpline(ldt[1:], lsp[:-1])

        # Save results
        tmpResDict = collections.OrderedDict()
        tmpResDict['GroundTruth'] = {}

        lsp_dist_cycle = cycle_dist_cubic(gt_dist)
        fig_dist_speed = plt.figure()
        ax_dist_speed = fig_dist_speed.add_subplot(311)
        ax_dist_speed.plot(ldt_dist, lsp_dist, label='ref. vehicle')
        ax_dist_speed.plot(ldt_dist, lsp_dist_cycle, label='cycle')
        ax_dist_speed.set_title('Distance - Speed (1 lap)')
        ax_dist_speed.set_ylabel('Speed $(m/s)$')
        ax_dist_speed.set_xlabel('Distance $(m)$')

        ax_dist_acc = fig_dist_speed.add_subplot(312)
        ax_dist_acc.plot(ldt_dist, lap_dist, label='ref. vehicle')
        ax_dist_acc.set_title('Distance - Acceleration (1 lap)')
        ax_dist_acc.set_ylabel('Acceleration $(m/s^2)$')
        ax_dist_acc.set_xlabel('Distance $(m)$')

        ax_time_speed = fig_dist_speed.add_subplot(313)
        ax_time_speed.plot(ltp, lsp, label='ref. vehicle')
        ax_time_speed.set_title('Time - Speed (1 lap)')
        ax_time_speed.set_ylabel('Speed $(m/s)$')
        ax_time_speed.set_xlabel('Time $(s)$')

        df = pd.DataFrame()
        df['ldt_dist'] = ldt_dist
        df['lsp_dist'] = lsp_dist
        df['lsp_dist_cycle'] = lsp_dist_cycle
        df['lap_dist'] = lap_dist
        # df['ltp'] = ltp
        # df['lsp'] = lsp

        for model_name in Models:
            model = Models[model_name]
            instance = {
                'vn': 0,  ### Desired speed of the follower in m/s
                'bn': bn,  ### Maximum deceleration of the follower in m/s2
                'an': np.random.uniform(0, 8),  ### Maximum acceleration of the follower in m/s2
                's0': 1,  ### Minimum distance in meters
                'rt': rt,  ### Reaction time in sec
                'car_id': car_id,
                'car': car,  ### The car for the MFC
                'hyd_mode': hyd_mode,
                'veh_load': 75 * 0,
                'will_acc_model': 'horizontal_b',  ### Will to accelerate
                'overshoot': 0,  ### overshoot - not used
                'automatic': car.transmission,  ### Gear box
                'mfc_curve': False,
                'gs': False,
                'driver_style': np.random.uniform(0, 1),
                'gs_style': np.random.uniform(0, 1),
                'followDistance': True,
                'distance_cycle': cycle_dist_cubic,
                'distTravelLimit': dist_travelled,
                'interp_gt': interp_gt,
            }

            ficticious_leader = [30] * len(ltp_long)
            Params = collections.OrderedDict()
            if model == msim.follow_leader_gipps:
                Params = {
                    'an': cal_res_df['an'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'rt': cal_res_df['rt'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'bn': cal_res_df['bn'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'beta': cal_res_df['beta'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'gamma': cal_res_df['gamma'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                }
            if model == msim.follow_leader_idm:
                Params = {
                    'an': cal_res_df['an'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'bn': cal_res_df['bn'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'delta': cal_res_df['delta'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                }
            if model == msim.follow_leader_mfc:
                # We need first to run Gipps to calibrate an, bn and rt
                Params = {
                    'rt': cal_res_df['rt'].loc[(cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'driver_style': cal_res_df['driver_style'].loc[
                        (cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                    'gs_style': cal_res_df['gs_style'].loc[
                        (cal_res_df.model == model_name) & (cal_res_df.file == CalibrateDataset)].values[0],
                }

            Params = collections.OrderedDict(sorted(Params.items()))

            # Save the parameters used for validation
            tmpResDict[model_name] = Params

            new_instance = {**instance, **Params}

            sim_step = 0.1
            m_sp = list(model(new_instance, ltp_long, ficticious_leader, sim_step, start_dist, start_speed))
            m_ap = np.append(np.diff(m_sp), 0) * 10
            # m_dt = list(np.cumsum([i * 0.1 + 0.005 * j for i, j in zip(m_sp, m_ap)]))
            m_dt = list(np.cumsum([0] + [(m_sp[i]+m_sp[i-1]) / 2 * 0.1 for i in range(1, len(m_sp))]))

            interp_m = interpolate.interp1d(m_dt, m_sp)
            
            m_sp_1 = np.asarray(m_sp)
            m_dt_1 = np.asarray(m_dt)

            m_sp_dist = interp_m(gt_dist)
            m_ap_dist = np.append(np.diff(m_sp_dist), 0) * 10
            m_ap_dist, m_tp_dist = [], [0]
            for i in range(1, len(m_sp_dist)):
                m_ap_dist.append((m_sp_dist[i] - m_sp_dist[i - 1]) * ((m_sp_dist[i - 1] + m_sp_dist[i]) / 4))
                m_tp_dist.append(4 / (m_sp_dist[i - 1] + m_sp_dist[i]))
            m_ap_dist.append(0)
            m_ap_dist = np.array(m_ap_dist)
            m_tp_dist = np.cumsum(np.array(m_tp_dist))
            m_dt_dist = gt_dist

            rmse_sp = np.sqrt(mean_squared_error(lsp_dist, m_sp_dist))
            rmse_ap = np.sqrt(mean_squared_error(lap_dist, m_ap_dist))
            rmse_tp = np.sqrt(mean_squared_error(ltp_dist, m_tp_dist))

            rel_speed = [k / j for k, j in zip(m_sp_dist, lsp_dist)]
            opt_metric = np.sum([np.log(k) ** 2 for k in rel_speed])

            # Save rmsn errors
            tmpResDict[model_name]['file'] = testDataset
            tmpResDict[model_name]['model'] = model_name
            tmpResDict[model_name]['rmse_sp'] = rmse_sp
            tmpResDict[model_name]['std_sp'] = np.std(np.array(lsp_dist) - np.array(m_sp_dist))
            tmpResDict[model_name]['rmse_ap'] = rmse_ap
            tmpResDict[model_name]['std_ap'] = np.std(np.array(lap_dist) - np.array(m_ap_dist))
            tmpResDict[model_name]['rmse_tp'] = rmse_tp
            tmpResDict[model_name]['std_tp'] = np.std(np.array(ltp_dist) - np.array(m_tp_dist))
            tmpResDict[model_name]['fit'] = opt_metric

            print('Average speed %s:%.2f' % (model_name, np.mean(m_sp_dist)))
            print('Average absolute acceleration %s:%.2f' % (model_name, np.mean(np.absolute(m_ap_dist))))
            print('Time dev %s:%.2f sec' % (model_name, (len(m_sp) - len(lsp)) / freq))

            # Save macro results
            tmpResDict[model_name]['avg_speed'] = np.mean(m_sp_dist)
            tmpResDict[model_name]['avg_absolute_acceleration'] = np.mean(np.absolute(m_ap_dist))
            tmpResDict[model_name]['time_difference'] = (len(m_sp) - len(lsp)) / freq

            print('fit: %.3d' % opt_metric)
            print('rmse_speed: %.3f' % rmse_sp) 
            print('rmse_acc: %.3f' % rmse_ap)
            print('rmse_dist: %.3f' % rmse_tp)
            ax_dist_speed.plot(m_dt_dist, m_sp_dist, label='%s RMSE(time):%.3f'%(model_name, rmse_tp))
            ax_dist_acc.plot(m_dt_dist, m_ap_dist, label='%s RMSE(time):%.3f'%(model_name, rmse_tp))
            ax_time_speed.plot(m_tp_dist, m_sp_dist, label='%s RMSE(time):%.3f'%(model_name, rmse_tp))

            df[model_name+'_m_dt_dist'] = m_dt_dist
            df[model_name+'_m_sp_dist'] = m_sp_dist
            df[model_name+'_m_ap_dist'] = m_ap_dist
            df[model_name+'_m_tp_dist'] = m_tp_dist

        fig_dist_speed.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.95, hspace=0.20)
        ax_dist_speed.legend(bbox_to_anchor=(1.1, 1.05))
        ax_dist_acc.legend(bbox_to_anchor=(1.1, 1.05))
        ax_time_speed.legend(bbox_to_anchor=(1.1, 1.05))

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(val_res_path, '%s_models_distance_speed.png' % testDataset), dpi=300)
        plt.close(fig_dist_speed)

        # Macro stats
        print('Average speed GT:%.2f' % np.mean(lsp_dist))
        print('Average absolute acceleration GT:%.2f' % (np.mean(np.absolute(lap_dist))))

        # Save macro results
        tmpResDict['GroundTruth']['avg_speed'] = np.mean(lsp_dist)
        tmpResDict['GroundTruth']['avg_absolute_acceleration'] = np.mean(np.absolute(lap_dist))
        tmpResDict['GroundTruth']['duration'] = len(lsp) / freq
        tmpResDict['GroundTruth']['file'] = testDataset
        tmpResDict['GroundTruth']['model'] = 'GroundTruth'

        # Append the results for the specific file to the total results' list.
        results_list.append(tmpResDict['GroundTruth'])
        for model_name in Models:
            results_list.append(tmpResDict[model_name])

    if not results_list:
        print('empty list')
    else:
        val_res_df = pd.DataFrame(results_list)
        val_res_df.to_csv(os.path.join(val_res_path, 'free_flow_validation.csv'),
                      index=False)

# %%
if __name__ == "__main__":
    car_id = 55559 # Golf 8, https://www.cars-data.com/en/volkswagen-golf-1-4-ehybrid-204hp-style-specs/166618/tech
    db_name = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                        'car_database', '2019_07_03_car_db_full')
    db = rno.load_db_to_dictionary(db_name)
    car = rno.get_vehicle_from_db(db, car_id)

    data_path = os.path.join(pathRoot, 'c2art_env', 'sim_env', 'hybrid_mfc',
                        'datasets', 'volkswagen_golf8')
    trajDataset = [
        'logfile_raw2022-04-01_17-55-56__processed_1.xlsx', # CD - driver 2
        'logfile_raw2022-04-01_17-55-56__processed_3.xlsx', # CD - driver 2
        'logfile_raw2022-02-19_19-50-06__processed_2.xlsx', # CS - driver 9
        'logfile_raw2022-02-19_19-50-06__processed_3.xlsx', # CS - driver 9
        ]

    # """
    hyd_mode = 'CD'  # driver 2
    res_path = os.path.join(pathRoot, 'PostProcessing', 'HybridMFC', 'Outputs', car.model, hyd_mode, 'traj_cal_val')
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    df_cd_a = data_preprocess(data_path, trajDataset[0])
    df_cd_a = df_cd_a.loc[(df_cd_a['Time [s]']>=250.0) & (df_cd_a['Time [s]']<=430.0)].reset_index(drop=True)
    df_cd_a['Time [s]'] = df_cd_a['Time [s]'] - df_cd_a['Time [s]'][0]
    df_cd_a.to_csv(os.path.join(res_path, 'Interp_Smooth_'+hyd_mode+'_a'+'.csv'), index=False)

    df_cd_b = data_preprocess(data_path, trajDataset[1])
    # df_cd_b = df_cd_b.loc[df_['Time [s]']>=0.0].reset_index(drop=True)
    df_cd_b['Time [s]'] = df_cd_b['Time [s]'] - df_cd_b['Time [s]'][0]
    df_cd_b.to_csv(os.path.join(res_path, 'Interp_Smooth_'+hyd_mode+'_b'+'.csv'), index=False)

    cal_val_traj(df_cd_a, df_cd_b, car, hyd_mode, res_path)
    # """

    # """
    hyd_mode = 'CS' # driver 9
    res_path = os.path.join(pathRoot, 'PostProcessing', 'HybridMFC', 'Outputs', car.model, hyd_mode, 'traj_cal_val')
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    df_cs_a = data_preprocess(data_path, trajDataset[2])
    df_cs_a = df_cs_a.loc[(df_cs_a['Time [s]']>=1660.0) & (df_cs_a['Time [s]']<=3330.0)].reset_index(drop=True)
    df_cs_a['Time [s]'] = df_cs_a['Time [s]'] - df_cs_a['Time [s]'][0]
    df_cs_a.to_csv(os.path.join(res_path, 'Interp_Smooth_'+hyd_mode+'_a'+'.csv'), index=False)

    df_cs_b = data_preprocess(data_path, trajDataset[3])
    df_cs_b = df_cs_b.loc[(df_cs_b['Time [s]']>=3520.0) & (df_cs_b['Time [s]']<=6650.0)].reset_index(drop=True)
    df_cs_b['Time [s]'] = df_cs_b['Time [s]'] - df_cs_b['Time [s]'][0]
    df_cs_b.to_csv(os.path.join(res_path, 'Interp_Smooth_'+hyd_mode+'_b'+'.csv'), index=False)

    cal_val_traj(df_cs_a, df_cs_b, car, hyd_mode, res_path)
    # """
# %%
