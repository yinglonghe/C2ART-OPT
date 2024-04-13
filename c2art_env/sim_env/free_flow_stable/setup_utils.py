import os, re, math, time, datetime, shutil
import numpy as np
from numba import njit, types 
from numba.typed import Dict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import c2art_env.sim_env.free_flow_stable.free_flow_env_njit as ff_env
from c2art_env.veh_model.mfc.mfc_constraints import mfc_curves
from c2art_env.opt_tool.ga import geneticalgorithm as ga
from c2art_env import utils


def read_config(pathConfig):
    with open(pathConfig) as f:
        config = {}
        for line in f:
            (col_a, col_b) = line.split()
            config[col_a.replace(':', '')] = int(col_b)
    return config


def read_constraints(pathConstraints):
    with open(pathConstraints) as f:
        values = {}
        for line in f:
            (col_a, col_b) = line.split()
            values[col_a.replace(':', '')] = float(col_b)
    return values


def read_bounds(pathBounds, bounds, LB, UB):
    with open(pathBounds) as f:
        for line in f:
            (col_a, col_b, col_c) = line.split()
            bounds.append(col_a.replace(':', ''))
            LB.append(float(col_b))
            UB.append(float(col_c))
    return bounds, LB, UB


def read_calibrated_parms(pathResult, ModelID, CalVehID, CalExpIDX, GOFName):
    with open(os.path.join(pathResult, 'Model '+str(ModelID), 'VehicleID ' + str(CalVehID),\
            'Calibration_Report_Exp_' + str(CalExpIDX+1) + '_' + GOFName + '.txt')) as f:
        for line in f:
            x = [float(x) for x in line.strip().split('\t')]
    del f, line
    return np.array(x[:-1])


def setup_constraints(pathInput, VehID, CarID):
    parm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64
    )
    # parm['veh_a_max'] = 4
    # parm['veh_a_min'] = -5
    # parm['mfc_veh_load'] = 250
    # parm['mfc_rolling_coef'] = 0.009
    # parm['mfc_aero_coef'] = 0.25
    # parm['mfc_res_coef_1'] = 0.84
    # parm['mfc_res_coef_2'] = -71.735
    # parm['mfc_res_coef_3'] = 2.7609
    # parm['mfc_ppar_0'] = 0.0045
    # parm['mfc_ppar_1'] = -0.1710
    # parm['mfc_ppar_2'] = -1.8835
    # parm['mfc_mh_base'] = 0.75

    values = read_constraints(os.path.join(pathInput, 'Constraints', 'Vehicle ID '+str(VehID), 'veh_acc.txt'))
    parm['veh_a_max'] = values['veh_a_max']
    parm['veh_a_min'] = -values['veh_a_min']
    parm['acc_a_max'] = values['acc_a_max']
    parm['acc_a_min'] = -values['acc_a_min']
    del values

    values = read_constraints(os.path.join(pathInput, 'Constraints', 'Vehicle ID '+str(VehID), 'mfc.txt'))

    parm['mfc_veh_load'] = values['veh_load']
    parm['mfc_rolling_coef'] = values['rolling_coef']
    parm['mfc_aero_coef'] = values['aero_coef']
    parm['mfc_res_coef_1'] = values['res_coef_1']
    parm['mfc_res_coef_2'] = values['res_coef_2']
    parm['mfc_res_coef_3'] = values['res_coef_3']
    parm['mfc_ppar_0'] = values['ppar_0']
    parm['mfc_ppar_1'] = values['ppar_1']
    parm['mfc_ppar_2'] = values['ppar_2']
    parm['mfc_mh_base'] = values['mh_base']
    mfc_model = mfc_curves(
        CarID,
        parm['mfc_veh_load'],
        parm['mfc_rolling_coef'],
        parm['mfc_aero_coef'],
        parm['mfc_res_coef_1'],
        parm['mfc_res_coef_2'],
        parm['mfc_res_coef_3'],
        parm['mfc_ppar_0'],
        parm['mfc_ppar_1'],
        parm['mfc_ppar_2'],
        parm['mfc_mh_base'])
    del values
    # parm['mfc_speed'] = mfc_model['mfc_speed']
    # parm['mfc_acc'] = mfc_model['mfc_acc']
    # parm['mfc_dec'] = mfc_model['mfc_dec']
    parm['mfc_f_0'] = mfc_model['mfc_f_0']
    parm['mfc_f_1'] = mfc_model['mfc_f_1']
    parm['mfc_f_2'] = mfc_model['mfc_f_2']
    parm['car_width'] = mfc_model['car_width']
    parm['car_height'] = mfc_model['car_height']
    parm['car_mass'] = mfc_model['car_mass']
    parm['car_phi'] = mfc_model['car_phi']

    parm_mfc_curve = np.array([
        mfc_model['mfc_speed'],
        mfc_model['mfc_acc'],
        mfc_model['mfc_dec']
    ])

    return parm, parm_mfc_curve


def setup_bounds(pathInput, ModelID, VehID, optimize_resistance_and_load):
    bounds = []
    LB = []
    UB = []

    ModelClass = math.ceil(ModelID/18)
    bounds, LB, UB = read_bounds(os.path.join(pathInput, 'Bounds', \
        'Model Class '+str(ModelClass), 'parameters.txt'), bounds, LB, UB)

    TurningClass = ModelID % 18
    if TurningClass == 0 or TurningClass >= 10:
        bounds, LB, UB = read_bounds(os.path.join(pathInput, 'Bounds', \
            'turning_factors.txt'), bounds, LB, UB)

    ModelType = ModelID % 9
    if ModelType == 0:
        ModelType = 9
    if ModelType >= 4:
        bounds, LB, UB = read_bounds(os.path.join(pathInput, 'Bounds', \
            'actuation_delay.txt'), bounds, LB, UB)

    if ModelType >= 7:
        if optimize_resistance_and_load == 1:
            bounds, LB, UB = read_bounds(os.path.join(pathInput, 'Bounds', \
                'veh_load.txt'), bounds, LB, UB)
            bounds, LB, UB = read_bounds(os.path.join(pathInput, 'Bounds', \
                'Vehicle ID '+str(VehID), 'road_loads.txt'), bounds, LB, UB)
    
    return bounds, LB, UB


def setup_data(pathInput, VehID, SL_Exp_Names):
    x_lead_real_set = [None]*len(SL_Exp_Names)
    v_lead_real_set = [None]*len(SL_Exp_Names)
    v_des_data_set = [None]*len(SL_Exp_Names)
    t_step_data_set = [None]*len(SL_Exp_Names)

    for i in range(len(SL_Exp_Names)):
        x_lead_real_set[i], v_lead_real_set[i], v_des_data_set[i], t_step_data_set[i] \
            = import_trajectory_data(pathInput, VehID, SL_Exp_Names[i])
    del i

    x_lead_real_concat = []
    v_lead_real_concat = []
    xv_concat_split = [0]
    for i in range(len(t_step_data_set)):
        x_lead_real_concat += x_lead_real_set[i]
        v_lead_real_concat += v_lead_real_set[i]
        xv_concat_split.append(len(x_lead_real_concat))

    track_data = pd.read_csv(os.path.join(pathInput, 'trackData2D.csv')).to_dict(orient='list')

    track_data = np.array([
        track_data['X'],            # 0
        track_data['SlopeSin'],     # 1
        track_data['Curvature'],    # 2
        ])

    track_len = track_data[0][-1]   # 2-X

    return np.array(x_lead_real_concat), np.array(v_lead_real_concat), np.array(xv_concat_split), np.array(v_des_data_set), \
        np.array(t_step_data_set), track_len, track_data


def import_trajectory_data(pathInput, VehID, ExpName):

    experimentData = pd.read_csv(os.path.join(pathInput, 'PyData', ExpName+'.csv'), index_col=None)
    
    v_des_data = float(re.search('_V(\d+)', ExpName, re.IGNORECASE).group(1)) / 3.6

    # Sample rate
    t_step_data = experimentData['Time'][1] - experimentData['Time'][0]
    
    K = experimentData.shape[0]

    v_real = experimentData[['SpeedDoppler1']].T.values.tolist()
    x_real = [[None]*K]
    for k in range(K):
        if k == 0:
            x_real[0][k] = experimentData['X1'][0]
        else:
            x_real[0][k] = x_real[0][k-1] + ((v_real[0][k-1]+v_real[0][k])/2) * t_step_data

    return x_real[0], v_real[0], v_des_data, t_step_data


@njit(nogil=True)
def setup_model(x, ModelID, parm_cstr, optimize_resistance_and_load, *args):
    parm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64
    )

    parm['perception_delay'] = 0
    parm['car_mass'] = parm_cstr['car_mass']        # Added
    parm['car_phi'] = parm_cstr['car_phi']          # Added
    parm['veh_a_max'] = parm_cstr['veh_a_max']          # Added
    parm['veh_a_min'] = parm_cstr['veh_a_min']          # Added
    parm['acc_a_max'] = parm_cstr['acc_a_max']          # Added
    parm['acc_a_min'] = parm_cstr['acc_a_min']          # Added
    parm['veh_load'] = parm_cstr['mfc_veh_load']
    parm['rolling_coef'] = parm_cstr['mfc_rolling_coef']
    parm['aero_coef'] = parm_cstr['mfc_aero_coef']
    parm['res_coef_1'] = parm_cstr['mfc_res_coef_1']
    parm['res_coef_2'] = parm_cstr['mfc_res_coef_2']
    parm['res_coef_3'] = parm_cstr['mfc_res_coef_3']
    parm['f_0'] = parm_cstr['mfc_f_0']
    parm['f_1'] = parm_cstr['mfc_f_1']
    parm['f_2'] = parm_cstr['mfc_f_2']

    ctrl_cstr_onoff = 'off'

    if 1 <= ModelID <= 18:
        control_type = 'acc_free_flow_pid'
        parm['k_p'] = x[0]
        parm['k_i'] = x[1]
        parm['k_d'] = x[2]
        nPar = 2
    ###

    turning_class_id = ModelID % 18
    if turning_class_id == 0:
        turning_class_id = 18
    ###
    if turning_class_id >= 10:
        turning_effect = 'on'
        parm['a_turning'] = -x[nPar+1]
        nPar = nPar + 1
    else:
        turning_effect = 'off'
    ###

    type_id = ModelID % 9
    if type_id == 0:
        type_id = 9
    ###

    if type_id == 1:
        veh_dyn_type = 'car_none'
        veh_cstr_type = 'off'
    elif type_id == 2:
        veh_dyn_type = 'car_none'
        veh_cstr_type = 'constant'
    elif type_id == 3:
        veh_dyn_type = 'car_none'
        veh_cstr_type = 'mfc'
    elif type_id == 4:
        veh_dyn_type = 'car_linear'
        veh_cstr_type = 'off'
        parm['tau_i'] = x[nPar+1]
    elif type_id == 5:
        veh_dyn_type = 'car_linear'
        veh_cstr_type = 'constant'
        parm['tau_i'] = x[nPar+1]
    elif type_id == 6:
        veh_dyn_type = 'car_linear'
        veh_cstr_type = 'mfc'
        parm['tau_i'] = x[nPar+1]
    elif type_id == 7:
        veh_dyn_type = 'car_nonlinear'
        veh_cstr_type = 'off'
        parm['tau_i'] = x[nPar+1]
        if optimize_resistance_and_load == 1:
            parm['veh_load'] = x[nPar+2]
            # parm['rolling_coef'] = x[nPar+3]
            # parm['aero_coef'] = x[nPar+4]
            # parm['res_coef_1'] = x[nPar+5]
            # parm['res_coef_2'] = x[nPar+6]
            # parm['res_coef_3'] = x[nPar+7]
            parm['f_0'] = x[nPar+3]
            parm['f_1'] = x[nPar+4]
            parm['f_2'] = x[nPar+5]
        ###
    elif type_id == 8:
        veh_dyn_type = 'car_nonlinear'
        veh_cstr_type = 'constant'
        parm['tau_i'] = x[nPar+1]
        if optimize_resistance_and_load == 1:
            parm['veh_load'] = x[nPar+2]
            # parm['rolling_coef'] = x[nPar+3]
            # parm['aero_coef'] = x[nPar+4]
            # parm['res_coef_1'] = x[nPar+5]
            # parm['res_coef_2'] = x[nPar+6]
            # parm['res_coef_3'] = x[nPar+7]
            parm['f_0'] = x[nPar+3]
            parm['f_1'] = x[nPar+4]
            parm['f_2'] = x[nPar+5]
        ###
    elif type_id == 9:
        veh_dyn_type = 'car_nonlinear'
        veh_cstr_type = 'mfc'
        parm['tau_i'] = x[nPar+1]
        if optimize_resistance_and_load == 1:
            parm['veh_load'] = x[nPar+2]
            # parm['rolling_coef'] = x[nPar+3]
            # parm['aero_coef'] = x[nPar+4]
            # parm['res_coef_1'] = x[nPar+5]
            # parm['res_coef_2'] = x[nPar+6]
            # parm['res_coef_3'] = x[nPar+7]
            parm['f_0'] = x[nPar+3]
            parm['f_1'] = x[nPar+4]
            parm['f_2'] = x[nPar+5]

    flow_ctrl = (
        control_type,       # 0
        ctrl_cstr_onoff,    # 1
        turning_effect,     # 2
        veh_dyn_type,       # 3
        veh_cstr_type       # 4
    )

    return parm, flow_ctrl


@njit(nogil=True)
def cost_function(x_gen, args):
    ModelID = args[0]
    x_lead_real_set = args[1]
    v_lead_real_set = args[2]
    xv_concat_split = args[3]
    v_des_data_set = args[4]
    t_step_data_set = args[5]
    track_data = args[6]
    track_len = args[7]
    parm_cstr = args[8]
    parm_cstr_mfc = args[9]
    CalGOFIDX = args[10]
    Precision = args[11]
    optimize_resistance_and_load = args[12]

    x = np.empty(len(x_gen), np.float64)
    for i in range(len(x_gen)):
        x[i] = utils.my_round(x_gen[i], Precision)
    parm, flow_ctrl = setup_model(x, ModelID, parm_cstr, optimize_resistance_and_load)

    # Simulation
    ObjFuncValue = 0

    gof, count_cut_acc, count_cut_dec, success, debug \
        = chained_simulation(
                x_lead_real_set,
                v_lead_real_set,
                xv_concat_split,
                v_des_data_set,
                t_step_data_set,
                track_data,
                track_len,
                flow_ctrl,
                parm_cstr_mfc,
                parm
    )
                
    if success == 0 or count_cut_acc + count_cut_dec > 0:
        ObjFuncValue = 100000

    if ObjFuncValue == 0:
        ObjFuncValue = gof[CalGOFIDX]

    return ObjFuncValue


def optimization(
    paramlist,
    ModelID,
    CalGOFIDX,  # for
    CalVehID,   # for
    CalExpIDX,  # for
    GOF_Names,
    CarMap,
    Exp_Names,
    Algo,
    pathInput, 
    pathResult,
    Precision,
    optimize_resistance_and_load
    ):

    if Algo['UseParallel']:
        CalExpIDX = paramlist[0]
        CalVehID = paramlist[1]
    ###################################
    # Setup bounds (calibration only) #
    ###################################
    bounds, LB, UB = setup_bounds(pathInput, ModelID, CalVehID, optimize_resistance_and_load)

    #######################
    # Setup cstr and data #
    #######################

    CalExp_Names = Exp_Names[CalExpIDX]

    CarID = int(CarMap.loc[CarMap[0] == CalVehID, 1].values[0]) # ID used to search car specs.

    parm_cstr, parm_cstr_mfc = setup_constraints(pathInput, CalVehID, CarID)

    x_lead_real_set, v_lead_real_set, xv_concat_split, v_des_data_set, t_step_data_set, track_len, track_data \
            = setup_data(pathInput, CalVehID, CalExp_Names)

    #########################
    # Optimization: Started #
    #########################
    startClock = time.time()
    print('Optimization: Started...')
    tempPath = os.path.join(pathResult, 'Model ' + str(ModelID), 'VehicleID ' + str(CalVehID),
                            'Calibration_Temp_Exp_' + str(CalExpIDX+1) + '_' + GOF_Names[CalGOFIDX])
    tempStatus = ' ModelID=' + str(ModelID) + ', CalGOFID=' + str(CalGOFIDX+1) \
                    + ', CalVehID=' + str(CalVehID) + ', CalExpID=' + str(CalExpIDX+1)

    algorithm_param = {'max_num_iteration': Algo['MaxIteration'],
                        'population_size': Algo['PopulationSize'],
                        'mutation_probability': 0.1,
                        'elit_ratio': 0.01,
                        'crossover_probability': 0.5,
                        'parents_portion': 0.3,
                        'crossover_type': 'uniform',
                        'max_iteration_without_improv': int(0.8*Algo['MaxIteration']),
                        'tempPath': tempPath,
                        'tempStatus': tempStatus}
    varbound = np.column_stack((LB, UB))
    model = ga(cost_function, dimension=len(LB), \
        variable_boundaries=varbound,
        variable_type='real', 
        function_timeout=200,
        algorithm_parameters=algorithm_param,
        args=(
            ModelID,            # 0
            x_lead_real_set,    # 1
            v_lead_real_set,    # 2
            xv_concat_split,    # 3
            v_des_data_set,     # 4
            t_step_data_set,    # 5
            track_data,         # 6
            track_len,          # 7
            parm_cstr,          # 8
            parm_cstr_mfc,      # 9
            CalGOFIDX,          # 10
            Precision,          # 11
            optimize_resistance_and_load    # 12
        )   
    )

    fval_temp = 100000.0
    max_run = 0
    while fval_temp > 15.0 - 1e-5 and max_run < 10:
        print('Optimization run = ' + str(max_run+1))
        model.run()
        fval_temp = model.output_dict['function']
        val_temp = model.output_dict['variable']
        max_run += 1
        #################################
        # Remove temp folders and files #
        #################################
        if os.path.exists(tempPath) and True:
            shutil.rmtree(tempPath)

    val = []
    for i in range(len(val_temp)):
        val.append(utils.my_round(val_temp[i], Precision))
    del i
    fval = utils.my_round(fval_temp, Precision)

    print('Optimization: Completed.')   
    endClock = (time.time() - startClock)

    #####################################
    # Report calidation log and results #
    #####################################
    log_dict = {
        'ModelID': ModelID,
        'CalVehID': CalVehID,
        'CalGOFID': CalGOFIDX+1,
        'CalExpID': CalExpIDX+1,
        'Duration (min)': round(endClock/60),
        'DateTime': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }

    with open(os.path.join(pathResult, 'log.txt'), 'a') as f:
        print(log_dict, file=f)
    
    CalReport = os.path.join(pathResult, 'Model '+str(ModelID), 'VehicleID ' + str(CalVehID),
                        'Calibration_Report_Exp_' + str(CalExpIDX+1) + '_' + GOF_Names[CalGOFIDX] + '.txt')
    fid = open(CalReport, "w")
    for v in val:
        fid.write(str(v) + '\t')
    fid.write(str(fval) + '\n')
    fid.close()
    del v

    return val, fval


def validation(
    paramlist,
    ModelID,
    CalGOFIDX,  # for
    CalVehID,   # for
    CalExpIDX,  # for
    GOF_Names,
    CarMap,
    Exp_Names,
    Algo,
    pathInput, 
    pathResult,
    Precision,
    optimize_resistance_and_load
    ):

    if Algo['UseParallel']:
        CalExpIDX = paramlist[0]
        CalVehID = paramlist[1]
    #############################
    # Report validation results #
    #############################
    ValReport = os.path.join(pathResult, 'Model ' + str(ModelID), 'VehicleID ' + str(CalVehID),
                        'Validation_Report_CalExp_' + str(CalExpIDX+1) + '_' + GOF_Names[CalGOFIDX] + '.txt')
    fid = open(ValReport, 'w')

    ######################################################
    # Setup parameters (validation only), cstr, and data #
    ######################################################
    CarID = int(CarMap.loc[CarMap[0] == CalVehID, 1].values[0]) # ID used to search car specs.

    parm_cstr, parm_cstr_mfc = setup_constraints(pathInput, CalVehID, CarID)

    x = read_calibrated_parms(pathResult, ModelID, CalVehID, CalExpIDX, GOF_Names[CalGOFIDX])
    parm, flow_ctrl = setup_model(x, ModelID, parm_cstr, optimize_resistance_and_load)
    
    for ValExpIDX in range(len(Exp_Names)):
        ValExp_Names = Exp_Names[ValExpIDX]

        x_lead_real_set, v_lead_real_set, xv_concat_split, v_des_data_set, t_step_data_set, track_len, track_data \
                = setup_data(pathInput, CalVehID, ValExp_Names)

        gof, count_cut_acc, count_cut_dec, success, debug \
            = chained_simulation(
                    x_lead_real_set,
                    v_lead_real_set,
                    xv_concat_split,
                    v_des_data_set,
                    t_step_data_set,
                    track_data,
                    track_len,
                    flow_ctrl,
                    parm_cstr_mfc,
                    parm
        )
        fid.write('ValExp_'+str(ValExpIDX+1) + '\t')
        for k in range(len(gof)):
            fid.write(str(utils.my_round(gof[k], 5)) + '\t')
        fid.write(str(count_cut_acc) + '\t' + str(count_cut_dec) + '\n')

        if True:
            fileName = 'Model_' + str(ModelID) + '_Vehicle_' + str(CalVehID) \
                + '_CalExp_' + str(CalExpIDX+1) + '_ValExp_' + str(ValExpIDX+1)

            pathExport = os.path.join(pathResult, 'Export')
            
            if not os.path.exists(pathExport):
                os.makedirs(pathExport, exist_ok=True)

            export_time_series(
                x,
                GOF_Names[CalGOFIDX]+': '+str(utils.my_round(gof[CalGOFIDX], 3))+' ',
                debug,
                fileName,
                pathExport,
                exportData=False,
                exportFig=True,
                htmlFig=False)

    fid.close()


@njit(nogil=True)
def chained_simulation(
    x_lead_real_set,
    v_lead_real_set,
    xv_concat_split,
    v_des_data_set,
    t_step_data_set,
    track_data,
    track_len,
    flow_ctrl,
    parm_cstr_mfc,
    parm
    ):

    parm['track_len'] = track_len

    for i in range(len(t_step_data_set)):
        parm['dt'] = t_step_data_set[i]
        # parm['v_set'] = v_des_data_set[i]
        x_lead_real = x_lead_real_set[xv_concat_split[i]:xv_concat_split[i+1]]
        v_lead_real = v_lead_real_set[xv_concat_split[i]:xv_concat_split[i+1]]
        time_real = np.array([utils.my_round(parm['dt'] * k, 1) for k in range(len(x_lead_real))])
        parm['v_set'] = np.median(v_lead_real)
        
        exp_data = np.vstack((
            time_real,
            x_lead_real, # 1
            v_lead_real, # 2
        ))
        gof_, count_cut_acc_, count_cut_dec_, success, comfort, debug_ \
            = ff_env.run(parm, flow_ctrl, parm_cstr_mfc, exp_data, track_data)

        if i == 0:
            gof = gof_
            count_cut_acc = count_cut_acc_
            count_cut_dec = count_cut_dec_
            debug = debug_
        else:
            gof += gof_
            count_cut_acc += count_cut_acc_
            count_cut_dec += count_cut_dec_
            debug = np.hstack((debug, debug_))

        if success == 0:
            break

    return gof, count_cut_acc, count_cut_dec, success, debug


def export_time_series(
    x,
    gofText,
    debug,
    fileName,
    pathExport,
    exportData=False,
    exportFig=False,
    htmlFig=False
    ):
    ###################################################
    # Export time series of each simulation for debug #
    ###################################################
    time_concat = np.array([utils.my_round(0.1*k, 1) for k in range(len(debug[0]))])

    res_ ={
        'time': debug[0],
        'exp_x': debug[1],
        'exp_v': debug[2],
        'exp_a': debug[3],
        'exp_v_err': debug[4],
        'sim_x': debug[5],
        'sim_v': debug[6],
        'sim_a': debug[7],
        'sim_v_err': debug[8],
        'sim_sin': debug[9],
        'sim_curv': debug[10],
    }
    
    if exportData == True:
        df_res = pd.DataFrame.from_dict(res_)
        df_res.to_csv(os.path.join(pathExport, fileName+'.csv'), index=False)
        ######################################################
        # Plot and save net spacing, speed, and acceleration #
        ######################################################
    if exportFig == True:
        fig = make_subplots(rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.02,
                            specs=[[{'secondary_y': True}], [{}], [{}]])

        idx_a = [9, 10]
        name_a = ['sim_sin', 'sim_curv']
        sec_y_a = [False, True]
        
        for k in range(len(idx_a)):
            fig.add_trace(go.Scatter(
                x=time_concat,
                y=debug[idx_a[k]],
                name=name_a[k],
            ), row=1, col=1, secondary_y=sec_y_a[k])

        idx_b = [2, 6]
        name_b = ['exp_v', 'sim_v']
        for k in range(len(idx_b)):
            fig.add_trace(go.Scatter(
                x=time_concat,
                y=debug[idx_b[k]],
                name=name_b[k],
            ), row=2, col=1)

        idx_c = [3, 7]
        name_c = ['exp_a', 'sim_a']
        for k in range(len(idx_c)):
            fig.add_trace(go.Scatter(
                x=time_concat,
                y=debug[idx_c[k]],
                name=name_c[k]
            ), row=3, col=1)
        
        fig.update_xaxes(title_text='Time [s]', row=3, col=1)
        fig.update_yaxes(title_text='Sim_SinSlope', row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text='Sim_Curvature', row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Speed [m/s]', row=2, col=1)
        fig.update_yaxes(title_text='Acceleration [m/s2]', row=3, col=1)

        if htmlFig:
            fig.update_layout(title_text=fileName+' '+gofText+'<br>'+str([ '%.1f' % elem for elem in x ]))
            plot(fig, filename=os.path.join(pathExport, fileName+'.html'), auto_open=False)
        else:
            fig.update_layout(title_text=fileName+' '+gofText+'<br>'+str([ '%.1f' % elem for elem in x ]), autosize=False, width=900, height=600)
            fig.write_image(os.path.join(pathExport, fileName+'.png'))
