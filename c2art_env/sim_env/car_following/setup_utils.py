import os, math, time, datetime, shutil
import numpy as np
from numba import njit, types 
from numba.typed import Dict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import c2art_env.sim_env.car_following.car_following_env_njit as cf_env
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

    DelayClass = ModelID % 18
    if DelayClass == 0 or DelayClass >= 10:
        bounds, LB, UB = read_bounds(os.path.join(pathInput, 'Bounds', \
            'perception_delay.txt'), bounds, LB, UB)

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
    x_foll_real_set = [None]*len(SL_Exp_Names)
    v_lead_real_set = [None]*len(SL_Exp_Names)
    v_foll_real_set = [None]*len(SL_Exp_Names)
    length_lead_set = [None]*len(SL_Exp_Names)
    t_step_data_set = [None]*len(SL_Exp_Names)

    for i in range(len(SL_Exp_Names)):
        x_lead_real_set[i], x_foll_real_set[i], v_lead_real_set[i], v_foll_real_set[i], \
            length_lead_set[i], t_step_data_set[i] = import_trajectory_data(pathInput, VehID, SL_Exp_Names[i])
    del i

    track_data = pd.read_csv(os.path.join(pathInput, 'Slope.csv')).to_dict(orient='list')
    track_len = track_data['X'][-1]
    track_data = np.array([
        track_data['X'],
        track_data['Y']
        ])
    return np.array(x_lead_real_set), np.array(x_foll_real_set), np.array(v_lead_real_set), \
        np.array(v_foll_real_set), np.array(length_lead_set), np.array(t_step_data_set), \
            track_len, track_data


def import_trajectory_data(pathInput, VehID, ExpName):

    labelsData = pd.read_csv(os.path.join(pathInput, 'PyData', 'Labels.csv'), index_col=None)
    experimentData = pd.read_csv(os.path.join(pathInput, 'PyData', ExpName+'.csv'), index_col=None)
    # Sample rate
    t_step_data = experimentData['Time'][1] - experimentData['Time'][0]

    # Leader and follower vehicle index
    labelsDataExp = labelsData.loc[labelsData['File'] == ExpName]
    for i in range(1, 6):
        if labelsDataExp['type'+str(i)].values[0] == VehID:
            follIndex = i
            leadIndex = i-1
            length_lead = labelsDataExp['length'+str(leadIndex)].values[0]
            break

    K = experimentData.shape[0]

    v_real = experimentData[['SpeedDoppler'+str(leadIndex), 'SpeedDoppler'+str(follIndex)]].T.values.tolist()
    x_real = [[None]*K, [None]*K]
    for k in range(K):
        if k == 0:
            x_real[0][k] = experimentData['X'+str(leadIndex)][0]
            x_real[1][k] = experimentData['X'+str(follIndex)][0]
        else:
            x_real[0][k] = x_real[0][k-1] + ((v_real[0][k-1]+v_real[0][k])/2) * t_step_data
            x_real[1][k] = x_real[1][k-1] + ((v_real[1][k-1]+v_real[1][k])/2) * t_step_data

    return x_real[0], x_real[1], v_real[0], v_real[1], length_lead, t_step_data


@njit
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
        control_type = 'acc_idm'
        spacing_type = 'd_idm_des'
        parm['delta'] = x[0]
        parm['t_h'] = x[1]
        parm['v_set'] = x[2]
        parm['acc_a_max'] = x[3]
        parm['acc_a_min'] = -x[4]
        parm['d_0'] = x[5]
        nPar = 5
    elif 19 <= ModelID <= 36:
        control_type = 'acc_gipps'
        spacing_type = 'd_none'
        parm['t_h'] = x[0]
        parm['teta'] = x[1]
        parm['acc_a_max'] = x[2]
        parm['v_set'] = x[3]
        parm['acc_a_min'] = -x[4]
        parm['est_a_min_p'] = -x[5]
        parm['d_0'] = x[6]
        nPar = 6
    elif 37 <= ModelID <= 54:
        control_type = 'acc_linear'
        spacing_type = 'd_cth'
        parm['k_p'] = x[0]
        parm['k_v'] = x[1]
        parm['k_a'] = x[2]
        parm['k_set'] = x[3]
        parm['d_0'] = x[4]
        parm['t_h'] = x[5]
        parm['v_set'] = x[6]
        # parm['acc_a_max'] = x[7]
        # parm['acc_a_min'] = -x[8]
        # nPar = 8  
        nPar = 6
    elif 55 <= ModelID <= 72:
        control_type = 'acc_linear'
        spacing_type = 'd_idm_des'
        parm['k_p'] = x[0]
        parm['k_v'] = x[1]
        parm['k_a'] = x[2]
        parm['k_set'] = x[3]
        parm['d_0'] = x[4]
        parm['t_h'] = x[5]
        parm['v_set'] = x[6]
        parm['acc_a_max'] = x[7]
        parm['acc_a_min'] = -x[8]
        nPar = 8
    elif 73 <= ModelID <= 90:
        control_type = 'acc_linear'
        spacing_type = 'd_gipps_eq'
        parm['k_p'] = x[0]
        parm['k_v'] = x[1]
        parm['k_a'] = x[2]
        parm['k_set'] = x[3]
        parm['d_0'] = x[4]
        parm['t_h'] = x[5]
        parm['v_set'] = x[6]
        parm['acc_a_min'] = -x[7]
        parm['teta'] = x[8]
        parm['est_a_min_p'] = -x[9]
        nPar = 9
    ###

    delay_class_id = ModelID % 18
    if delay_class_id == 0:
        delay_class_id = 18
    ###
    if delay_class_id >= 10:
        parm['perception_delay'] = x[nPar+1]
        nPar = nPar + 1
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
        spacing_type,       # 1  
        ctrl_cstr_onoff,    # 2            
        veh_dyn_type,       # 3
        veh_cstr_type       # 4        
    )

    return parm, flow_ctrl


@njit
def cost_function(x_gen, args):
    ModelID = args[0]
    x_lead_real_set = args[1]
    x_foll_real_set = args[2]
    v_lead_real_set = args[3]
    v_foll_real_set = args[4]
    length_lead_set = args[5]
    t_step_data_set = args[6]
    track_data = args[7]
    track_len = args[8]
    parm_cstr = args[9]
    parm_cstr_mfc = args[10]
    CalGOFIDX = args[11]
    Precision = args[12]
    optimize_resistance_and_load = args[13]

    x = np.empty(len(x_gen), np.float64)
    for i in range(len(x_gen)):
        x[i] = utils.my_round(x_gen[i], Precision)
    parm, flow_ctrl = setup_model(x, ModelID, parm_cstr, optimize_resistance_and_load)

    # Simulation
    ObjFuncValue = 0

    gof, count_cut_acc, count_cut_dec, success, comfort, terminal, \
    errors_d, errors_v, errors_a, debug = chained_simulation(
                x_lead_real_set,
                x_foll_real_set,
                v_lead_real_set,
                v_foll_real_set,
                length_lead_set,
                t_step_data_set,
                track_data,
                track_len,
                flow_ctrl,
                parm_cstr_mfc,
                parm)
                
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

    CalExp_Names = [Exp_Names[CalExpIDX]]

    CarID = int(CarMap.loc[CarMap[0] == CalVehID, 1].values) # ID used to search car specs.

    parm_cstr, parm_cstr_mfc = setup_constraints(pathInput, CalVehID, CarID)

    x_lead_real_set, x_foll_real_set, v_lead_real_set, v_foll_real_set, \
        length_lead_set, t_step_data_set, track_len, track_data \
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
        function_timeout=100,
        algorithm_parameters=algorithm_param,
        args=(
            ModelID,
            np.array(x_lead_real_set),
            np.array(x_foll_real_set),
            np.array(v_lead_real_set),
            np.array(v_foll_real_set),
            np.array(length_lead_set),
            np.array(t_step_data_set),
            track_data,
            track_len,
            parm_cstr,
            parm_cstr_mfc,
            CalGOFIDX,
            Precision,
            optimize_resistance_and_load)
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
                        'Validation_Report_CalExp_' + str(CalExpIDX + 1) + '_' + GOF_Names[CalGOFIDX] + '.txt')
    fid = open(ValReport, 'w')

    ######################################################
    # Setup parameters (validation only), cstr, and data #
    ######################################################
    CarID = int(CarMap.loc[CarMap[0] == CalVehID, 1].values) # ID used to search car specs.

    parm_cstr, parm_cstr_mfc = setup_constraints(pathInput, CalVehID, CarID)

    x = read_calibrated_parms(pathResult, ModelID, CalVehID, CalExpIDX, GOF_Names[CalGOFIDX])
    parm, flow_ctrl = setup_model(x, ModelID, parm_cstr, optimize_resistance_and_load)
    
    for ValExpIDX in range(len(Exp_Names)):
        ValExp_Names = [Exp_Names[ValExpIDX]]

        x_lead_real_set, x_foll_real_set, v_lead_real_set, v_foll_real_set, \
            length_lead_set, t_step_data_set, track_len, track_data \
                = setup_data(pathInput, CalVehID, ValExp_Names)

        gof, count_cut_acc, count_cut_dec, success, comfort, terminal, \
        errors_d, errors_v, errors_a, debug = chained_simulation(
                    x_lead_real_set,
                    x_foll_real_set,
                    v_lead_real_set,
                    v_foll_real_set,
                    length_lead_set,
                    t_step_data_set,
                    track_data,
                    track_len,
                    flow_ctrl,
                    parm_cstr_mfc,
                    parm)
        fid.write(Exp_Names[ValExpIDX] + '\t')
        for k in range(len(gof)):
            fid.write(str(utils.my_round(gof[k], 5)) + '\t')
        fid.write(str(count_cut_acc) + '\t' + str(count_cut_dec) + '\n')

        if False:
            print('Model_' + str(ModelID) + '_VehID_'+str(CalVehID) \
                + '_CalExpID_' + str(CalExpIDX+1) + '_ValExpID_' + str(ValExpIDX+1) \
                + '_NoCrash_' + str(success))

        if success == 0 or comfort == 0 or CalExpIDX == ValExpIDX:
            fileName = 'Model_' + str(ModelID) + '_Vehicle_' + str(CalVehID) \
                + '_CalExp_' + str(CalExpIDX+1) + '_ValExp_' + str(ValExpIDX+1) \
                + '_NoCrash_' + str(success) + '_NoAccOsc_' + str(comfort)
            # NoAccOsc and NoTrmDev don't make sense when NoCrash=0

            # pathExport = os.path.join(pathResult, 'Model '+str(ModelID),\
            #     'VehicleID '+str(CalVehID), 'Export')
            pathExport = os.path.join(pathResult, 'Export')
            
            if not os.path.exists(pathExport):
                os.makedirs(pathExport, exist_ok=True)

            export_time_series(
                debug,
                fileName,
                pathExport,
                exportData=False,
                exportFig=True,
                htmlFig=False)


@njit
def chained_simulation(
    x_lead_real_set,
    x_foll_real_set,
    v_lead_real_set,
    v_foll_real_set,
    length_lead_set,
    t_step_data_set,
    track_data,
    track_len,
    flow_ctrl,
    parm_cstr_mfc,
    parm
    ):

    parm['track_len'] = track_len

    for i in range(len(length_lead_set)):
        parm['l_i_1'] = length_lead_set[i]
        parm['dt'] = t_step_data_set[i]
        time_real = np.zeros_like(x_lead_real_set[i])
        for k in range(len(time_real)):
            time_real[k] = utils.my_round(parm['dt'] * k, 1)
        exp_data = np.vstack((
            time_real,
            x_lead_real_set[i], # 1
            x_foll_real_set[i], # 2
            v_lead_real_set[i], # 3
            v_foll_real_set[i], # 4
        ))
        gof, count_cut_acc, count_cut_dec, success, comfort, terminal, \
        errors_d, errors_v, errors_a, debug = cf_env.run(\
            parm, flow_ctrl, parm_cstr_mfc, exp_data, track_data)

        if success == 0:
            break

    ### The function here returns the results of a single leader-follower pair simulation.
    ### If there are multiple car-following pair in the real_set (chained), results processing is needed before return. 
    
    return gof, count_cut_acc, count_cut_dec, success, comfort, terminal, \
        errors_d, errors_v, errors_a, debug


def export_time_series(
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
    res_ ={
        'time': debug[0],
        'exp_i_1_p': debug[1],
        'exp_i_1_v': debug[2],
        'exp_i_1_a': debug[3],
        'exp_i_p': debug[4],
        'exp_i_v': debug[5],
        'exp_i_a': debug[6],
        'exp_d': debug[7],
        'sim_i_p': debug[8],
        'sim_i_v': debug[9],
        'sim_i_a': debug[10],
        'sim_d': debug[11],
        'sim_sin': debug[12],
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
                            vertical_spacing=0.02)
        name_ = ['exp_p', 'exp_f', 'sim_f']
        s_, v_, a_ = [0, 7, 11], [2, 5, 9], [3, 6, 10]
        for k in range(3):
            if k > 0:
                fig.add_trace(go.Scatter(x=debug[0], y=debug[s_[k]], 
                    name='Spacing_' + name_[k] + ' (min: ' + str(round(min(debug[s_[k]]), 3)) + ' m)'),
                    row=1, col=1)
        for k in range(3):
            fig.add_trace(go.Scatter(x=debug[0], y=debug[v_[k]],
                    name='Speed_' + name_[k] + ' (min: ' + str(round(min(debug[v_[k]]), 1)) + ' m/s)'), 
                    row=2, col=1)
        for k in range(3):
            fig.add_trace(go.Scatter(x=debug[0], y=debug[a_[k]], 
                    name='Accel_' + name_[k] + ' (min-max: ' + str(round(min(debug[a_[k]]), 1)) + ', ' + str(round(max(debug[a_[k]]), 1)) + ' m/s2)'), 
                    row=3, col=1)
        
        fig.update_xaxes(title_text='Time [s]', row=3, col=1)
        fig.update_yaxes(title_text='Spacing [m]', row=1, col=1)
        fig.update_yaxes(title_text='Speed [m/s]', row=2, col=1)
        fig.update_yaxes(title_text='Acceleration [m/s2]', row=3, col=1)

        if htmlFig:
            fig.update_layout(title_text=fileName)
            plot(fig, filename=os.path.join(pathExport, fileName+'.html'), auto_open=False)
        else:
            fig.update_layout(title_text=fileName, autosize=False, width=900, height=600)
            fig.write_image(os.path.join(pathExport, fileName+'.png'))
