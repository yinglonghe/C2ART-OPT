import os, re
import shutil
import numpy as np
import pandas as pd
from itertools import product
from tqdm.contrib.itertools import product as product_tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from PostProcessing.AstaZero_CF_CAL.Scripts.c2art_res import utils as utils_cf


def create_res_csv(
    srcResLite,
    pathResLite,
    expLabel,
    pathParmKeys,
    CalGOF_IDs, 
    Model_IDs, 
    Veh_IDs, 
    Exp_IDs,
    GOF_Names,
    Exp_Names
    ):
    
    ######################################
    # Get parameter parameters of models #
    ######################################
    parmKeys = utils_cf.read_parm_keys(pathParmKeys)
    #######################
    # Store data and info #
    #######################
    srcRes  = []
    idModel, idVehicle, idExpCal, idGOFCal, setSpdCal = [], [], [], [], []
    accControl, turningEffect, vehicleDynamics, vehicleConstraint = [], [], [], []
    CalParm, CalParmNames, CalGOF, ValGOFs = [], [], [], []

    for imod, iveh, iexc, igof in product_tqdm(Model_IDs, Veh_IDs, Exp_IDs, CalGOF_IDs):
        ########################################################
        # Get IDs of model, vehicle, platoon, and gof and etc. #
        ########################################################
        idModel.append(imod)
        idVehicle.append(iveh)
        idExpCal.append(iexc)
        idGOFCal.append(igof)
        setSpdCal.append(expLabel['setSpd_kmph'].tolist()[iexc-1])
        srcRes.append(srcResLite)
        #######################################
        # Get model configuration information #
        #######################################
        if 1 <= imod <= 18:
            accControl.append('acc_pid_free')
        # elif 19 <= imod <= 36:
        #     accControl.append('acc_gipps')
        
        turning_class_idx = imod % 18
        if turning_class_idx == 0:
            turning_class_idx = 18
        if turning_class_idx >= 10:
            turningEffect.append('yes')
        else:
            turningEffect.append('no')
        del turning_class_idx

        veh_type_idx = imod % 9
        if veh_type_idx == 0:
            veh_type_idx = 9
        if veh_type_idx == 1:
            vehicleDynamics.append('car_none')
            vehicleConstraint.append('off')
        elif veh_type_idx == 2:
            vehicleDynamics.append('car_none')
            vehicleConstraint.append('constant')
        elif veh_type_idx == 3:
            vehicleDynamics.append('car_none')
            vehicleConstraint.append('mfc')
        elif veh_type_idx == 4:
            vehicleDynamics.append('car_linear')
            vehicleConstraint.append('off')
        elif veh_type_idx == 5:
            vehicleDynamics.append('car_linear')
            vehicleConstraint.append('constant')
        elif veh_type_idx == 6:
            vehicleDynamics.append('car_linear')
            vehicleConstraint.append('mfc')
        elif veh_type_idx == 7:
            vehicleDynamics.append('car_nonlinear')
            vehicleConstraint.append('off')
        elif veh_type_idx == 8:
            vehicleDynamics.append('car_nonlinear')
            vehicleConstraint.append('constant')
        elif veh_type_idx == 9:
            vehicleDynamics.append('car_nonlinear')
            vehicleConstraint.append('mfc')
        del veh_type_idx
        #####################################################
        # Get results of calibration GOF (1) and parameters #
        #####################################################
        with open(os.path.join(pathResLite, 
                'Model ' + str(imod), 
                'VehicleID ' + str(iveh),
                'Calibration_Report_Exp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')) as f:
            for line in f:
                line = line.strip()
                x_temp = [round(float(x), 5) for x in line.split('\t')]
        del f, line
        CalParm.append(x_temp[:-1])
        CalGOF.append(x_temp[-1])
        CalParmNames.append(parmKeys['VarNames'][imod-1])
        ######################################
        # Get results of validation GOFs (8) #
        ######################################
        val_gof_temp = []

        with open(os.path.join(pathResLite, 
                'Model ' + str(imod), 
                'VehicleID ' + str(iveh),
                'Validation_Report_CalExp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')) as f:
            for line in f:
                line = line.strip()
                line_list = [x for x in line.split('\t')[1:]]
                # For MATLAB results, there is a missing element in the validation GOF list when success==0
                if len(line_list) == len(GOF_Names) + 2:             
                    x_temp = [round(float(x), 5) for x in line.split('\t')[1:-2]]
                else:
                    x_temp = [round(float(x), 5) for x in line.split('\t')[1:-2] + [100000.00000]]
                val_gof_temp += x_temp
        
        ValGOFs.append(val_gof_temp)
        del x_temp, f, line, line_list, val_gof_temp
        
    del imod, iveh, iexc, igof

    col_ = ['idModel', 'idVehicle', 'idExpCal', 
            'accControl', 'turningEffect', 'vehicleDynamics', 'vehicleConstraint',
            'setSpdCal', 'idGOFCal', 'srcRes', 
            'Cal_' + GOF_Names[CalGOF_IDs[0]-1], 'Cal_Parm', 'Parm_Names']
    for iexv, name_gof in product(Exp_IDs, GOF_Names):
        col_.append('ValExp' + str(iexv) + '_' + name_gof)

    cal_list = list(map(list, zip(*[idModel, idVehicle, idExpCal, 
                                    accControl, turningEffect, vehicleDynamics, vehicleConstraint,
                                    setSpdCal, idGOFCal, srcRes, 
                                    CalGOF, CalParm, CalParmNames])))
    cal_val_list = [cal_list[idx] + val for idx, val in enumerate(ValGOFs)]

    df = pd.DataFrame(data=cal_val_list, columns=col_)
    print('Convert *.txt in ' + srcResLite + ' into ' + srcResLite + '.csv in Outputs.')

    return df


def organize_cal_res(
    df_,
    Veh_IDs,
    Exp_IDs,
    GOF_Names,
    CalGOF_IDs
    ):
    GOFName = GOF_Names[CalGOF_IDs[0]-1]
    df = df_[['idModel', 'idExpCal', 'idVehicle', 'setSpdCal', 'Cal_'+GOFName]]

    idExpCal = []
    idVehicle = []
    setSpdCal = []
    for iveh, iexc in product_tqdm(Veh_IDs, Exp_IDs):
        df_temp = df.loc[(df['idExpCal'] == iexc) & (df['idVehicle'] == iveh), ['idModel', 'Cal_'+GOFName]].sort_values(by=['idModel'])
        df_temp.columns = ['idModel', 'VehId' + str(iveh) + '_Exp' + str(iexc)]
        try:
            df_cal = pd.merge(df_cal, df_temp, on='idModel', how='left')
        except:
            df_cal = df_temp.copy()

        idVehicle.append(iveh)
        idExpCal.append(iexc)
        setSpdCal.append(int(df.loc[(df['idExpCal'] == iexc) & (df['idVehicle'] == iveh), 'setSpdCal'].unique()))

    df_cal.set_index('idModel', inplace=True)
    df_cal = df_cal.T.add_prefix('Model_')

    df_cal.insert(loc=0, column='idVehicle', value=idVehicle)
    df_cal.insert(loc=1, column='idExpCal', value=idExpCal)
    df_cal.insert(loc=2, column='setSpdCal', value=setSpdCal)
    df_cal.insert(loc=3, column='Model_MIN', value=df_cal.iloc[:,3:].min(axis=1))
    df_cal.insert(loc=4, column='Model_MAX', value=df_cal.iloc[:,4:].max(axis=1))
    
    return df_cal


def organize_val_res_all_gof(
    df,
    Model_IDs,
    Veh_IDs,
    Exp_IDs,
    GOF_Names
    ):
    col_gofs=[]
    for iexv in Exp_IDs: 
        col_gofs.append([col for col in df.columns if 'ValExp'+str(iexv) in col])
    del iexv

    idModel = []
    idVehicle = []
    idExpCal = []
    idExpVal = []
    setSpdCal = []
    setSpdVal = []
    gof_list = []
    entire_val_list = []

    for imod, iveh, iexc, iexv in product_tqdm(Model_IDs, Veh_IDs, Exp_IDs, Exp_IDs):
        idModel.append(imod)
        idVehicle.append(iveh)
        idExpCal.append(iexc)
        idExpVal.append(iexv)
        setSpdCal.append(int(df.loc[(df['idExpCal'] == iexc) & (df['idVehicle'] == iveh), 'setSpdCal'].unique()))
        setSpdVal.append(int(df.loc[(df['idExpCal'] == iexv) & (df['idVehicle'] == iveh), 'setSpdCal'].unique()))

        gof_list.append(df.loc[(df['idModel'] == imod) & (df['idVehicle'] == iveh) & (df['idExpCal'] == iexc), col_gofs[iexv-1]].values[0].tolist())

    entire_val_list = [[idModel[i], idVehicle[i], idExpCal[i], idExpVal[i], setSpdCal[i], setSpdVal[i]] + val for i, val in enumerate(gof_list)]

    df_val_all_gofs = pd.DataFrame(entire_val_list, columns = ['idModel', 'idVehicle', 'idExpCal', 'idExpVal', 'setSpdCal', 'setSpdVal'] + GOF_Names)
    
    df_val_all_gofs.replace(100000, np.NaN, inplace=True)

    return df_val_all_gofs


def organize_val_res_single_gof(
    df_all_gofs,
    Model_IDs,
    ValGOFName
    ):
    for i in Model_IDs:
        df = df_all_gofs.loc[df_all_gofs['idModel']==i].sort_values(by=['idVehicle', 'idExpCal', 'idExpVal'])
        try:
            df_all['Model_'+str(i)] = df[ValGOFName].reset_index(drop=True)
        except:
            df_all = df[['idVehicle', 'idExpCal', 'idExpVal', 'setSpdCal', 'setSpdVal']].copy()
            df_all['Model_'+str(i)] = df[ValGOFName]

    df_all.insert(loc=5, column='Model_MIN', value=df_all.iloc[:,5:].min(axis=1))
    df_all.insert(loc=6, column='Model_MAX', value=df_all.iloc[:,6:].max(axis=1))
    df_all.insert(loc=7, column='CNT_BLK', value=df_all.iloc[:,7:].isnull().sum(axis=1))

    df_val = df_all.loc[df_all['idExpCal']!=df_all['idExpVal']]
    df_val = df_val.reset_index(drop=True)

    df_cal = df_all.loc[df_all['idExpCal']==df_all['idExpVal']]
    df_cal = df_cal.reset_index(drop=True)

    return df_val, df_cal

