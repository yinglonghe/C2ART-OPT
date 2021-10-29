# %%
import os, sys, glob
rootDir = os.path.abspath(__file__ + "/../../../../")
sys.path.append(rootDir)

import numpy as np
import pandas as pd
from IPython.core import display as ICD
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from PostProcessing.AstaZero_CF_CAL.Scripts.c2art_res import utils as utils_cf
from PostProcessing.FreeFlow_Turning_CF.Scripts.c2art_res import utils as utils_ff

# %% # Provide global parameters

CalGOF_IDs = [7]
Project = ['ASTAZERO', 'ZalaZone_HC'][1]
Database = 'ExpFreeFlow'
Precision = 5       # Decimal

path = {
    'Root': rootDir,
    'ResRoot': os.path.abspath(os.path.join(__file__ ,'../..'))
}

path['ParmKeys'] = os.path.join(path['Root'], 'Data', Project, Database, 'ParmKeys.txt')

expLabel = pd.read_csv(os.path.join(path['Root'], 'Data', Project, Database, 'PyData', 'FreeFlowStableParts.csv'))
Exp_Names = [expLabel['File'][i]+'_'+str(expLabel['timeStart'][i])+'_'+str(expLabel['timeEnd'][i])+'_V'+str(expLabel['setSpd_kmph'][i]) for i in range(len(expLabel))]

Veh_Names = []

GOF_Names = [
    'RMSE(V)',      # GOF-1 (ID)
    'RMSE(A)',      # GOF-2
    'RMSPE(V)',     # GOF-3
    'RMSPE(std_V)', # GOF-4
    'NRMSE(V)',     # GOF-5
    'NRMSE(A)',     # GOF-6
    'NRMSE(V+A)',   # GOF-7
    'U(V)',         # GOF-8
    'U(A)',         # GOF-9
    'U(V+A)',       # GOF-10
    'RMSE(Vn)',     # GOF-11
    'RMSE(An)',     # GOF-12
    'RMSE(Vn+An)',  # GOF-13
]

Submodel_IDs = {
    'PID': np.arange(1, 19),
    # 'Gipps': np.arange(19, 37),
}
Model_IDs = []
for key, value in Submodel_IDs.items():
    Model_IDs += list(value)
del key, value

Veh_IDs = [1]
Exp_IDs = list(np.arange(len(Exp_Names))+1)

path['ResultsFull'] = os.path.join(path['ResRoot'], 'ResultsFull', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['ResultsLite'] = os.path.join(path['ResRoot'], 'ResultsLite', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['Outputs'] = os.path.join(path['ResRoot'], 'Outputs', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1], Project, Database)


# %% # .\ResultsFull\...\*.txt >>> .\ResultsLite\...\*.txt

if True:
    srcResFull = utils_cf.get_immediate_subdirectories(path['ResultsFull'])
    srcResFull = [srcRes for srcRes in srcResFull if 'PyResults' in srcRes]
    pathResFull = [os.path.join(path['ResultsFull'], \
        srcResFull[i], Project, Database) for i in range(len(srcResFull))]

    pathResLite = [os.path.join(path['ResultsLite'], \
        srcResFull[i], Project, Database) for i in range(len(srcResFull))]
    
    assert (len(pathResFull) == len(pathResLite)), \
        '\n The number of source directories must be equal to that of target directories.'
    for i in range(len(pathResFull)):
        utils_cf.remove_redundant_files(
            pathResFull[i],
            pathResLite[i],
            CalGOF_IDs,
            Model_IDs,
            Veh_IDs,
            Exp_IDs,
            GOF_Names)
    del i, pathResFull, pathResLite, srcResFull

# %% # .\ResultsLite\...\*.txt >>> .\Outputs\...\*.csv

if True:
    srcResLite = utils_cf.get_immediate_subdirectories(path['ResultsLite'])
    srcResLite = [srcRes for srcRes in srcResLite if 'PyResults' in srcRes]
    pathResLite = [os.path.join(path['ResultsLite'], \
        srcResLite[i], Project, Database) for i in range(len(srcResLite))]

    for i in range(len(pathResLite)):
        pathOutput = os.path.join(path['Outputs'], srcResLite[i])
        os.makedirs(pathOutput, exist_ok=True)
        df_ = utils_ff.create_res_csv(
            srcResLite[i],
            pathResLite[i],
            expLabel,
            path['ParmKeys'],
            CalGOF_IDs,
            Model_IDs,
            Veh_IDs, 
            Exp_IDs,
            GOF_Names,
            Exp_Names)
        df_.to_csv(os.path.join(pathOutput, srcResLite[i] + '_res.csv'), index=False)

    del srcResLite, pathResLite, i, pathOutput, df_


# %% # .\Outputs\...\*.csv >>> ENTIRE.csv and MIN.csv #

if True:
    dirOutputs = utils_cf.get_immediate_subdirectories(path['Outputs'])
    srcResCSV = [srcRes for srcRes in dirOutputs if 'PyResults' in srcRes]
    pathResCSV = [os.path.join(path['Outputs'], \
        srcResCSV[i], srcResCSV[i]+'_res.csv') for i in range(len(srcResCSV))]

    for i in range(len(pathResCSV)):
        df_ = pd.read_csv(pathResCSV[i])
        try:
            df_entire = pd.concat([df_entire, df_], axis=0, sort=False).sort_values(by=['idModel', 'idVehicle', 'idExpCal'], ascending=True)
        except:
            df_entire = df_.copy()
    df_entire.to_csv(os.path.join(path['Outputs'], '_res_ENTIRE.csv'), index=False)

    # Select the minimum results from multiple calibrations
    df_min = utils_cf.select_min_res(
        df_entire,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names,
        CalGOF_IDs
    )
    df_min.to_csv(os.path.join(path['Outputs'], '_res_MIN.csv'), index=False)

    del dirOutputs, srcResCSV, pathResCSV, i, df_, df_entire, df_min


# %% # MIN.csv >>> _ObjCost.csv

if True:
    df_ = pd.read_csv(os.path.join(path['Outputs'], '_res_MIN.csv'))
    df_cal = utils_ff.organize_cal_res(
        df_,
        Veh_IDs,
        Exp_IDs,
        GOF_Names,
        CalGOF_IDs
    )
    df_cal.to_csv(os.path.join(path['Outputs'], \
        '_res_MIN_ObjCost_'+GOF_Names[CalGOF_IDs[0]-1]+'.csv'), index=False)
    
    del df_, df_cal


# %% # MIN.csv >>> _Parm.csv

if True:
    df_ = pd.read_csv(os.path.join(path['Outputs'], '_res_MIN.csv'))
    df_parm = utils_cf.organize_opt_parm(
        df_,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names,
        CalGOF_IDs
    )
    df_parm.to_csv(os.path.join(path['Outputs'], \
        '_res_MIN_Parm.csv'), index=False)
    
    del df_, df_parm


# %% # MIN.csv >>> All_GOFs.csv, _CAL.csv, _VAL.csv

# Combined validation results of all GOFs
if True:
    df_ = pd.read_csv(os.path.join(path['Outputs'], '_res_MIN.csv'))
    df_val_all_gofs = utils_ff.organize_val_res_all_gof(
        df_,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names
    )
    df_val_all_gofs.to_csv(os.path.join(path['Outputs'], \
        '_res_MIN_All_GOFs.csv'), index=False)

    del df_, df_val_all_gofs

# Separate validation results of each GOF
if True:
    df_ = pd.read_csv(os.path.join(path['Outputs'], \
        '_res_MIN_All_GOFs.csv'))
    for GOFName in GOF_Names:
        df_val, df_cal = utils_ff.organize_val_res_single_gof(
            df_,
            Model_IDs,
            GOFName
        )
        df_cal.to_csv(os.path.join(path['Outputs'], \
            '_res_MIN_CAL_'+GOFName+'.csv'), index=False)
        df_val.to_csv(os.path.join(path['Outputs'], \
            '_res_MIN_VAL_'+GOFName+'.csv'), index=False)

    del df_, GOFName, df_val, df_cal


# %% # Plot calibration results

if True:
    pathFig = os.path.join(path['Outputs'], 'FigCal')
    if not os.path.exists(pathFig):
        os.makedirs(pathFig, exist_ok=True)

    for GOFName in [GOF_Names[i-1] for i in [1,2]+CalGOF_IDs]:
        df_ = pd.read_csv(os.path.join(path['Outputs'], \
            '_res_MIN_CAL_'+GOFName+'.csv'))
        df_des = utils_cf.plot_cal_res(
            df_,
            Model_IDs,
            GOFName,
            pathFig,
            Submodel_IDs
        )
        df_temp = df_des['50%'].rename(GOFName)
        try:
            df_median = pd.concat([df_median, df_temp], axis=1)
        except:
            df_median = df_temp
        df_median.to_csv(os.path.join(pathFig, 'Median_Cal_SubNoCrash_BaseRel_Des.csv'))

    del pathFig, df_, df_des, df_median


# %% # Plot validation results

if True:
    pathFig = os.path.join(path['Outputs'], 'FigVal')
    if not os.path.exists(pathFig):
        os.makedirs(pathFig, exist_ok=True)

    df_ = pd.read_csv(os.path.join(path['Outputs'], \
            '_res_MIN_VAL_'+GOF_Names[0]+'.csv'))
    utils_cf.plot_val_crash(
        df_,
        Model_IDs,
        pathFig
    )

    for GOFName in [GOF_Names[i-1] for i in [1,2]+CalGOF_IDs]:
        df_ = pd.read_csv(os.path.join(path['Outputs'], \
            '_res_MIN_VAL_'+GOFName+'.csv'))
        df_des = utils_cf.plot_val_res(
            df_,
            Model_IDs,
            GOFName,
            pathFig,
            Submodel_IDs
        )
        df_temp = df_des['50%'].rename(GOFName)
        try:
            df_median = pd.concat([df_median, df_temp], axis=1)
        except:
            df_median = df_temp
        df_median.to_csv(os.path.join(pathFig, 'Median_Val_SubNoCrash_BaseRel_Des.csv'))

    del pathFig, df_, df_des, df_median

# %%
