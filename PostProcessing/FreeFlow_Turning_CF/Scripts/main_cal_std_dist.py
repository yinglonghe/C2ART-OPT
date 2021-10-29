#%%
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


#%%
#############################
# Provide global parameters #
#############################
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
path['Target'] = os.path.join(path['Outputs'], 'Cal_STD_Dist')

if not os.path.exists(path['Target']):
    os.makedirs(path['Target'], exist_ok=True)

#%%
srcResOutputs = utils_cf.get_immediate_subdirectories(path['Outputs'])
srcRes = [src for src in srcResOutputs if 'PyResults' in src]
pathResOutputs = [os.path.join(path['Outputs'], srcRes[i], srcRes[i]+'_res.csv') for i in range(len(srcRes))]

target_GoF_Names = [GOF_Names[i-1] for i in [1,2]+CalGOF_IDs]

#%%
for ires in range(len(pathResOutputs)):
    df = pd.read_csv(pathResOutputs[ires])

    df_parm = utils_cf.organize_opt_parm(
        df,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names,
        CalGOF_IDs
    )
    df_parm.to_csv(os.path.join(path['Target'], srcRes[ires]+'_Parm.csv'), index=False)

    df_val_all_gofs = utils_ff.organize_val_res_all_gof(
        df,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names
    )

    for GOFName in target_GoF_Names:
        df_val, df_cal = utils_ff.organize_val_res_single_gof(
            df_val_all_gofs,
            Model_IDs,
            GOFName
        )

        df_cal.to_csv(os.path.join(path['Target'], srcRes[ires]+'_'+GOFName+'.csv'), index=False)
    
    ICD.display(df)

#%%
for GOFName in target_GoF_Names:
    for ires in range(len(pathResOutputs)):
        df = pd.read_csv(os.path.join(path['Target'], srcRes[ires]+'_'+GOFName+'.csv'))

        df_1D = pd.DataFrame()
        for imod in Model_IDs:
            df_c = pd.DataFrame()
            df_c = df[['idVehicle', 'idExpCal', 'idExpVal']]
            df_c.insert(loc=0, column='idModel', value=imod)
            df_c[GOFName] = df['Model_'+str(imod)]

            try:
                df_1D = pd.concat([df_1D, df_c], axis=0)
            except:
                df_1D = df_c

        if ires == 0:
            dfGoF = pd.DataFrame()
            dfGoF = df_1D[['idModel', 'idVehicle', 'idExpCal', 'idExpVal']]
            dfGoF[srcRes[ires]] = df_1D[GOFName]
        else:
            dfGoF[srcRes[ires]] = df_1D[GOFName]

    dfGoF.to_csv(os.path.join(path['Target'], 'All_'+GOFName+'.csv'), index=False)


# %%
col_ = ['Veh'+str(iveh)+'Exp'+str(iexp) for iveh in Veh_IDs for iexp in Exp_IDs]

dfParm = pd.DataFrame()
for ires in range(len(pathResOutputs)):
    df = pd.read_csv(os.path.join(path['Target'], srcRes[ires]+'_Parm.csv'))
    df_1D = pd.DataFrame()
    for c in col_:
        df_c = pd.DataFrame()
        df_c = df[['idModel', 'keyParm']]
        df_c['CalExp'] = c
        df_c['valueParm'] = df[c]

        try:
            df_1D = pd.concat([df_1D, df_c], axis=0)
        except:
            df_1D = df_c
    df_1D.to_csv(os.path.join(path['Target'], srcRes[ires]+'_Parm_1D.csv'), index=False)
#%%
for ires in range(len(pathResOutputs)):
    df = pd.read_csv(os.path.join(path['Target'], srcRes[ires]+'_Parm_1D.csv'))
    if ires == 0:
        dfParm = pd.DataFrame()
        dfParm = df[['idModel', 'keyParm', 'CalExp']]
        dfParm[srcRes[ires]] = df['valueParm']
    else:
        dfParm[srcRes[ires]] = df['valueParm']

dfParm.sort_values(by=['idModel', 'CalExp'], inplace=True)
dfParm.dropna(axis='index', how='any', inplace=True)
dfParm.to_csv(os.path.join(path['Target'], 'All_Parm.csv'), index=False)



# %%
