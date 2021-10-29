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
from PostProcessing.AstaZero_CF_CAL.Scripts.c2art_res import utils


#%%
#############################
# Provide global parameters #
#############################
Project = 'ASTAZERO'
Database = 'Exp'
Precision = 5       # Decimal

path = {
    'Root': rootDir,
    'ResRoot': os.path.abspath(os.path.join(__file__ ,'../..'))
}

path['InputData'] = os.path.join(path['ResRoot'], 'Data', Project, Database)
path['Debug'] = os.path.join(path['ResRoot'], 'Debug')

path['ExpLabel'] = os.path.join(path['Root'], 'Data', Project, Database, 'PyData', 'Labels.csv')
path['ParmKeys'] = os.path.join(path['Root'], 'Data', Project, Database, 'ParmKeys.txt')

Exp_Names = [
    'ASta_040719_platoon5',  # P-1 (ID)
    'ASta_040719_platoon6',  # P-2
    'ASta_040719_platoon7',  # P-3
    'ASta_040719_platoon8',  # P-4
    'ASta_040719_platoon9',  # P-5
    'ASta_050719_platoon1',  # P-6
    'ASta_050719_platoon2']  # P-7
Veh_Names = [
    'Tesla model 3',    # C-2 (ID)
    'BMW X5',           # C-3
    'Audi A6',          # C-4
    'Mercedes A class'] # C-5
GOF_Names = [
    'RMSE(S)',          # GOF-1 (ID)
    'RMSE(V)',          # GOF-2
    'RMSE(A)',          # GOF-3
    'RMSPE(S)',         # GOF-4
    'RMSPE(V)',         # GOF-5
    'NRMSE(S)',         # GOF-6
    'NRMSE(V)',         # GOF-7
    'NRMSE(A)',         # GOF-8
    'NRMSE(S+V)',       # GOF-9
    'NRMSE(S+V+A)',     # GOF-10
    'U(S)',             # GOF-11
    'U(V)',             # GOF-12
    'U(A)',             # GOF-13
    'U(S+V)',           # GOF-14
    'U(S+V+A)',         # GOF-15
    'RMSE(Sn+Vn)',      # GOF-16
    'RMSE(Sn+Vn+An)',   # GOF-17
]

Submodel_IDs = {
    'IDM': np.arange(1, 19),
    'Gipps': np.arange(19, 37),
    'LNR-S-CTH': np.arange(37, 55),
    'LNR-S-IDM': np.arange(55, 73),
    'LNR-S-Gipps': np.arange(73, 91),
    # 'LNR-V-CTH': np.arange(91, 109),
    # 'LNR-V-FVDM': np.arange(109, 127),
    # 'LNR-V-Gipps': np.arange(127, 145),
}
Model_IDs = []
for key, value in Submodel_IDs.items():
    Model_IDs += list(value)
del key, value

CalGOF_IDs = [10]
Veh_IDs = [2, 3, 4, 5]
Exp_IDs = [1, 2, 3, 4, 5, 6, 7]

path['ResultsFull'] = os.path.join(path['ResRoot'], 'ResultsFull', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['ResultsLite'] = os.path.join(path['ResRoot'], 'ResultsLite', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['Outputs'] = os.path.join(path['ResRoot'], 'Outputs', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['Target'] = os.path.join(path['Outputs'], 'Cal_STD_Dist')

if not os.path.exists(path['Target']):
    os.makedirs(path['Target'], exist_ok=True)

#%%
srcResOutputs = utils.get_immediate_subdirectories(path['Outputs'])
srcRes = [src for src in srcResOutputs if 'PyResults' in src]
pathResOutputs = [os.path.join(path['Outputs'], srcRes[i], srcRes[i]+'_res.csv') for i in range(len(srcRes))]

target_GoF_Names = [GOF_Names[i-1] for i in [1,2,3]+CalGOF_IDs]

#%%
for ires in range(len(pathResOutputs)):
    df = pd.read_csv(pathResOutputs[ires])

    df_parm = utils.organize_opt_parm(
        df,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names,
        CalGOF_IDs
    )
    df_parm.to_csv(os.path.join(path['Target'], srcRes[ires]+'_Parm.csv'), index=False)

    df_val_all_gofs = utils.organize_val_res_all_gof(
        df,
        Model_IDs,
        Veh_IDs,
        Exp_IDs,
        GOF_Names
    )

    for GOFName in target_GoF_Names:
        df_val, df_cal = utils.organize_val_res_single_gof(
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

    df.loc[df['keyParm']=='k_a', 'valueParm'] = None
    df.loc[(df['keyParm']=='f1') & df['CalExp'].str.contains('Veh2Exp'), 'valueParm'] = None
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
