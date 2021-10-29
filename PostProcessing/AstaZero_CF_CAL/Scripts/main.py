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

# for i in list(range(41, 82)):
#     Model_IDs.remove(i)

CalGOF_IDs = [10]
Veh_IDs = [2, 3, 4, 5]
Exp_IDs = [1, 2, 3, 4, 5, 6, 7]

path['ResultsFull'] = os.path.join(path['ResRoot'], 'ResultsFull', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['ResultsLite'] = os.path.join(path['ResRoot'], 'ResultsLite', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])
path['Outputs'] = os.path.join(path['ResRoot'], 'Outputs', 'ObjGOF_'+GOF_Names[CalGOF_IDs[0]-1])


#%%
if __name__ == "__main__":
    #######################################################
    # .\ResultsFull\...\*.txt >>> .\ResultsLite\...\*.txt #
    #######################################################
    if False:
        srcResFull = utils.get_immediate_subdirectories(path['ResultsFull'])
        srcResFull = [srcRes for srcRes in srcResFull if 'PyResults' in srcRes]
        pathResFull = [os.path.join(path['ResultsFull'], \
            srcResFull[i], Project, Database) for i in range(len(srcResFull))]

        pathResLite = [os.path.join(path['ResultsLite'], \
            srcResFull[i], Project, Database) for i in range(len(srcResFull))]
        
        assert (len(pathResFull) == len(pathResLite)), \
            '\n The number of source directories must be equal to that of target directories.'
        for i in range(len(pathResFull)):
            utils.remove_redundant_files(
                pathResFull[i],
                pathResLite[i],
                CalGOF_IDs,
                Model_IDs,
                Veh_IDs,
                Exp_IDs,
                GOF_Names)
        del i, pathResFull, pathResLite, srcResFull
    ###################################################
    # .\ResultsLite\...\*.txt >>> .\Outputs\...\*.csv #
    ###################################################
    if True:
        srcResLite = utils.get_immediate_subdirectories(path['ResultsLite'])
        srcResLite = [srcRes for srcRes in srcResLite if 'PyResults' in srcRes]
        pathResLite = [os.path.join(path['ResultsLite'], \
            srcResLite[i], Project, Database) for i in range(len(srcResLite))]

        for i in range(len(pathResLite)):
            pathOutput = os.path.join(path['Outputs'], srcResLite[i])
            os.makedirs(pathOutput, exist_ok=True)
            df_ = utils.create_res_csv(
                srcResLite[i],
                pathResLite[i],
                path['ExpLabel'],
                path['ParmKeys'],
                CalGOF_IDs,
                Model_IDs,
                Veh_IDs, 
                Exp_IDs,
                GOF_Names,
                Exp_Names)
            df_.to_csv(os.path.join(pathOutput, srcResLite[i] + '_res.csv'), index=False)

        del srcResLite, pathResLite, i, pathOutput, df_
    ##################################################
    # .\Outputs\...\*.csv >>> ENTIRE.csv and MIN.csv #
    ##################################################
    if True:
        dirOutputs = utils.get_immediate_subdirectories(path['Outputs'])
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
        df_min = utils.select_min_res(
            df_entire,
            Model_IDs,
            Veh_IDs,
            Exp_IDs,
            GOF_Names,
            CalGOF_IDs
        )
        df_min.to_csv(os.path.join(path['Outputs'], '_res_MIN.csv'), index=False)

        del dirOutputs, srcResCSV, pathResCSV, i, df_, df_entire, df_min
    ############################
    # MIN.csv >>> _ObjCost.csv #
    ############################
    if True:
        df_ = pd.read_csv(os.path.join(path['Outputs'], '_res_MIN.csv'))
        df_cal = utils.organize_cal_res(
            df_,
            Veh_IDs,
            Exp_IDs,
            GOF_Names,
            CalGOF_IDs
        )
        df_cal.to_csv(os.path.join(path['Outputs'], \
            '_res_MIN_ObjCost_'+GOF_Names[CalGOF_IDs[0]-1]+'.csv'), index=False)
        
        del df_, df_cal
    #########################
    # MIN.csv >>> _Parm.csv #
    #########################
    if True:
        df_ = pd.read_csv(os.path.join(path['Outputs'], '_res_MIN.csv'))
        df_parm = utils.organize_opt_parm(
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
    ################################################
    # MIN.csv >>> All_GOFs.csv, _CAL.csv, _VAL.csv #
    ################################################
    # Combined validation results of all GOFs
    if True:
        df_ = pd.read_csv(os.path.join(path['Outputs'], '_res_MIN.csv'))
        df_val_all_gofs = utils.organize_val_res_all_gof(
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
            df_val, df_cal = utils.organize_val_res_single_gof(
                df_,
                Model_IDs,
                GOFName
            )
            # df_val_nocrash = df_val.loc[df_val['CNT_BLK']==0]
            df_cal.to_csv(os.path.join(path['Outputs'], \
                '_res_MIN_CAL_'+GOFName+'.csv'), index=False)
            df_val.to_csv(os.path.join(path['Outputs'], \
                '_res_MIN_VAL_'+GOFName+'.csv'), index=False)
            # df_val_nocrash.to_csv(os.path.join(path['Outputs'], \
            #     '_res_MIN_VAL_NoCrash_'+GOFName+'.csv'), index=False)

        del df_, GOFName, df_val, df_cal
    ############################
    # Plot calibration results #
    ############################
    if True:
        pathFig = os.path.join(path['Outputs'], 'FigCal')
        if not os.path.exists(pathFig):
            os.makedirs(pathFig, exist_ok=True)

        for GOFName in [GOF_Names[i-1] for i in [1,2,3]+CalGOF_IDs]:
            df_ = pd.read_csv(os.path.join(path['Outputs'], \
                '_res_MIN_CAL_'+GOFName+'.csv'))
            df_des = utils.plot_cal_res(
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
    ############################
    # Plot validation results #
    ############################
    if True:
        pathFig = os.path.join(path['Outputs'], 'FigVal')
        if not os.path.exists(pathFig):
            os.makedirs(pathFig, exist_ok=True)

        df_ = pd.read_csv(os.path.join(path['Outputs'], \
                '_res_MIN_VAL_'+GOF_Names[0]+'.csv'))
        utils.plot_val_crash(
            df_,
            Model_IDs,
            pathFig
        )

        for GOFName in [GOF_Names[i-1] for i in [1,2,3]+CalGOF_IDs]:
            df_ = pd.read_csv(os.path.join(path['Outputs'], \
                '_res_MIN_VAL_'+GOFName+'.csv'))
            df_des = utils.plot_val_res(
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
