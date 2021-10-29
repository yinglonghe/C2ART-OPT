#%%
import os, sys, time
rootDir = os.path.abspath(__file__ + "/../../../")
sys.path.append(rootDir)

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from IPython.core import display as ICD
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
from PreProcessing.Scripts.AstaZero import utils

#%%
#############################
# Provide global parameters #
#############################
Project = 'ASTAZERO'
Database = 'Exp'

path = {'Root': os.path.abspath(os.path.join(__file__ ,'../..'))}
path['InputData'] = os.path.join(path['Root'], 'Data', Project, Database, 'PyData')
path['Outputs'] = os.path.join(path['Root'], 'Outputs', 'AstaZero', 'AstaPyData')
os.makedirs(path['Outputs'], exist_ok=True)

Exp_Names = [
    'ASta_040719_platoon5',  # P-1 (ID)
    'ASta_040719_platoon6',  # P-2
    'ASta_040719_platoon7',  # P-3
    'ASta_040719_platoon8',  # P-4
    'ASta_040719_platoon9',  # P-5
    'ASta_050719_platoon1',  # P-6
    'ASta_050719_platoon2',  # P-7
]
Veh_Names = [
    'Tesla model 3',    # C-2 (ID)
    'BMW X5',           # C-3
    'Audi A6',          # C-4
    'Mercedes A class', # C-5
]

Veh_IDs = [1, 2, 3, 4, 5]           # C-#
Exp_IDs = [1, 2, 3, 4, 5, 6, 7]     # P-#
pathLabel = os.path.join(path['InputData'], 'Labels.csv')
pathData = [os.path.join(path['InputData'], exp_name+'.csv') for exp_name in Exp_Names]

timeStep = 0.1
trackLen = 5757.90385
# start_time = time.time()
# print(time.time() - start_time)


#%%
if __name__ == '__main__':
    ############################
    # Load platoon information #
    ############################
    expLabels = pd.read_csv(pathLabel)
    expLabels = expLabels[expLabels['File'].isin(Exp_Names)].reset_index(drop=True)
    expInfo =pd.DataFrame()
    expInfo['File'] = expLabels['File']
    expInfo['VehID'] = expLabels.iloc[:, 1:6].values.tolist()
    expInfo['VehA2F'] = expLabels.iloc[:, 6:11].round(3).values.tolist()
    expInfo['VehA2B'] = expLabels.iloc[:, 11:16].round(3).values.tolist()
    expInfo['VehLen'] = expLabels.iloc[:, 16:21].round(3).values.tolist()
    del expLabels
    expInfo

#%%
    for i in range(len(Exp_Names)):
        #############
        # Load data #
        #############
        df_ = pd.read_csv(pathData[i])
        t_x_col_ = ['Time'] + ['X'+str(m) for m in range(1,6)]
        t_v_col_ = ['Time'] + ['SpeedDoppler'+str(m) for m in range(1,6)]
        
        t_v_init = df_[t_v_col_].values[0].tolist()
        t_x_df = df_[t_x_col_]
        del df_
        expTX = t_x_df.to_numpy().transpose()
        t = expTX[0]
        print('Load data of '+Exp_Names[i])

        ###########################################################
        # Check time steps and apply interpolation when necessary #
        ###########################################################
        if all(0.099 < dt < 0.101 for dt in np.diff(t)):
            expTX_even = expTX 
        else:
            print('Warning: Interpolation is implemented for ' + Exp_Names[i] + ' (unevenly-spaced time series)!')
            expTX_even = utils.coord_interp(expTX)
        
        #####################################################
        # Reconstruct the data from curvilinear coordinates #
        #####################################################
        expTXVAS = utils.data_reconstruction(
            timeStep,
            expTX_even,
            t_v_init,
            expInfo.loc[expInfo['File']==Exp_Names[i], 'VehLen'].tolist()[0],
            trackLen,
            Exp_Names[i]
        )

        ######################################################
        # Plot and save net spacing, speed, and acceleration #
        ######################################################
        utils.plot_exp(
            expTXVAS, 
            path['Outputs'],
            Exp_Names[i],
            autoOpen=False
            )

        ############################################
        # Save data as single CSV in Outputs foler #
        ############################################
        t_x_v_col_ = t_x_col_ + t_v_col_[1:] + ['Accel'+str(m) for m in range(1,6)] + ['Spacing'+str(m) for m in range(2,6)]
        df_new = pd.DataFrame(data=expTXVAS.transpose(), columns=t_x_v_col_)
        if True:
            df_new.to_csv(os.path.join(path['Outputs'], Exp_Names[i]+'.csv'), index=False)

        #########################################################
        # Comparison between the reconstructed and the original #
        #########################################################
        # Original experimental data 
        expTXVASorg = utils.original_data_process(
            expTX,
            t_v_init,
            expInfo.loc[expInfo['File']==Exp_Names[i], 'VehLen'].tolist()[0],
            trackLen
        )
        # Plots for Comparison
        if True:
            utils.plot_comp(
                expTXVASorg,
                expTXVAS,
                t_x_v_col_,
                Exp_Names[i],
                path['Outputs'],
                ['png', 'html'][1]
            )

# %%
