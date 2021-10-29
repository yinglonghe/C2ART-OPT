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
path['InputData'] = os.path.join(path['Root'], 'Outputs', 'AstaZero', 'AstaPyData')
path['Outputs'] = os.path.join(path['Root'], 'Outputs', 'AstaZero', 'AstaFreeFlowStable')
os.makedirs(path['Outputs'], exist_ok=True)

timeStep = 0.1
trackLen = 5757.90385

dfLabel = pd.read_csv(os.path.join(path['Root'], 'Data', Project, Database, 'PyData', 'FreeFlowStableParts.csv'))

#%%
if __name__ == '__main__':
    ############################
    # Load platoon information #
    ############################
    for i in range(len(dfLabel)):
        df = pd.read_csv(os.path.join(path['InputData'], dfLabel['File'][i]+'.csv'))
        df_lead_stable = df.loc[(df['Time'] >= dfLabel['timeStart'][i]) & (df['Time'] <= dfLabel['timeEnd'][i]), \
            ['Time', 'X1', 'SpeedDoppler1', 'Accel1']].reset_index(drop=True)
        df_lead_stable['Time'] -= df_lead_stable['Time'][0]
        df_lead_stable['X1'] = df_lead_stable['X1']%trackLen

        # Save CSV
        fileName = dfLabel['File'][i]+'_'+str(dfLabel['timeStart'][i])+'_'+str(dfLabel['timeEnd'][i])+'_V'+str(dfLabel['setSpd_kmph'][i])
        df_lead_stable.to_csv(os.path.join(path['Outputs'], fileName+'.csv'), index=False)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_lead_stable['Time'], y=df_lead_stable['SpeedDoppler1'],
                name='Speed'))
        fig.add_trace(go.Scatter(x=df_lead_stable['Time'], y=np.ones(len(df_lead_stable['Time']))*dfLabel['setSpd_kmph'][i]/3.6,
                name='setSpeed'))
        fig.update_xaxes(title_text='Time [s]')
        fig.update_yaxes(title_text='Speed (the leading vehicle) [m/s]')
        fig.update_layout(title_text=fileName)
        plot(fig, filename=os.path.join(path['Outputs'], fileName+'.html'), auto_open=False)



# %%
