#%%
import os, sys, time, math
rootDir = os.path.abspath(__file__ + "/../../../")
sys.path.append(rootDir)

import numpy as np
import pandas as pd
from scipy import interpolate, signal
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
path['InputData'] = os.path.join(path['Root'], 'Data', Project, Database)
path['Outputs'] = os.path.join(path['Root'], 'Outputs', 'AstaZero', 'AstaSlope')
os.makedirs(path['Outputs'], exist_ok=True)

pathSlope = os.path.join(path['InputData'], 'Slope.csv')


#%%
df = pd.read_csv(pathSlope)
x = df['X'].to_numpy()
y = df['Y'].to_numpy()

df_ = df.copy()
df_.drop(df_.loc[(df_['X']>847.5) & (df_['X']<967.5)].index, inplace=True)
df_.drop(df_.loc[(df_['X']>4537.5) & (df_['X']<4587.5)].index, inplace=True)
x_ = df_['X'].to_numpy()
y_ = df_['Y'].to_numpy()

f = interpolate.interp1d(x_, y_)
xstep = 0.5
xinterp = np.arange(math.ceil(x_[0]), math.floor(x_[-1])+xstep, xstep)
yinterp = f(xinterp)

xfltr = xinterp
ytemp = yinterp
for i in range(3):
    yfltr = signal.savgol_filter(ytemp, 101, 3)
    ytemp = yfltr

ffltr = interpolate.interp1d(xfltr, yfltr, fill_value="extrapolate")
ynew = ffltr(x)

# plt.plot(x, y, '-y', label='xy')
# ## plt.plot(xinterp, yinterp, '--b', label='xyinterp')
# ## plt.plot(xfltr, yfltr, ':k', label='xyfltr')
# plt.plot(x, ynew, ':k', label='xynew')
# plt.legend()
# plt.show()

df['Ynew'] = ynew
df.to_csv(os.path.join(path['Outputs'], 'smoothRoadSlope.csv'), index=False)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
        name='X-Y'))
fig.add_trace(go.Scatter(x=x, y=ynew,
        name='X-Ynew')) 
fig.update_xaxes(title_text='X (longitudinal coordinate of AstaZero) [m]')
fig.update_yaxes(title_text='Y (sine of road slope)')
fig.update_layout(title_text='Smoothing road slope of AstaZero')
plot(fig, filename=os.path.join(path['Outputs'], 'smoothRoadSlope.html'), auto_open=False)
