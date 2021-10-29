#%%
import os, sys, time
rootDir = os.path.abspath(__file__ + "/../../../")
sys.path.append(rootDir)

import numpy as np
import pandas as pd
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from IPython.core import display as ICD
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot

#%%
#############################
# Provide global parameters #
#############################
Project = 'ASTAZERO'
Database = 'Exp'

path = {'Root': os.path.abspath(os.path.join(__file__ ,'../..'))}
path['InputData'] = os.path.join(path['Root'], 'Data', Project, Database, 'Astazero linear data')
path['Outputs'] = os.path.join(path['Root'], 'Outputs', 'AstaZero', 'Asta2DTrack')
os.makedirs(path['Outputs'], exist_ok=True)

del Project, Database

df = pd.read_csv(os.path.join(path['InputData'], \
    'AstaZero_rural_road_axis_curvLast.csv'))[['Latitude', 'Longitude', 'E', 'N']]

plt.plot(df['E'], df['N'])
plt.axes().set_aspect('equal')
plt.xlabel('E (m)')
plt.ylabel('N (m)')
plt.show()
plt.close()
df

#%%
df['X'] = np.array([None] * len(df))
for i in range(len(df)):
    if i == 0:
       df['X'][i] = 0
    else:
        df['X'][i] = df['X'][i-1] + np.sqrt((df['E'][i]-df['E'][i-1])**2 + (df['N'][i]-df['N'][i-1])**2)
del i
df

#%%
df_x_alt = pd.read_csv(os.path.join(path['InputData'], 'look_up.csv'))
f_x_alt = interp1d(df_x_alt['X'], df_x_alt['Alt'], fill_value='extrapolate')
df['Alt'] = f_x_alt(df['X'].astype(np.float64))
df

#%%
f_x_alt = interpolate.interp1d(df['X'], df['Alt'], fill_value='extrapolate')
f_x_e = interpolate.interp1d(df['X'], df['E'], fill_value='extrapolate')
f_x_n = interpolate.interp1d(df['X'], df['N'], fill_value='extrapolate')
df_ = pd.DataFrame()
X = df['X'].to_list()
df_['X'] = list(np.arange(0, np.floor(X[-1]), 10)) + [X[-1]]
df_['Alt'] = f_x_alt(df_['X'])
df_['E'] = f_x_e(df_['X'])
df_['N'] = f_x_n(df_['X'])
df = df_
df
#%%
df['SlopeSin'] = np.array([None] * len(df))
for i in range(len(df)):
    if i == 0:
        pass
    else:
        if df['X'][i] == df['X'][i-1]:
            df['SlopeSin'][i] = df['SlopeSin'][i-1]        
        else:
            df['SlopeSin'][i] = np.sin(np.arctan2(df['Alt'][i]-df['Alt'][i-1], df['X'][i]-df['X'][i-1]))
df['SlopeSin'][0] = df['SlopeSin'][1]
del i
plt.plot(df['X'], df['SlopeSin'])
df['SlopeSin'].iloc[:-1] = signal.savgol_filter(df['SlopeSin'].iloc[:-1], 31, 3)
df['SlopeSin'].iloc[-1] = df['SlopeSin'][0]
plt.plot(df['X'], df['SlopeSin'])
plt.xlabel('X (m)')
plt.ylabel('SlopeSin')
plt.savefig(os.path.join(path['Outputs'], 'TrackData2D-X-SlopeSin.png'))
plt.show()
plt.close()
df

#%%
df['Yaw'] = np.array([None] * len(df))
for i in range(len(df)):
    if i == 0:
        pass
    elif i == len(df)-1 or df['E'][i+1] == df['E'][i-1]:
        df['Yaw'][i] = df['Yaw'][i-1]
    else:
        df['Yaw'][i] = np.arctan2(df['N'][i+1]-df['N'][i-1], df['E'][i+1]-df['E'][i-1])
df['Yaw'][0] = df['Yaw'][1]
del i
plt.plot(df['X'], df['Yaw'])
plt.xlabel('X (m)')
plt.ylabel('Yaw (rad)')
plt.show()
plt.close()
df

#%%
# https://en.wikipedia.org/wiki/Menger_curvature
df['Curvature'] = np.array([None] * len(df))
for i in range(len(df)):
    if i == 0:
        pass
    elif i == len(df)-1:
        df['Curvature'][i] = df['Curvature'][i-1]
    else:
        # https://ncalculators.com/geometry/triangle-area-by-3-points.htm (The following two equations of A_ are the same)
        # A_ = 0.5 * ((df['E'][i]-df['E'][i-1])*(df['N'][i+1]-df['N'][i-1]) - (df['N'][i]-df['N'][i-1])*(df['E'][i+1]-df['E'][i-1]))
        A_ = 0.5 * (df['E'][i-1]*(df['N'][i] - df['N'][i+1]) + df['E'][i]*(df['N'][i+1] - df['N'][i-1]) + df['E'][i+1]*(df['N'][i-1] - df['N'][i]))
        norm_a = np.sqrt((df['E'][i-1]-df['E'][i])**2 + (df['N'][i-1]-df['N'][i])**2)
        norm_b = np.sqrt((df['E'][i]-df['E'][i+1])**2 + (df['N'][i]-df['N'][i+1])**2)
        norm_c = np.sqrt((df['E'][i+1]-df['E'][i-1])**2 + (df['N'][i+1]-df['N'][i-1])**2)

        df['Curvature'][i] = 4 * A_ / (norm_a * norm_b * norm_c)
df['Curvature'][0] = df['Curvature'][1]
del i
plt.plot(df['X'], df['Curvature'])
df['Curvature'].iloc[:-1] = signal.savgol_filter(df['Curvature'].iloc[:-1], 41, 3)
df['Curvature'].iloc[-1] = df['Curvature'][0]
plt.plot(df['X'], df['Curvature'])
plt.xlabel('X (m)')
plt.ylabel('Curvature (1/m)')
plt.savefig(os.path.join(path['Outputs'], 'TrackData2D-X-Curvature.png'))
plt.show()
plt.close()
df

#%%
df.to_csv(os.path.join(path['Outputs'], 'trackData2D.csv'), index=False)

# %%
