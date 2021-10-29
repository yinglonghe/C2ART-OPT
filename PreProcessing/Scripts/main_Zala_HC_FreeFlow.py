#%%
import os, sys, time
rootDir = os.path.abspath(__file__ + "/../../../")
sys.path.append(rootDir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core import display as ICD
from scipy import interpolate, signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import PreProcessing.Scripts.GeoConvert.geofun as geo
import PreProcessing.Scripts.ZalaZone.utils as utils
from c2art_env.utils import numbadiff


#%%
def smoothSpd(df, pathOutputs, fileName):
    v = df['SpeedDoppler'].tolist()
    for i in range(2):
        v_fltr = signal.savgol_filter(v, 51, 3)
        v = v_fltr

    v_fltr_pos = np.clip(v_fltr, 0, None)

    df['SpeedDoppler'] = v_fltr
    
    return df


def getENU(df):

    df['E'], df['N'], _ = geo.geo2enu(df['Lat'] / 180 * np.pi, df['Lon'] / 180 * np.pi, 0, center[0], center[1], 0)
    
    plt.plot(df['E'], df['N'], '-ok')
    plt.axes().set_aspect('equal')
    plt.xlabel('E')
    plt.ylabel('N')
    plt.show()

    return df


def getX(df):
    x0 = df['X'][0]
    df['X'] = np.array([None] * len(df))
    for i in range(len(df)):
        if i == 0:
            df['X'][i] = x0
        else:
            df['X'][i] = df['X'][i-1] + np.sqrt((df['E'][i]-df['E'][i-1])**2 + (df['N'][i]-df['N'][i-1])**2)
    del i

    return df


def getSlopeSin(df):
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

    return df


def getYaw(df):
    df['Yaw'] = np.array([None] * len(df))
    for i in range(len(df)):
        if i == 0:
            pass
        elif i == len(df)-1 or df['E'][i+1] == df['E'][i-1]:
            df['Yaw'][i] = df['Yaw'][i-1]
        else:
            df['Yaw'][i] = np.arctan2(df['N'][i+1]-df['N'][i-1], df['E'][i+1]-df['E'][i-1])
    df['Yaw'][0] = df['Yaw'][1]

    return df


def getCurvature(df):
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

    return df


#%%
Project = 'ZalaZone_HC'
Database = 'FreeFlowExp'

path = {'Root': os.path.abspath(os.path.join(__file__ ,'../..'))}
path['InputData'] = os.path.join(path['Root'], 'Data', Project, Database)
path['Outputs'] = os.path.join(path['Root'], 'Outputs', Project, Database)
os.makedirs(path['Outputs'], exist_ok=True)

del Project, Database

dfLabel = pd.read_csv(os.path.join(path['InputData'], 'FreeFlowStableParts.csv'))
dfLabel

#%%
center = [0.8183036453779022, 0.29399757077550087]##handling
for k in range(len(dfLabel)):
    fileName = dfLabel['File'][k]+'_'+str(dfLabel['timeStart'][k])+'_'+str(dfLabel['timeEnd'][k])+'_V'+str(dfLabel['setSpd_kmph'][k])

    df_ = pd.read_csv(os.path.join(path['InputData'], dfLabel['File'][k]+'.csv'))
    df_ = df_.loc[(df_['Time'] >= dfLabel['timeStart'][k]) & (df_['Time'] <= dfLabel['timeEnd'][k])].reset_index(drop=True)

    col_ = list(df_.columns.values)
    df = pd.DataFrame()
    T = df_['Time'].to_list()
    df['Time'] = np.arange(T[0], T[-1]+0.1, 0.1)

    for i in range(1,len(col_)):
        f_ = interpolate.interp1d(df_['Time'], df_[col_[i]], fill_value='extrapolate')
        df[col_[i]] = f_(df['Time'])

    del i, f_, df_, col_, T
    df['Time'] = df['Time'] - df['Time'][0]

    plt.plot(df['Time'], df['SpeedDoppler'])
    df = smoothSpd(df, path['Outputs'], fileName)
    plt.plot(df['Time'], df['SpeedDoppler'])
    plt.savefig(os.path.join(path['Outputs'], fileName+'_speed.png'))
    plt.show()
    plt.close()

    df = df.rename(columns={'SpeedDoppler': 'SpeedDoppler1', 'X': 'X1'})

    df['Accel1'] = np.append((numbadiff(df['SpeedDoppler1'].to_numpy()) / numbadiff(df['Time'].to_numpy())), 0)
    plt.plot(df['Time'], df['Accel1'])
    plt.savefig(os.path.join(path['Outputs'], fileName+'_accel.png'))
    plt.show()
    plt.close()

    df.to_csv(os.path.join(path['Outputs'], fileName+'.csv'), index=False)

    ICD.display(df)


# %%
df_ = pd.read_csv(os.path.join(path['InputData'], 'TrackData2D.csv'))
df_

f_x_alt = interpolate.interp1d(df_['X'], df_['Alt'], fill_value='extrapolate')
f_x_e = interpolate.interp1d(df_['X'], df_['E'], fill_value='extrapolate')
f_x_n = interpolate.interp1d(df_['X'], df_['N'], fill_value='extrapolate')

df = pd.DataFrame()
X = df_['X'].to_list()
df['X'] = list(np.arange(0, np.floor(X[-1]), 5)) + [X[-1]]
df['Alt'] = f_x_alt(df['X'])
df['E'] = f_x_e(df['X'])
df['N'] = f_x_n(df['X'])

df = getSlopeSin(df)
plt.plot(df['X'], df['SlopeSin'])
df['SlopeSin'].iloc[:-1] = signal.savgol_filter(df['SlopeSin'].iloc[:-1], 39, 5)
df['SlopeSin'].iloc[-1] = df['SlopeSin'][0]
plt.plot(df['X'], df['SlopeSin'])
plt.xlabel('X (m)')
plt.ylabel('SlopeSin')
plt.savefig(os.path.join(path['Outputs'], 'TrackData2D-X-SlopeSin.png'))
plt.show()
plt.close()

df = getYaw(df)
plt.plot(df['X'], df['Yaw'])
plt.xlabel('X (m)')
plt.ylabel('Yaw (rad)')
plt.show()
plt.close()

df = getCurvature(df)
plt.plot(df['X'], df['Curvature'])
df['Curvature'].iloc[:-1] = signal.savgol_filter(df['Curvature'].iloc[:-1], 39, 6)
df['Curvature'].iloc[-1] = df['Curvature'][0]
plt.plot(df['X'], df['Curvature'])
plt.xlabel('X (m)')
plt.ylabel('Curvature (1/m)')
plt.savefig(os.path.join(path['Outputs'], 'TrackData2D-X-Curvature.png'))
plt.show()
plt.close()

ICD.display(df)

#%%
df.to_csv(os.path.join(path['Outputs'], 'trackData2D.csv'), index=False)

df
# %%
