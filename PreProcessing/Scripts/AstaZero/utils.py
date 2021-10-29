import os
import numpy as np
from numba import jit, njit, prange
from scipy import interpolate, signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt


@njit(nogil=True)
def numbadiff(x):
    return x[1:] - x[:-1]


# @njit(nogil=True)
def original_data_process(
    expTX,
    t_v_init,
    veh_len_list,
    trackLen
    ):

    expTS = np.zeros((len(expTX)-1, len(expTX[0])))
    expTS[0] = expTX[0]
    for i in range(2, len(expTX)):
        for j in range(len(expTX[0])):
            s = expTX[i-1][j] - expTX[i][j]
            if s < -5000:
                s = s % trackLen
            expTS[i-1][j] = s - veh_len_list[i-2]

    expTXnew = np.zeros_like(expTX)
    expTXnew[0] = expTX[0]
    for i in range(1, len(expTX)):
        for j in range(len(expTX[0])):
            if j == 0:
                expTXnew[i][j] = expTX[i][j]
            else:
                dx = expTX[i][j] - expTX[i][j-1]
                if dx < -5000:
                    dx = dx % trackLen
                expTXnew[i][j] = expTXnew[i][j-1] + dx

    expTV = np.zeros_like(expTX)
    expTV[0] = expTX[0]
    expTA = np.zeros_like(expTX)
    expTA[0] = expTX[0]
    for i in range(1, len(expTX)):
        expTV[i] = np.append(t_v_init[i], np.diff(expTXnew[i]) / np.diff(expTXnew[0]))
        expTV[i][expTV[i] < 0] = 0
        expTA[i] = np.append(np.diff(expTV[i]) / np.diff(expTV[0]), 0)

    return np.vstack((expTXnew, expTV[1:], expTA[1:], expTS[1:]))


# @njit(nogil=True)
def coord_interp(expTX):
    t = expTX[0]
    tnew = np.round(np.arange(round(t[0]*10), round(t[-1]*10+1), 1)*0.1, 1)
    expTX_Interp = np.empty([len(expTX), len(tnew)])
    expTX_Interp[0] = tnew

    for k in range(1, len(expTX)):
        f = interpolate.interp1d(t, expTX[k])
        expTX_Interp[k] = f(tnew)
    
    return expTX_Interp


@njit(nogil=True)
def curvilinear_pos2spd(
    x,
    dt,
    v_init,
    trackLen
    ):

    v_avg = np.zeros_like(x)
    v_ins = np.zeros_like(x)
    for j in range(len(x)):
        if j == 0:
            v_avg[j] = v_init
            v_ins[j] = v_avg[j]
        else:
            if x[j] - x[j-1] < -5000:
                v_avg[j] = ((x[j] - x[j-1]) % trackLen) / dt
            elif x[j] - x[j-1] < 0:
                # v_avg[j] = 0
                # print('Warning: negative average speed!')
                v_avg[j] = (x[j] - x[j-1]) / dt
            else:
                v_avg[j] = (x[j] - x[j-1]) / dt

            v_ins[j] = 2*v_avg[j] - v_avg[j-1]
        v_ins[j] = max(0, v_ins[j])

    return v_ins, v_avg


def spd_filter(
    v,
    fltr_window=None, 
    fltr_polyorder=None
    ):
    v_fltr = np.zeros_like(v)
    v_fltr = signal.savgol_filter(v, fltr_window, fltr_polyorder)
    v_fltr = np.maximum(v_fltr, 0)

    return v_fltr


@njit(nogil=True)
def spd2pos(
    v,
    dt,
    x_init
    ):

    x_cor = np.zeros_like(v)
    for j in range(len(v)):
        if j == 0:
            x_cor[j] = x_init
        else:
            x_cor[j] = x_cor[j-1] + (v[j]+v[j-1]) / 2 * dt

    return x_cor


@njit(nogil=True)
def pos2spacing(
    x_pre,
    x_ego,
    len_pre
    ):
    s_cor = x_pre - x_ego - len_pre

    return s_cor


def plot_exp(
    expTXVAS, 
    pathOutputs,
    expName,
    autoOpen=False
    ):
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.02)
    for k in range(1, 6):
        if k > 1:
            fig.add_trace(go.Scatter(x=expTXVAS[0], y=expTXVAS[k+14], 
                name='Spacing_' + str(k) + ' (min: ' + str(round(min(expTXVAS[k+14]), 1)) + ' m)'),
                row=1, col=1)
    for k in range(1, 6):
        fig.add_trace(go.Scatter(x=expTXVAS[0], y=expTXVAS[k+5],
                name='Speed_' + str(k) + ' (min: ' + str(round(min(expTXVAS[k+5]), 1)) + ' m/s)'), 
                row=2, col=1)
    for k in range(1, 6):
        fig.add_trace(go.Scatter(x=expTXVAS[0], y=expTXVAS[k+10], 
                name='Accel_' + str(k) + ' (min-max: ' + str(round(min(expTXVAS[k+10]), 1)) + ', ' + str(round(max(expTXVAS[k+10]), 1)) + ' m/s2)'), 
                row=3, col=1)
    
    fig.update_xaxes(title_text='Time [s]', row=3, col=1)
    fig.update_yaxes(title_text='Spacing [m]', row=1, col=1)
    fig.update_yaxes(title_text='Speed [m/s]', row=2, col=1)
    fig.update_yaxes(title_text='Acceleration [m/s2]', row=3, col=1)
    fig.update_layout(title_text=expName+' (reconstructed)')
    plot(fig, filename=os.path.join(pathOutputs, expName+'.html'), auto_open=autoOpen)


def plot_comp(
    expTXVASorg,
    expTXVAS,
    t_x_v_col_,
    expName,
    pathOutputs,
    format
    ):
    expOutputDir = os.path.join(pathOutputs, expName)
    if not os.path.exists(expOutputDir):
        os.makedirs(expOutputDir)

    if format == 'png':
        for k in range(1, len(expTXVASorg)):
            plt.figure()
            plt.plot(expTXVASorg[0], expTXVASorg[k], label='Original')
            plt.plot(expTXVAS[0], expTXVAS[k], label='Reconstructed')
            plt.legend()
            plt.title(expName)
            plt.xlabel(t_x_v_col_[0])
            plt.ylabel(t_x_v_col_[k])
            plt.savefig(os.path.join(expOutputDir, expName+' ('+ t_x_v_col_[k] + ')'))
            plt.show()
            plt.close()
    elif format == 'html':
        for k in range(1, len(expTXVASorg)):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=expTXVASorg[0], y=expTXVASorg[k],
                                mode='lines',
                                name='Original'))
            fig.add_trace(go.Scatter(x=expTXVAS[0], y=expTXVAS[k],
                                mode='lines',
                                name='Reconstructed'))
            fig.update_layout(title=expName,
                            xaxis_title=t_x_v_col_[0],
                            yaxis_title=t_x_v_col_[k])

            plot(fig, filename=os.path.join(os.path.join(expOutputDir, expName+' ('+ t_x_v_col_[k] + ')'+'.html')), auto_open=False)


def legendre_params(
    start_position, 
    stop_postion, 
    start_speed, 
    stop_speed, 
    duration
    ):
    dleft = start_speed * duration / 2
    dright = stop_speed * duration / 2

    a = np.array([[1,1,0,0], [-1,1,1,1],[1,1,-3,3],[-1,1,6,6]]).T
    b = np.array([start_position,stop_postion, dleft, dright ])
    res = np.linalg.solve(a, b)
    v0, v1, v2, v3 = res[0],res[1],res[2],res[3]
    
    return v0, v1, v2, v3


def rescale(
    duration,
    v0, 
    v1, 
    v2, 
    v3
    ):
    points_of_interest = np.arange(0, round(duration+0.1, 1), 0.1)
    scaled_points = points_of_interest/duration*2 - 1

    p0 = np.array([1]*len(scaled_points))
    p1 = scaled_points
    p2 = (3*np.power(scaled_points,2)-1)/2
    p3 = (5*np.power(scaled_points,3)-3*scaled_points)/2

    y = v0 * p0 + v1 * p1 + v2 * p2 + v3 * p3
    
    return points_of_interest,y


def v_peak_smooth(
    x,
    t,
    dt,
    v_init,
    trackLen,
    a_lim,
    periods,
    ):

    v_ins, v_avg = curvilinear_pos2spd(
        x,
        dt,
        v_init,
        trackLen
        )
    
    v = v_avg

    for p in periods:
        v_before = v[t<p[0]]
        v_peak = v[(t>=p[0]) & (t<=p[1])]
        t_peak = t[(t>=p[0]) & (t<=p[1])]
        v_after = v[t>p[1]]

        v_peak_new = np.zeros_like(v_peak)
        v_peak_new[0] = v_peak[0]
        v_peak_new[-1] = v_peak[-1]

        i = 1
        while i < len(v_peak)-1:
            a_start_l = (v_peak[i] - v_peak[i-1]) / dt
            a_start_r = (v_peak[i+1] - v_peak[i]) / dt
            if a_lim[0] <= a_start_r <= a_lim[1]:
                v_peak_new[i] = v_peak[i]
                i += 1
            else:
                for j in range(i+2, len(v_peak)-1):
                    a_mean = (v_peak[j] - v_peak[i]) / ((j-i)*dt)
                    a_end_r = (v_peak[j+1] - v_peak[j]) / dt
                    if (a_lim[0] <= a_mean <= a_lim[1]) and (a_lim[0] <= a_end_r <= a_lim[1]):
                        break
                    else:
                        j += 1
                
                if j < len(v_peak):
                    dur = t_peak[j]-t_peak[i]
                    v0,v1,v2,v3 = legendre_params(
                        v_peak[i], 
                        v_peak[j], 
                        a_start_l, 
                        a_end_r, 
                        dur
                        )
                    points_of_interest, y = rescale(dur,v0, v1, v2, v3)
                    # plt.plot(points_of_interest, y, 'x')
                    # plt.plot(points_of_interest[:-1], np.diff(y)*10, 'x')
                    v_peak_new[i:j+1] = y
                    i = j+1
        v = np.concatenate((v_before, v_peak_new, v_after), axis=0)
    
    v_avg_new = v
    
    v_ins_new = np.zeros_like(v)
    v_ins_new[0] = v[0]
    for j in range(1, len(v)):
        v_ins_new[j] = 2*v[j] - v[j-1]
        v_ins_new[j] = max(0, v_ins_new[j])

    return v_ins_new, v_avg_new


def data_reconstruction(
    dt,
    expTX,
    t_v_init,
    veh_len_list,
    trackLen,
    expName
    ):
    # Time - X (position, m)
    newTX = np.zeros_like(expTX)
    newTX[0] = expTX[0]
    # Time - V (speed, m/s)
    newTV = np.zeros_like(expTX)
    newTV[0] = expTX[0]
    # Time - A (acceleration, m/s)
    newTA = np.zeros_like(expTX)
    newTA[0] = expTX[0]
    # Time - S (spacing, m)
    newTS = np.zeros_like(expTX[1:])
    newTS[0] = expTX[0]

    for i in range(1, len(expTX)):
        #################################
        # Calculate speed from position #
        #################################
        v_ins0, v_avg0 = curvilinear_pos2spd(
            expTX[i],
            dt,
            t_v_init[i],
            trackLen
            )

        # Speed filtering 
        v_fltr0 = spd_filter(
            v_ins0,
            fltr_window=51, 
            fltr_polyorder=3
            )

        ########################
        # Correct the position #
        ########################        
        x_cor0 = spd2pos(
            v_fltr0,
            dt,
            expTX[i][0]
            )

        ###################
        # Peak correction #
        ###################
        a_lim = np.array([-2, 1])
        if expName == 'ASta_040719_platoon9':
            periods = np.array([[258, 272]])
            v_ins1, v_avg1 = v_peak_smooth(
                x_cor0,
                expTX[0],
                dt,
                t_v_init[i],
                trackLen,
                a_lim,
                periods
            )
            # Speed filtering 
            v_fltr1 = spd_filter(
                v_ins1,
                fltr_window=51, 
                fltr_polyorder=3
                )
            # Position
            x_cor1 = spd2pos(
                v_fltr1,
                dt,
                expTX[i][0]
                )
        elif expName == 'ASta_050719_platoon2':
            periods = np.array([[1295, 1315], [1600, 1615]])
            v_ins1, v_avg1 = v_peak_smooth(
                x_cor0,
                expTX[0],
                dt,
                t_v_init[i],
                trackLen,
                a_lim,
                periods
            )
            # Speed filtering 
            v_fltr1 = spd_filter(
                v_ins1,
                fltr_window=51, 
                fltr_polyorder=3
                )
            # Position
            x_cor1 = spd2pos(
                v_fltr1,
                dt,
                expTX[i][0]
                )
        else:
            v_ins1, v_avg1 = curvilinear_pos2spd(
                x_cor0,
                dt,
                t_v_init[i],
                trackLen
            )
            # Speed filtering 
            v_fltr1 = spd_filter(
                v_ins1,
                fltr_window=51, 
                fltr_polyorder=3
                )
            # Position
            x_cor1 = spd2pos(
                v_fltr1,
                dt,
                expTX[i][0]
                )
            # pass

        ######################################
        # Reconstruct speed and acceleration #
        ######################################
        newTX[i] = x_cor1
        newTV[i] = v_fltr1
        newTA[i] = np.append(np.diff(v_fltr1) / dt, 0)

        ###########################
        # Correct the net spacing #
        ###########################
        if i > 1:
            s_cor = pos2spacing(
                newTX[i-1], 
                newTX[i], 
                veh_len_list[i-2]
            )
            newTS[i-1] = s_cor
            if min(s_cor) < 0:
                print('Warning: Negative spacing point(s) exist! Vehicle = ' + str(i) + ', MIN = ' + str(round(min(s_cor), 1)) + ' m')
        
    return np.vstack((newTX, newTV[1:], newTA[1:], newTS[1:]))