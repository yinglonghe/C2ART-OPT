import os, re
import shutil
import numpy as np
import pandas as pd
from itertools import product
from tqdm.contrib.itertools import product as product_tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot


def get_immediate_subdirectories(
    a_dir
    ):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def remove_redundant_files(
    pathFullRes, 
    pathLiteRes,
    CalGOF_IDs, 
    Model_IDs, 
    Veh_IDs, 
    CalExp_IDs, 
    GOF_Names
    ):
    for imod, iveh, iexc, igof in product_tqdm(Model_IDs, Veh_IDs, CalExp_IDs, CalGOF_IDs):
        # Remove redundant files
        src_dir_cal_txt = os.path.join(pathFullRes, 
                            'Model ' + str(imod), 
                            'VehicleID ' + str(iveh),
                            'Calibration_Report_Exp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')
        src_dir_val_txt = os.path.join(pathFullRes,
                            'Model ' + str(imod), 
                            'VehicleID ' + str(iveh),
                            'Validation_Report_CalExp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')

        dst_dir_cal_txt = os.path.join(pathLiteRes,
                            'Model ' + str(imod), 
                            'VehicleID ' + str(iveh),
                            'Calibration_Report_Exp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')
        dst_dir_val_txt = os.path.join(pathLiteRes,
                            'Model ' + str(imod), 
                            'VehicleID ' + str(iveh),
                            'Validation_Report_CalExp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')

        os.makedirs(os.path.dirname(dst_dir_cal_txt), exist_ok=True)
        shutil.copy(src_dir_cal_txt, dst_dir_cal_txt)

        os.makedirs(os.path.dirname(dst_dir_val_txt), exist_ok=True)
        shutil.copy(src_dir_val_txt, dst_dir_val_txt)

    print('Extracted calibration and validation data from ' + pathFullRes + ' to ' + pathLiteRes)


def read_parm_keys(
    pathParmKeys
    ):
    parmKeys = {
        'ModelID': [],
        'VarNum': [],
        'VarNames': []
    }
    with open(pathParmKeys) as f:
        for line in f:
            list = line.strip().split('\t')
            parmKeys['ModelID'].append(int(list[0]))
            parmKeys['VarNum'].append(int(list[1]))
            parms = []
            for name in list[2:]:
                parms.append(name)
            parmKeys['VarNames'].append(parms)

    return parmKeys


def create_res_csv(
    srcResLite,
    pathResLite,
    pathLabel,
    pathParmKeys,
    CalGOF_IDs, 
    Model_IDs, 
    Veh_IDs, 
    Exp_IDs,
    GOF_Names,
    Exp_Names
    ):
    ###########################
    # Get platoon information #
    ###########################
    expLabels = pd.read_csv(pathLabel, index_col=None)
    expLabels = expLabels[expLabels['File'].isin(Exp_Names)].reset_index(drop=True)

    expInfo =pd.DataFrame()
    expInfo['File'] = expLabels['File']
    expInfo['VehID'] = expLabels.iloc[:, 2:6].values.tolist()
    expInfo['VehA2F'] = expLabels.iloc[:, 7:11].round(3).values.tolist()
    expInfo['VehA2B'] = expLabels.iloc[:, 12:16].round(3).values.tolist()
    expInfo['VehLen'] = expLabels.iloc[:, 17:21].round(3).values.tolist()
    del pathLabel, expLabels
    
    ######################################
    # Get parameter parameters of models #
    ######################################
    parmKeys = read_parm_keys(pathParmKeys)
    #######################
    # Store data and info #
    #######################
    srcRes  = []
    idModel, idVehicle, idExpCal, idGOFCal, ordVehCal = [], [], [], [], []
    accControl, spacingPolicy, perceptionDelay, vehicleDynamics, vehicleConstraint = [], [], [], [], []
    CalParm, CalParmNames, CalGOF, ValGOFs = [], [], [], []

    for imod, iveh, iexc, igof in product_tqdm(Model_IDs, Veh_IDs, Exp_IDs, CalGOF_IDs):
        ########################################################
        # Get IDs of model, vehicle, platoon, and gof and etc. #
        ########################################################
        idModel.append(imod)
        idVehicle.append(iveh)
        idExpCal.append(iexc)
        idGOFCal.append(igof)
        ordVehCal.append(expInfo.loc[expInfo['File']==Exp_Names[iexc-1], 'VehID'].tolist()[0].index(iveh)+2)
        srcRes.append(srcResLite)
        #######################################
        # Get model configuration information #
        #######################################
        if 1 <= imod <= 18:
            accControl.append('acc_idm')
            spacingPolicy.append('d_idm_des')
        elif 19 <= imod <= 36:
            accControl.append('acc_gipps')
            spacingPolicy.append('d_none')
        elif 37 <= imod <= 54:
            accControl.append('acc_linear')
            spacingPolicy.append('d_cth')
        elif 55 <= imod <= 72:
            accControl.append('acc_linear')
            spacingPolicy.append('d_idm_des')
        elif 73 <= imod <= 90:
            accControl.append('acc_linear')
            spacingPolicy.append('d_gipps_eq')
        
        delay_class_idx = imod % 18
        if delay_class_idx == 0:
            delay_class_idx = 18
        if delay_class_idx >= 10:
            perceptionDelay.append('yes')
        else:
            perceptionDelay.append('no')
        del delay_class_idx

        veh_type_idx = imod % 9
        if veh_type_idx == 0:
            veh_type_idx = 9
        if veh_type_idx == 1:
            vehicleDynamics.append('car_none')
            vehicleConstraint.append('off')
        elif veh_type_idx == 2:
            vehicleDynamics.append('car_none')
            vehicleConstraint.append('constant')
        elif veh_type_idx == 3:
            vehicleDynamics.append('car_none')
            vehicleConstraint.append('mfc')
        elif veh_type_idx == 4:
            vehicleDynamics.append('car_linear')
            vehicleConstraint.append('off')
        elif veh_type_idx == 5:
            vehicleDynamics.append('car_linear')
            vehicleConstraint.append('constant')
        elif veh_type_idx == 6:
            vehicleDynamics.append('car_linear')
            vehicleConstraint.append('mfc')
        elif veh_type_idx == 7:
            vehicleDynamics.append('car_nonlinear')
            vehicleConstraint.append('off')
        elif veh_type_idx == 8:
            vehicleDynamics.append('car_nonlinear')
            vehicleConstraint.append('constant')
        elif veh_type_idx == 9:
            vehicleDynamics.append('car_nonlinear')
            vehicleConstraint.append('mfc')
        del veh_type_idx
        #####################################################
        # Get results of calibration GOF (1) and parameters #
        #####################################################
        with open(os.path.join(pathResLite, 
                'Model ' + str(imod), 
                'VehicleID ' + str(iveh),
                'Calibration_Report_Exp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')) as f:
            for line in f:
                line = line.strip()
                x_temp = [round(float(x), 5) for x in line.split('\t')]
        del f, line
        CalParm.append(x_temp[:-1])
        CalGOF.append(x_temp[-1])
        CalParmNames.append(parmKeys['VarNames'][imod-1])
        ######################################
        # Get results of validation GOFs (8) #
        ######################################
        val_gof_temp = []

        with open(os.path.join(pathResLite, 
                'Model ' + str(imod), 
                'VehicleID ' + str(iveh),
                'Validation_Report_CalExp_' + str(iexc) + '_' + GOF_Names[igof-1] + '.txt')) as f:
            for line in f:
                line = line.strip()
                line_list = [x for x in line.split('\t')[1:]]
                # For MATLAB results, there is a missing element in the validation GOF list when success==0
                if len(line_list) == len(GOF_Names) + 2:             
                    x_temp = [round(float(x), 5) for x in line.split('\t')[1:-2]]
                else:
                    x_temp = [round(float(x), 5) for x in line.split('\t')[1:-2] + [100000.00000]]
                val_gof_temp += x_temp
        
        ValGOFs.append(val_gof_temp)
        del x_temp, f, line, line_list, val_gof_temp
        
    del imod, iveh, iexc, igof, expInfo

    col_ = ['idModel', 'idVehicle', 'idExpCal', 
            'accControl', 'spacingPolicy', 'perceptionDelay', 'vehicleDynamics', 'vehicleConstraint',
            'ordVehCal', 'idGOFCal', 'srcRes', 
            'Cal_' + GOF_Names[CalGOF_IDs[0]-1], 'Cal_Parm', 'Parm_Names']
    for iexv, name_gof in product(Exp_IDs, GOF_Names):
        col_.append('ValExp' + str(iexv) + '_' + name_gof)

    cal_list = list(map(list, zip(*[idModel, idVehicle, idExpCal, 
                                    accControl, spacingPolicy, perceptionDelay, vehicleDynamics, vehicleConstraint,
                                    ordVehCal, idGOFCal, srcRes, 
                                    CalGOF, CalParm, CalParmNames])))
    cal_val_list = [cal_list[idx] + val for idx, val in enumerate(ValGOFs)]

    df = pd.DataFrame(data=cal_val_list, columns=col_)
    print('Convert *.txt in ' + srcResLite + ' into ' + srcResLite + '.csv in Outputs.')

    return df


def select_min_res(
    df_entire,
    Model_IDs,
    Veh_IDs,
    Exp_IDs,
    GOF_Names,
    CalGOF_IDs
    ):
    GOFName = GOF_Names[CalGOF_IDs[0]-1]

    for imod, iveh, iexc in product_tqdm(Model_IDs, Veh_IDs, Exp_IDs):
        df_pair = df_entire[(df_entire['idModel'] == imod) & 
                            (df_entire['idVehicle'] == iveh) &
                            (df_entire['idExpCal'] == iexc)]

        df_row = df_pair[df_pair['Cal_' + GOFName] == df_pair['Cal_' + GOFName].min()].reset_index(drop=True)
        if len(df_row) > 1:
            df_row = df_row[df_row.index == 0]

        try:
            df_min = pd.concat([df_min, df_row], axis=0, sort=False)
        except:
            df_min = df_row.copy()
    
    print('Select minimum results from multiple calibrations.')

    return df_min


def organize_cal_res(
    df_,
    Veh_IDs,
    Exp_IDs,
    GOF_Names,
    CalGOF_IDs
    ):
    GOFName = GOF_Names[CalGOF_IDs[0]-1]
    df = df_[['idModel', 'idExpCal', 'idVehicle', 'ordVehCal', 'Cal_'+GOFName]]

    idExpCal = []
    idVehicle = []
    ordVehCal = []
    for iveh, iexc in product_tqdm(Veh_IDs, Exp_IDs):
        df_temp = df.loc[(df['idExpCal'] == iexc) & (df['idVehicle'] == iveh), ['idModel', 'Cal_'+GOFName]].sort_values(by=['idModel'])
        df_temp.columns = ['idModel', 'VehId' + str(iveh) + '_Exp' + str(iexc)]
        try:
            df_cal = pd.merge(df_cal, df_temp, on='idModel', how='left')
        except:
            df_cal = df_temp.copy()

        idVehicle.append(iveh)
        idExpCal.append(iexc)
        ordVehCal.append(int(df.loc[(df['idExpCal'] == iexc) & (df['idVehicle'] == iveh), 'ordVehCal'].unique()))

    df_cal.set_index('idModel', inplace=True)
    df_cal = df_cal.T.add_prefix('Model_')

    df_cal.insert(loc=0, column='idVehicle', value=idVehicle)
    df_cal.insert(loc=1, column='idExpCal', value=idExpCal)
    df_cal.insert(loc=2, column='ordVehCal', value=ordVehCal)
    df_cal.insert(loc=3, column='Model_MIN', value=df_cal.iloc[:,3:].min(axis=1))
    df_cal.insert(loc=4, column='Model_MAX', value=df_cal.iloc[:,4:].max(axis=1))
    
    return df_cal


def organize_val_res_all_gof(
    df,
    Model_IDs,
    Veh_IDs,
    Exp_IDs,
    GOF_Names
    ):
    col_gofs=[]
    for iexv in Exp_IDs: 
        col_gofs.append([col for col in df.columns if 'ValExp'+str(iexv) in col])
    del iexv

    idModel = []
    idVehicle = []
    idExpCal = []
    idExpVal = []
    ordVehCal = []
    ordVehVal = []
    gof_list = []
    entire_val_list = []

    for imod, iveh, iexc, iexv in product_tqdm(Model_IDs, Veh_IDs, Exp_IDs, Exp_IDs):
        idModel.append(imod)
        idVehicle.append(iveh)
        idExpCal.append(iexc)
        idExpVal.append(iexv)
        ordVehCal.append(int(df.loc[(df['idExpCal'] == iexc) & (df['idVehicle'] == iveh), 'ordVehCal'].unique()))
        ordVehVal.append(int(df.loc[(df['idExpCal'] == iexv) & (df['idVehicle'] == iveh), 'ordVehCal'].unique()))

        gof_list.append(df.loc[(df['idModel'] == imod) & (df['idVehicle'] == iveh) & (df['idExpCal'] == iexc), col_gofs[iexv-1]].values[0].tolist())

    entire_val_list = [[idModel[i], idVehicle[i], idExpCal[i], idExpVal[i], ordVehCal[i], ordVehVal[i]] + val for i, val in enumerate(gof_list)]

    df_val_all_gofs = pd.DataFrame(entire_val_list, columns = ['idModel', 'idVehicle', 'idExpCal', 'idExpVal', 'ordVehCal', 'ordVehVal'] + GOF_Names)
    
    df_val_all_gofs.replace(100000, np.NaN, inplace=True)

    return df_val_all_gofs


def organize_val_res_single_gof(
    df_all_gofs,
    Model_IDs,
    ValGOFName
    ):
    for i in Model_IDs:
        df = df_all_gofs.loc[df_all_gofs['idModel']==i].sort_values(by=['idVehicle', 'idExpCal', 'idExpVal'])
        try:
            df_all['Model_'+str(i)] = df[ValGOFName].reset_index(drop=True)
        except:
            df_all = df[['idVehicle', 'idExpCal', 'idExpVal', 'ordVehCal', 'ordVehVal']].copy()
            df_all['Model_'+str(i)] = df[ValGOFName]

    df_all.insert(loc=5, column='Model_MIN', value=df_all.iloc[:,5:].min(axis=1))
    df_all.insert(loc=6, column='Model_MAX', value=df_all.iloc[:,6:].max(axis=1))
    df_all.insert(loc=7, column='CNT_BLK', value=df_all.iloc[:,7:].isnull().sum(axis=1))

    df_val = df_all.loc[df_all['idExpCal']!=df_all['idExpVal']]
    df_val = df_val.reset_index(drop=True)

    df_cal = df_all.loc[df_all['idExpCal']==df_all['idExpVal']]
    df_cal = df_cal.reset_index(drop=True)

    return df_val, df_cal


def organize_opt_parm(
    df,
    Model_IDs,
    Veh_IDs,
    Exp_IDs,
    GOF_Names,
    CalGOF_IDs
    ):
    idModel, keyParm, valParm = [], [], []
    for imod in Model_IDs:
        str_key = df.loc[df['idModel']==imod, 'Parm_Names'].unique()[0]
        list_key = re.split('\[\'|\'\]|\', \'', str_key)[1:-1]
        keyParm += list_key
        idModel += [imod]*len(list_key)

        val_ = []
        col_ = []
        for iveh, iexc in product(Veh_IDs, Exp_IDs):
            col_.append('Veh'+str(iveh)+'Exp'+str(iexc))

            str_val = df.loc[
                (df['idModel']==imod) & \
                (df['idVehicle']==iveh) & \
                (df['idExpCal']==iexc), 'Cal_Parm'].values[0]

            list_val = [float(x) for x in re.split('\[|\]|, ', str_val)[1:-1]]
            val_.append(list_val)
        valParm += np.array(val_).T.tolist()

    df_parm = pd.DataFrame(valParm, columns=col_)
    df_parm.insert(loc=0, column='idModel', value=idModel)
    df_parm.insert(loc=1, column='keyParm', value=keyParm)

    return df_parm


def plot_val_crash(
    df,
    Model_IDs,
    pathFig
    ):
    ModColName = ['Model_'+str(i) for i in Model_IDs]

    des_crash = (168-df[ModColName].describe().T['count']).astype(int).astype(str)+'/'+str(len(df))
    des_crash.to_csv(os.path.join(pathFig, 'Crash_each_model.csv'))

    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=False, 
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=Model_IDs, y=df[ModColName].isnull().sum(axis=0)/len(df),
        mode='lines+markers', 
        name='Crashes of each ModelID'),
        row=1, col=1)
    fig.update_xaxes(title_text='ModelID', row=1, col=1)
    fig.update_yaxes(title_text='% of crashes in validation CaseIDs ('+str(len(df))+' in total)', row=1, col=1)
    
    caseID = list(zip(df['idVehicle'], df['idExpCal'], df['idExpVal']))
    caseID = [str(caseID[i]) for i in range(len(caseID))]
    fig.add_trace(go.Scatter(x=caseID, y=df['CNT_BLK']/len(Model_IDs), 
        mode='lines+markers',
        name='Crashes of each validation CaseID'),
        row=2, col=1)
    fig.update_xaxes(title_text='CaseID (idVehicle, idExpCal, idExpVal) - '+str(len(df))+' in total', row=2, col=1)
    fig.update_yaxes(title_text='% of crashes of models in each validation CaseID', row=2, col=1)
    
    fig.update_layout(
        showlegend=False,
        font=dict(
            size=12
        )
    )

    plot(fig, filename=os.path.join(pathFig, 'Crashes in validation.html'), auto_open=False)


def plot_raw_res(
    df,
    Model_IDs,
    GOFName,
    pathFig
    ):
    caseID = list(zip(df['idVehicle'], df['idExpCal'], df['idExpVal']))
    caseID = [str(caseID[i]) for i in range(len(caseID))]
    fig = go.Figure()
    for i in Model_IDs:
        fig.add_trace(go.Scatter(x=caseID, y=df['Model_'+str(i)],
                    mode='lines+markers',
                    name='Model_'+str(i)))
    figName = GOFName+'-CaseID (line)'
    fig.update_layout(title=figName,
                    xaxis_title='CaseID (idVehicle, idExpCal, idExpVal) - '+str(len(df))+' in total',
                    yaxis_title=GOFName)
    plot(fig, filename=os.path.join(pathFig, figName+'.html'), auto_open=False)

    fig = go.Figure()
    for i in np.arange(len(df)):
        df_data = df[['Model_'+str(i) for i in Model_IDs]]
        fig.add_trace(go.Scatter(x=Model_IDs, y=df_data.iloc[i,:],
                    mode='lines+markers',
                    name='CaseID '+caseID[i]))
    figName = GOFName+'-ModelID (line)'
    fig.update_layout(title=figName,
                xaxis_title='ModelID',
                yaxis_title=GOFName)
    plot(fig, filename=os.path.join(pathFig, figName+'.html'), auto_open=False)

    fig = go.Figure()
    for i in Model_IDs:
        data_ = df['Model_'+str(i)]
        fig.add_trace(go.Box(y=data_, name='Model_'+str(i)))
    figName = GOFName+'-ModelID (box)'
    fig.update_layout(title=figName,
            xaxis_title='ModelID',
            yaxis_title=GOFName)
    plot(fig, filename=os.path.join(pathFig, figName+'.html'), auto_open=False)


def plot_percentage_min(
    df,
    Model_IDs,
    GOFName,
    pathFig
    ):
    fig = go.Figure()
    for i in Model_IDs:
        data_ = (df['Model_'+str(i)]-df['Model_MIN'])/df['Model_MIN']*100
        fig.add_trace(go.Box(y=data_, name='Model_'+str(i)))
    figName = GOFName+'-ModelID (box-MIN%)'
    fig.update_layout(title=figName,
            xaxis_title='ModelID',
            yaxis_title=GOFName+' (min %) '+str(len(df))+' cases without collisions for selected models')
    plot(fig, filename=os.path.join(pathFig, figName+'.html'), auto_open=False)


def plot_cal_res(
    df,
    Model_IDs,
    GOFName,
    pathFig,
    Submodel_IDs
    ):

    plot_raw_res(
        df,
        Model_IDs,
        GOFName,
        pathFig
        )

    df_nocrash = df.loc[df['CNT_BLK']==0]
    plot_percentage_min(
        df_nocrash,
        Model_IDs,
        GOFName,
        pathFig
        )

    df_des = des_res_base_rel(
        df,
        pathFig,
        GOFName,
        Submodel_IDs
        )

    return df_des


def plot_val_res(
    df,
    Model_IDs,
    GOFName,
    pathFig,
    Submodel_IDs
    ):

    plot_raw_res(
        df,
        Model_IDs,
        GOFName,
        pathFig
        )

    df_nocrash = df.loc[df['CNT_BLK']==0]
    plot_percentage_min(
        df_nocrash,
        Model_IDs,
        GOFName,
        pathFig
        )

    df_des = des_res_base_rel(
        df,
        pathFig,
        GOFName,
        Submodel_IDs
        )

    return df_des


def des_res_base_rel(
    df,
    pathFig,
    GOFName,
    Submodel_IDs
    ):
    for _, (Submodel, IDs) in enumerate(Submodel_IDs.items()):
        col_ = ['Model_'+str(i) for i in IDs]
        df_sub = df[col_].dropna(axis='index').reset_index(drop=True)
        ##############
        # Plot graph #
        ##############
        fig = go.Figure()
        for i in IDs:
            data_ = (df_sub['Model_'+str(i)]-df_sub['Model_'+str(IDs[0])])/df_sub['Model_'+str(IDs[0])]*100
            fig.add_trace(go.Box(y=data_, name='Model_'+str(i)))
        figName = GOFName+'-ModelID (box-BASE%) '+Submodel
        fig.update_layout(title=figName,
                xaxis_title='ModelID',
                yaxis_title=GOFName+' (base %) '+str(len(df_sub))+' cases without collisions for selected models')
        plot(fig, filename=os.path.join(pathFig, figName+'.html'), auto_open=False)

        #######################
        # Describe statistics #
        #######################
        df_sub_base_rel = pd.DataFrame()
        for i in IDs:
            df_sub_base_rel['Model_'+str(i)] = \
                (df_sub['Model_'+str(i)]-df_sub['Model_'+str(IDs[0])])/df_sub['Model_'+str(IDs[0])]*100

        des_sub_base_rel = df_sub_base_rel.describe().T

        try:
            des_concat = pd.concat([des_concat, des_sub_base_rel], axis=0)
        except:
            des_concat = des_sub_base_rel

    des_concat.to_csv(os.path.join(pathFig, GOFName+'_SubNoCrash_BaseRel_Des.csv'))

    return des_concat
