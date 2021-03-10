#%%
import sys, os, time
import numpy as np
import pandas as pd
import multiprocessing
from itertools import product
from functools import partial
import c2art_env.sim_env.free_flow_stable.setup_utils as ffs_su


#%% 
# if __name__ == "__main__":
def main(ModelID):
    ####################
    # Paths and inputs #
    ####################
    # ProjectID = int(sys.argv[2])
    ProjectID = 1
    Project = ['ASTAZERO', 'ZalaZone_HC'][ProjectID-1]
    Database = 'ExpFreeFlow'
    Precision = 5   # Decimal
    pathRoot = os.path.dirname(__file__)
    pathInput = os.path.join(pathRoot, 'Data', Project, Database)
    pathResult = os.path.join(pathRoot, 'PyResults_2021_03_09', Project, Database)
    
    Veh_Names = [
        'Audi A8',          # C-1 (ID)
    ]

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

    df = pd.read_csv(os.path.join(pathInput, 'PyData', 'FreeFlowStableParts.csv'))
    Exp_ = [df['File'][i]+'_'+str(df['timeStart'][i])+'_'+str(df['timeEnd'][i])+'_V'+str(df['setSpd_kmph'][i]) for i in range(len(df))]
    
    Exp_Names = [[Exp_[i]] for i in range(len(Exp_))]

    # Chained simulation
    # if Project == 'ASTAZERO':
    #     Exp_Names = [[Exp_[i] for i in [0, 1, 3, 5]], [Exp_[j] for j in [2, 4, 6, 7]]]
    # elif Project == 'ZalaZone_HC':
    #     Exp_Names = [[Exp_[i] for i in [0, 1, 3]], [Exp_[j] for j in [2, 4]]]

    del Veh_Names, pathRoot, Project, Database, df, Exp_
    #################################################
    # ID(X)s of model, vehicle, GOF, and experiment #
    #################################################
    config = ffs_su.read_config(os.path.join(pathInput, 'config.txt'))

    # ModelID #############################
    # ModelID = config['ModelID']
    # ModelID = int(sys.argv[1])

    # VehID: {0 = all, or 2, 3, 4, 5} #############################
    VehID = config['VehicleID']
    Veh_IDs = [VehID]

    # ExpID: {0 = all, or 1, 2, 3, 4, 5, 6, 7} #############################
    CalExpID = 0
    # CalExpID = int(sys.argv[2])
    if CalExpID == 0:
        CalExp_IDXs = np.arange(len(Exp_Names))
    else:
        CalExp_IDXs = [CalExpID-1]

    # GOFID: {0 = all, or 1, 2, 3, 4, 5, 6, 7, 8}
    CalGOFID = config['GOF']
    if CalGOFID == 0:
        CalGOF_IDXs = np.arange(len(GOF_Names))
    else:
        CalGOF_IDXs = [CalGOFID-1]
    
    CarMap = pd.read_excel(os.path.join(pathInput, 'CarIDs.xlsx'), index_col=None, header=None)

    optimize_resistance_and_load = config['OptimizeResistanceAndLoad']

    del CalExpID, VehID, CalGOFID
    ######################
    # GA tool parameters #
    ######################
    Algo = {
        'PopulationSize': config['PopSize'],
        'MaxIteration': 10000,
        'UseParallel': config['Parallel'],
        'NumCPUs': config['NumCPUs']
    }
    del config
    ###########################
    # Optimization parameters #
    ###########################
    optimizationStatus = 1

    for CalGOFIDX in CalGOF_IDXs:
        ################
        # Optimization #
        ################
        if optimizationStatus == 1:
            if not Algo['UseParallel']: # Non-parallel
                for CalVehID in Veh_IDs:
                    for CalExpIDX in CalExp_IDXs:

                        _, _ = ffs_su.optimization(
                            [],
                            ModelID,
                            CalGOFIDX,
                            CalVehID,
                            CalExpIDX,
                            GOF_Names,
                            CarMap,
                            Exp_Names,
                            Algo,
                            pathInput,
                            pathResult,
                            Precision,
                            optimize_resistance_and_load
                        )
            if Algo['UseParallel']:     # Parallel  
                paramlist = list(product(CalExp_IDXs, Veh_IDs))
                pool = multiprocessing.Pool(processes=Algo['NumCPUs'])
                pool.map(partial(ffs_su.optimization, \
                            ModelID = ModelID,
                            CalGOFIDX = CalGOFIDX,
                            CalVehID = None, 
                            CalExpIDX = None,
                            GOF_Names = GOF_Names,
                            CarMap = CarMap,
                            Exp_Names = Exp_Names,
                            Algo = Algo,
                            pathInput = pathInput, 
                            pathResult = pathResult,
                            Precision = Precision,
                            optimize_resistance_and_load = optimize_resistance_and_load), 
                            paramlist)  
                pool.close()
                pool.join()
        ##############
        # Validation #
        ##############
        if not Algo['UseParallel']: # Non-parallel
            # start_time = time.time()
            for CalVehID in Veh_IDs:
                for CalExpIDX in CalExp_IDXs:
                    ffs_su.validation(
                        [],
                        ModelID,
                        CalGOFIDX,  # for
                        CalVehID,   # for
                        CalExpIDX,  # for
                        GOF_Names,
                        CarMap,
                        Exp_Names,
                        Algo,
                        pathInput, 
                        pathResult,
                        Precision,
                        optimize_resistance_and_load
                    )

            # print(time.time() - start_time)
            # print('ModelID='+str(ModelID)) 
        if Algo['UseParallel']:     # Parallel  
            paramlist = list(product(CalExp_IDXs, Veh_IDs))
            pool = multiprocessing.Pool(processes=Algo['NumCPUs'])
            pool.map(partial(ffs_su.validation, \
                        ModelID = ModelID,
                        CalGOFIDX = CalGOFIDX,
                        CalVehID = None, 
                        CalExpIDX = None,
                        GOF_Names = GOF_Names,
                        CarMap = CarMap,
                        Exp_Names = Exp_Names,
                        Algo = Algo,
                        pathInput = pathInput, 
                        pathResult = pathResult,
                        Precision = Precision,
                        optimize_resistance_and_load = optimize_resistance_and_load), 
                        paramlist)
            pool.close()
            pool.join()

# %%
if __name__ == "__main__":
    import tqdm

    # with multiprocessing.Pool(4) as p:
    #     modelIDs = list(range(1, 19))[::-1]
    #     r = list(tqdm.tqdm(p.imap(main, modelIDs), total=len(modelIDs)))
    
    for i in tqdm.tqdm(list(range(18,19))):
        start_time = time.time()
        main(i)
        print('MODEL ' + str(i) + ' DONE!')
        print(time.time() - start_time)