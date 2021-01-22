#%%
import sys, os, time
import numpy as np
import pandas as pd
import multiprocessing
from itertools import product
from functools import partial
import c2art_env.sim_env.car_following.setup_utils as cf_su


#%% 
# if __name__ == "__main__":
def main(ModelID):
    ####################
    # Paths and inputs #
    ####################
    Project = 'ASTAZERO'
    Database = 'Exp'
    Precision = 5   # Decimal
    pathRoot = os.path.dirname(__file__)
    pathInput = os.path.join(pathRoot, 'Data', Project, Database)
    pathResult = os.path.join(pathRoot, 'PyResults', Project, Database)
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
        'RMSE(A)',          # GOF-1 (ID)
        'RMSE(V)',          # GOF-2
        'RMSE(S)',          # GOF-3
        'RMSPE(V)',         # GOF-4
        'RMSPE(S)',         # GOF-5
        'RMSPE(V+S)',       # GOF-6
        'RMSPE(stdV)',      # GOF-7
        'RMSPE(stdV+S)',    # GOF-8
        'NRMSE(S+V)',       # GOF-9
        'NRMSE(S+V+A)',     # GOF-10
        'U(S+V)',           # GOF-11
        'U(S+V+A)',         # GOF-12
        'RMSE(Sn+Vn)',      # GOF-13
        'RMSE(Sn+Vn+An)',   # GOF-14
    ]
    

    del Veh_Names, pathRoot, Project, Database
    #################################################
    # ID(X)s of model, vehicle, GOF, and experiment #
    #################################################
    config = cf_su.read_config(os.path.join(pathInput, 'config.txt'))

    # ModelID: {1...90} #############################
    # ModelID = config['ModelID']
    # ModelID = int(sys.argv[1])

    # VehID: {0 = all, or 2, 3, 4, 5} #############################
    VehID = config['VehicleID']
    if VehID == 0:
        Veh_IDs = [2, 3, 4, 5]
    else:
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
    optimizationStatus = 0

    for CalGOFIDX in CalGOF_IDXs:
        ################
        # Optimization #
        ################
        if optimizationStatus == 1:
            if not Algo['UseParallel']: # Non-parallel
                for CalVehID in Veh_IDs:
                    for CalExpIDX in CalExp_IDXs:

                        _, _ = cf_su.optimization(
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
                pool.map(partial(cf_su.optimization, \
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
                    cf_su.validation(
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
            pool.map(partial(cf_su.validation, \
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
    for i in list(range(1,91)):
        start_time = time.time()
        main(i)
        print('MODEL ' + str(i) + ' DONE!')
        print(time.time() - start_time)