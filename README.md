# c2art-car-following-env
 
# MFC-based Platoon simulation with V2V communication and distributed model predictive control

## REFERENCES

### Platoon with model predictive control

[1] [Zheng Y, Li SE, Li K, Borrelli F, Hedrick JK. Distributed model predictive control for heterogeneous vehicle platoons under unidirectional topologies. IEEE Transactions on Control Systems Technology. 2016 Aug 17;25(3):899-910.](https://ieeexplore.ieee.org/document/7546918)

[2] [Li K, Bian Y, Li SE, Xu B, Wang J. Distributed model predictive control of multi-vehicle systems with switching communication topologies. Transportation Research Part C: Emerging Technologies. 2020 Sep 1;118:102717.](https://www.sciencedirect.com/science/article/pii/S0968090X2030632X)

[3] [Li SE, Qin X, Zheng Y, Wang J, Li K, Zhang H. Distributed platoon control under topologies with complex eigenvalues: Stability analysis and controller synthesis. IEEE Transactions on Control Systems Technology. 2017 Nov 10;27(1):206-20.](https://ieeexplore.ieee.org/abstract/document/8103336)

### MFC car-following model

[4] [He Y, Makridis M, Mattas K, Fontaras G, Ciuffo B, Xu H. Introducing Electrified Vehicle Dynamics in Traffic Simulation. Transportation Research Record. 2020 Sep;2674(9):776-91.](https://journals.sagepub.com/doi/full/10.1177/0361198120931842)

[5] [Makridis M, Fontaras G, Ciuffo B, Mattas K. MFC free-flow model: Introducing vehicle dynamics in microsimulation. Transportation Research Record. 2019 Apr;2673(4):762-77.](https://journals.sagepub.com/doi/full/10.1177/0361198119838515)

## How to use

1. (Recommended) MATLAB: Run the script of 'MAIN_platoon_mpc.m' in MATLAB

2. (Not recommended) PYTHON: Run the script of 'MAIN_platoon_mpc.py' in PYTHON. (I have not yet finalized this script because the solver (scipy.optimize.minimize) is unstable and not optimal in this case.)

pyenv('Version', "C:\Users\leiti\miniconda3\envs\jrc\python.exe")

conda env export > environment.yml
