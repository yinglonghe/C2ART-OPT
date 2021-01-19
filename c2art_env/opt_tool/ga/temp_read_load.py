import numpy as np
import os


def save_temp(temp_path, t, counter, report, pop):

    if not os.path.exists(temp_path):
        os.makedirs(temp_path, exist_ok=True)

    fid1 = open(os.path.join(temp_path, 't_counter.txt'), "w")
    fid1.write('t:' + '\t' + str(t) + '\n')
    fid1.write('counter:' + '\t' + str(counter) + '\n')
    fid1.close()

    fid2 = open(os.path.join(temp_path, 'report.txt'), "w")
    for item in report:
        fid2.write(str(item) + '\n')
    fid2.close()

    np.savetxt(os.path.join(temp_path, 'pop.txt'), pop, delimiter=',')


def read_temp(temp_path):

    if os.path.exists(os.path.join(temp_path, 't_counter.txt')):
        with open(os.path.join(temp_path, 't_counter.txt')) as f:
            temp_it = {}
            for line in f:
                (col_a, col_b) = line.split()
                temp_it[col_a.replace(':', '')] = int(col_b)
        t = temp_it['t']
        counter = temp_it['counter']

    if os.path.exists(os.path.join(temp_path, 'report.txt')):
        with open(os.path.join(temp_path, 'report.txt')) as f:
            temp_re = []
            for line in f:
                col_a = line
                temp_re.append(float(col_a))
        report = temp_re

    if os.path.exists(os.path.join(temp_path, 'pop.txt')):
        pop = np.loadtxt(os.path.join(temp_path, 'pop.txt'), delimiter=',')

    return t, counter, report, pop

# parm={}
# parm['Project'] = 'ASTAZERO'
# parm['Database'] = 'Exp'
# parm['model_id'] = 1
# parm['vehicle_id'] = 2
# parm['textGOFs'] = ['RMSE(V)',
#                     'RMSE(S)',
#                     'RMSPE(V)',
#                     'RMSPE(S)',
#                     'RMSPE(V+S)',
#                     'RMSPE(stdV)',
#                     'RMSPE(stdV+S)']
# ID_Exp = 2
# ID_GOF = 1
# t = 200
# counter = 20
# report = [16.0256, 12.03564, 11.0154, 10.0564564]
# pop = np.array([np.ones(15 + 1)] * 100)+np.random.random()
# pop[2][3] = 0
# save_temp(parm, ID_Exp, ID_GOF, t, counter, report, pop)
#
# t, counter, report, pop = read_temp(parm, ID_Exp, ID_GOF)
# print('Done')