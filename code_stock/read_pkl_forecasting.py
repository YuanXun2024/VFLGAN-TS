import pickle
import numpy as np

def getDicFromPKL(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
    return dic

def eval(trtr, trts, tstr, tsts):
    mae = []
    for i in range(10):
        mae += [trtr[i]]
    mae = np.array(mae)
    ave = mae.mean()
    std = mae.std()
    print('TRTR mae ave %f, std %f' % (ave, std))

    for d_type in ['withSD', 'withoutSD', 'coscigan', 'vflgan','DP']:
        mae = []
        for i in range(10):
            mae += [trts[i][d_type]]
        mae = np.array(mae)
        ave = mae.mean()
        std = mae.std()
        print(f'TRTS mae {d_type} ave', '%f, std %f' % (ave, std))

    for d_type in ['withSD', 'withoutSD', 'coscigan', 'vflgan','DP']:
        mae = []
        for i in range(10):
            mae += [tstr[i][d_type]]
        mae = np.array(mae)
        ave = mae.mean()
        std = mae.std()
        print(f'TSTR mae {d_type} ave', '%f, std %f' % (ave, std))

    for d_type in ['withSD', 'withoutSD', 'coscigan', 'vflgan','DP']:
        mae = []
        f1 = []
        for i in range(10):
            mae += [tsts[i][d_type]]
        mae = np.array(mae)
        ave = mae.mean()
        std = mae.std()
        print(f'TSTS mae {d_type} ave', '%f, std %f' % (ave, std))

if __name__=='__main__':
    path_TRTR = 'TRTR_mae.pkl'
    path_TRTS = 'TRTS_mae.pkl'
    path_TSTR = 'TSTR_mae.pkl'
    path_TSTS = 'TSTS_mae.pkl'
    TRTR = getDicFromPKL(path_TRTR)
    TRTS = getDicFromPKL(path_TRTS)
    TSTR = getDicFromPKL(path_TSTR)
    TSTS = getDicFromPKL(path_TSTS)

    eval(TRTR, TRTS, TSTR, TSTS)
