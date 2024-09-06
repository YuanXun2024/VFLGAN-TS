import pickle
import numpy as np

def getDicFromPKL(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
    return dic

def eval(trtr, trts, tstr, tsts):
    acc = []
    f1 = []
    for i in range(10):
        acc += [trtr[i]]
        f1 += [TRTR_f1[i]]
    acc = np.array(acc)
    ave = acc.mean()
    std = acc.std()

    f1 = np.array(f1)
    ave_f1 = f1.mean()
    std_f1 = f1.std()
    print('TRTR acc ave %f, std %f' % (ave, std))
    print('TRTR f1 ave %f, std %f' % (ave_f1, std_f1))

    for d_type in ['withSD', 'withoutSD', 'coscigan', 'vflgan', 'DP']:
        acc = []
        f1 = []
        for i in range(10):
            acc += [trts[i][d_type]]
            f1 += [TRTS_f1[i][d_type]]
        acc = np.array(acc)
        ave = acc.mean()
        std = acc.std()
        f1 = np.array(f1)
        ave_f1 = f1.mean()
        std_f1 = f1.std()
        print(f'TRTS acc {d_type} ave', '%f, std %f' % (ave, std))
        print(f'TRTS f1 {d_type} ave', '%f, std %f' % (ave_f1, std_f1))

    for d_type in ['withSD', 'withoutSD', 'coscigan', 'vflgan', 'DP']:
        acc = []
        f1 = []
        for i in range(10):
            acc += [tstr[i][d_type]]
            f1 += [TSTR_f1[i][d_type]]
        acc = np.array(acc)
        ave = acc.mean()
        std = acc.std()
        f1 = np.array(f1)
        ave_f1 = f1.mean()
        std_f1 = f1.std()
        print(f'TSTR acc {d_type} ave', '%f, std %f' % (ave, std))
        print(f'TSTR f1 {d_type} ave', '%f, std %f' % (ave_f1, std_f1))

    for d_type in ['withSD', 'withoutSD', 'coscigan', 'vflgan', 'DP']:
        acc = []
        f1 = []
        for i in range(10):
            acc += [tsts[i][d_type]]
            f1 += [TSTS_f1[i][d_type]]
        acc = np.array(acc)
        ave = acc.mean()
        std = acc.std()
        f1 = np.array(f1)
        ave_f1 = f1.mean()
        std_f1 = f1.std()
        print(f'TSTS acc {d_type} ave', '%f, std %f' % (ave, std))
        print(f'TSTS f1 {d_type} ave', '%f, std %f' % (ave_f1, std_f1))


if __name__=='__main__':
    path_TRTR = 'TRTR.pkl'
    path_TRTS = 'TRTS.pkl'
    path_TSTR = 'TSTR.pkl'
    path_TSTS = 'TSTS.pkl'
    TRTR = getDicFromPKL(path_TRTR)
    TRTS = getDicFromPKL(path_TRTS)
    TSTR = getDicFromPKL(path_TSTR)
    TSTS = getDicFromPKL(path_TSTS)

    path_TRTR_f1 = 'TRTR_f1.pkl'
    path_TRTS_f1 = 'TRTS_f1.pkl'
    path_TSTR_f1 = 'TSTR_f1.pkl'
    path_TSTS_f1 = 'TSTS_f1.pkl'
    TRTR_f1 = getDicFromPKL(path_TRTR_f1)
    TRTS_f1 = getDicFromPKL(path_TRTS_f1)
    TSTR_f1 = getDicFromPKL(path_TSTR_f1)
    TSTS_f1 = getDicFromPKL(path_TSTS_f1)
    eval(TRTR, TRTS, TSTR, TSTS)
