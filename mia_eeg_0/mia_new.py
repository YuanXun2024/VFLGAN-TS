import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
np.random.seed(0)
import argparse
import warnings
warnings.filterwarnings('ignore')

np.random.seed(111)


def compute_acc(scores_shadow, scores_loo):
    idx = np.random.choice(a=128, size=128, replace=False, p=None)
    train_s = np.array(scores_shadow)[idx[:90]]
    test_s = np.array(scores_shadow)[idx[90:]]
    idx = np.random.choice(a=128, size=128, replace=False, p=None)
    train_l = np.array(scores_loo)[idx[:90]]
    test_l = np.array(scores_loo)[idx[90:]]

    max_train_acc = 0
    thres_idx = -1

    for i in range(90):
        thres_temp = train_s[i]
        acc_tmp = (len(np.where(train_s<=thres_temp)[0]) + len(np.where(train_l>thres_temp)[0])) / (len(train_s)+len(train_l))
        if acc_tmp > max_train_acc:
            max_train_acc = acc_tmp
            thres_idx = i

    thres = train_s[thres_idx]
    test_acc = len(np.where(test_s<=thres)[0]) + len(np.where(test_l>thres)[0])

    return test_acc / (len(test_s)+len(test_l))


def main():
    ks=[1,3,5,10,15,20,25,30,35,40]
    
    AUCs = {}
    Acc = {}
    scores = {}
    labels = {}
    scores_shadow = {}
    scores_loo = {}

    for k in ks:
        scores_shadow[k] = []
        scores_loo[k] = []
        AUCs[k] = []
        Acc[k] = []

    for i in range(10):
        for dirpath, dirnames, filenames in os.walk(shadow_folder, topdown=True, onerror=None, followlinks=False):   
            for filename in filenames:            
                data = np.load(os.path.join(shadow_folder,filename))
                idx = np.random.choice(a=data.shape[0], size=num, replace=False, p=None)
                data = data[idx,:]
                diff =  np.square(data-target).sum(axis=1)
                diff = np.sort(diff)
                
                for k in ks:
                    scores_shadow[k].append((diff[:k].sum()))

        for dirpath, dirnames, filenames in os.walk(loo_folder, topdown=True, onerror=None, followlinks=False):   
            for filename in filenames:
                data = np.load(os.path.join(loo_folder,filename))
                idx = np.random.choice(a=data.shape[0], size=num, replace=False, p=None)
                data = data[idx,:]
                diff = np.square(data-target).sum(axis=1)
                diff = np.sort(diff)
                for k in ks:
                    scores_loo[k].append((diff[:k].sum()))

        for k in ks:
            scores[k] = scores_shadow[k] + scores_loo[k]
            labels[k] = [0]*len(scores_shadow[k]) + [1]*len(scores_loo[k])

            auc_score1 = roc_auc_score(labels[k], scores[k])
            AUCs[k].append(auc_score1)
            Acc[k].append(compute_acc(scores_shadow[k], scores_loo[k]))

            # return(np.array(AUCs).mean(), np.array(AUCs).std(), np.array(test_accs).mean(), np.array(test_accs).std())
    result_mean = []
    result_std = []
    acc_mean = []
    acc_std = []
    for k in ks:
        result_mean.append(np.array(AUCs[k]).mean())
        result_std.append(np.array(AUCs[k]).std())
        acc_mean.append(np.array(Acc[k]).mean())
        acc_std.append(np.array(Acc[k]).std())

    print(LOO)
    print('mean: ', result_mean, result_mean.index(max(result_mean)), max(result_mean))
    print('std: ', result_std)
    print('acc mean: ', acc_mean, acc_mean.index(max(acc_mean)), max(acc_mean))
    print('acc std: ', acc_std)
    np.save(f'{LOO}_k_curve_mean.npy', np.array(result_mean))
    np.save(f'{LOO}_k_curve_std.npy', np.array(result_std))

if __name__=='__main__':
    
    LOOs = [737, 913, 809]   # 911: proposed method (L1); 808: proposed v4; 737: Record 1 (groundhog day); 913: Record 2.

    # LOOs = [809]
    LOOs = [223]
    LOOs = [222]

    df = pd.read_csv('../Dataset/EEG_Eye_State_ZeroOne_chop_5best_0.csv')
    num = df.shape[0]

    print(num)

    for LOO in LOOs:

        target = df.values[LOO,:]
        
        shadow_folder = 'synthetic_datasets/shadow'
        loo_folder = f'synthetic_datasets/LOO_{LOO}'

        main()
