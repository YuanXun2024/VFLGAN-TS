import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tqdm import trange, tqdm

import pickle
import argparse
import random
import tensorflow as tf

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.config.experimental.enable_op_determinism()

n_groups_1 = 3 
n_groups_2 = 2 
n_groups = n_groups_1 + n_groups_2

# load eye detection data
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# real data
real_0 = pd.read_csv('../Dataset/EEG_Eye_State_ZeroOne_chop_5best_0.csv').iloc[:, :n_groups*100]
real_1 = pd.read_csv('../Dataset/EEG_Eye_State_ZeroOne_chop_5best_1.csv').iloc[:, :n_groups*100]

data_real_0 = real_0.values
data_real_1 = real_1.values

real_0['label'] = 0
real_1['label'] = 1

target_real = np.hstack([np.array(real_0['label']), np.array(real_1['label'])])

data_real = np.vstack([data_real_0, data_real_1])

# fake data preprocessing withSD
fake_data = {}
withSD_0 = pd.read_csv('withSD_0_v2.csv').iloc[:, :n_groups*100]
withSD_1 = pd.read_csv('withSD_1_v2.csv').iloc[:, :n_groups*100]

data_withSD_0 = withSD_0.values
data_withSD_1 = withSD_1.values

withSD_0['label'] = 0
withSD_1['label'] = 1

target_fake = {}
target_fake['withSD'] = np.hstack([np.array(withSD_0['label']), np.array(withSD_1['label'])])
 
# fake data preprocessing withoutSD
withoutSD_0 = pd.read_csv('withoutSD_0_v2.csv').iloc[:, :n_groups*100]
withoutSD_1 = pd.read_csv('withoutSD_1_v2.csv').iloc[:, :n_groups*100]

data_withoutSD_0 = withoutSD_0.values
data_withoutSD_1 = withoutSD_1.values

withoutSD_0['label'] = 0
withoutSD_1['label'] = 1

target_fake['withoutSD'] = np.hstack([np.array(withoutSD_0['label']), np.array(withoutSD_1['label'])])

# fake data preprocessing withoutSD coscigan
coscigan_0 = pd.read_csv('coscigan_0_v2.csv').iloc[:, :n_groups*100]
coscigan_1 = pd.read_csv('coscigan_1_v2.csv').iloc[:, :n_groups*100]

data_coscigan_0 = coscigan_0.values
data_coscigan_1 = coscigan_1.values

coscigan_0['label'] = 0
coscigan_1['label'] = 1

target_fake['coscigan'] = np.hstack([np.array(coscigan_0['label']), np.array(coscigan_1['label'])])

# fake data preprocessing vflgan
vflgan_0 = pd.read_csv('vflgan_0.csv').iloc[:, :n_groups*100]
vflgan_1 = pd.read_csv('vflgan_1.csv').iloc[:, :n_groups*100]

data_vflgan_0 = vflgan_0.values
data_vflgan_1 = vflgan_1.values

vflgan_0['label'] = 0
vflgan_1['label'] = 1

target_fake['vflgan'] = np.hstack([np.array(vflgan_0['label']), np.array(vflgan_1['label'])])


# fake data preprocessing dpvflgan
DP_0 = pd.read_csv('DP_0_v2.csv').iloc[:, :n_groups*100]
DP_1 = pd.read_csv('DP_1_v2.csv').iloc[:, :n_groups*100]

data_DP_0 = DP_0.values
data_DP_1 = DP_1.values

DP_0['label'] = 0
DP_1['label'] = 1

target_fake['DP'] = np.hstack([np.array(DP_0['label']), np.array(DP_1['label'])])

TRTR = {}
TRTS = {}
TSTR = {}
TSTS = {}
TRTR_f1 = {}
TRTS_f1 = {}
TSTR_f1 = {}
TSTS_f1 = {}
adam = Adam(learning_rate=0.001)

for iteration in tqdm(range(10)):
    TRTR[iteration] = {}
    TRTS[iteration] = {}
    TSTR[iteration] = {}
    TSTS[iteration] = {}
    TRTR_f1[iteration] = {}
    TRTS_f1[iteration] = {}
    TSTR_f1[iteration] = {}
    TSTS_f1[iteration] = {}
    # real data
    X_train = {}
    X_val = {}
    X_test = {}
    y_train = {}
    y_val= {}
    y_test = {}

    X_train['real'], X_test['real'], y_train['real'], y_test['real'] = train_test_split(data_real, target_real, test_size=0.2, random_state=iteration, stratify=target_real)
    non_test_X, non_test_y = X_train['real'], y_train['real']
    X_train['real'], X_val['real'], y_train['real'], y_val['real'] = train_test_split(X_train['real'], y_train['real'], test_size=0.2, random_state=iteration, stratify=y_train['real'])

    data_types = ['vflgan' ,'withSD', 'withoutSD', 'coscigan', 'DP']

    idxs_1 = np.random.choice(a=len(withSD_0), size=1024, replace=False, p=None)
    idxs_2 = np.random.choice(a=len(withSD_1), size=1024, replace=False, p=None)
    data_0_tmp = data_withSD_0[idxs_1,:]
    data_1_tmp = data_withSD_1[idxs_2,:]
    data_tmp = np.vstack([data_0_tmp, data_1_tmp])
    target_fake['withSD'] = np.hstack([np.array(withSD_0['label'])[idxs_1], np.array(withSD_1['label'])[idxs_2]])
    fake_data['withSD'] = data_tmp

    idxs_1 = np.random.choice(a=len(withoutSD_0), size=1024, replace=False, p=None)
    idxs_2 = np.random.choice(a=len(withoutSD_1), size=1024, replace=False, p=None)
    data_0_tmp = data_withoutSD_0[idxs_1,:]
    data_1_tmp = data_withoutSD_1[idxs_2,:]
    data_tmp = np.vstack([data_0_tmp, data_1_tmp])
    target_fake['withoutSD'] = np.hstack([np.array(withoutSD_0['label'])[idxs_1], np.array(withoutSD_1['label'])[idxs_2]])
    fake_data['withoutSD'] = data_tmp

    idxs_1 = np.random.choice(a=len(coscigan_0), size=1024, replace=False, p=None)
    idxs_2 = np.random.choice(a=len(coscigan_1), size=1024, replace=False, p=None)
    data_0_tmp = data_coscigan_0[idxs_1,:]
    data_1_tmp = data_coscigan_1[idxs_2,:]
    data_tmp = np.vstack([data_0_tmp, data_1_tmp])
    target_fake['coscigan'] = np.hstack([np.array(coscigan_0['label'])[idxs_1], np.array(coscigan_1['label'])[idxs_2]])
    fake_data['coscigan'] = data_tmp

    idxs_1 = np.random.choice(a=len(coscigan_0), size=1024, replace=False, p=None)
    idxs_2 = np.random.choice(a=len(coscigan_1), size=1024, replace=False, p=None)
    data_0_tmp = data_vflgan_0[idxs_1,:]
    data_1_tmp = data_vflgan_1[idxs_2,:]
    data_tmp = np.vstack([data_0_tmp, data_1_tmp])
    target_fake['vflgan'] = np.hstack([np.array(vflgan_0['label'])[idxs_1], np.array(vflgan_1['label'])[idxs_2]])
    fake_data['vflgan'] = data_tmp
    
    idxs_1 = np.random.choice(a=len(coscigan_0), size=1024, replace=False, p=None)
    idxs_2 = np.random.choice(a=len(coscigan_1), size=1024, replace=False, p=None)
    data_0_tmp = data_DP_0[idxs_1,:]
    data_1_tmp = data_DP_1[idxs_2,:]
    data_tmp = np.vstack([data_0_tmp, data_1_tmp])
    target_fake['DP'] = np.hstack([np.array(DP_0['label'])[idxs_1], np.array(DP_1['label'])[idxs_2]])
    fake_data['DP'] = data_tmp

    for fake_type in data_types:
        X_train[fake_type], X_test[fake_type], y_train[fake_type], y_test[fake_type] = train_test_split(fake_data[fake_type],target_fake[fake_type], test_size=0.2, random_state=iteration, stratify=target_fake[fake_type])
        X_train[fake_type], X_val[fake_type], y_train[fake_type], y_val[fake_type] = train_test_split(X_train[fake_type], y_train[fake_type], test_size=0.2, random_state=iteration, stratify=y_train[fake_type])

    data_types.append('real')

    models = {}
    for data_type in data_types:
        models[data_type] = Sequential()
        models[data_type].add(Dense(64, activation='relu', input_shape=(100*n_groups,)))
        models[data_type].add(Dense(1, activation='sigmoid'))
    
    chk = {}
    for data_type in data_types:
        chk[data_type] = ModelCheckpoint(f'best_model_{data_type}_{iteration}.pkl', monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)

    for data_type in data_types:
        models[data_type].compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        models[data_type].fit(X_train[data_type], y_train[data_type], batch_size=64, epochs=100, validation_data=(X_val[data_type], y_val[data_type]), callbacks=[chk[data_type]], verbose=0)

    for data_type in data_types:
        #loading the model and checking accuracy on the test data
        model = load_model(f'best_model_{data_type}_{iteration}.pkl')

        if data_type == 'real':          
            for test_type in data_types:
                if test_type == 'real':
                    preds = model.predict(X_test['real'])
                    results = np.zeros(preds.shape)
                    results[preds > 0.5] = 1
                    acc = accuracy_score(y_test['real'], results)
                    f1 = f1_score(y_test['real'], results)
                    TRTR[iteration] = acc
                    TRTR_f1[iteration] = f1
                else:
                    preds = model.predict(X_test[test_type])
                    results = np.zeros(preds.shape)
                    results[preds > 0.5] = 1
                    acc = accuracy_score(y_test[test_type], results)
                    f1 = f1_score(y_test[test_type], results)
                    TRTS[iteration][test_type] = acc
                    TRTS_f1[iteration][test_type] = f1
        else:
            preds = model.predict(X_test['real'])
            results = np.zeros(preds.shape)
            results[preds > 0.5] = 1
            acc = accuracy_score(y_test['real'], results)
            f1 = f1_score(y_test['real'], results)
            TSTR[iteration][data_type] = acc
            TSTR_f1[iteration][data_type] = f1

            preds = model.predict(X_test[data_type])
            results = np.zeros(preds.shape)
            results[preds > 0.5] = 1
            acc = accuracy_score(y_test[data_type], results)
            f1 = f1_score(y_test[data_type], results)
            TSTS[iteration][data_type] = acc
            TSTS_f1[iteration][data_type] = f1

    # print and save the final scores
    with open(f'TRTR.pkl', 'wb') as f:
        pickle.dump(TRTR, f)
    with open(f'TRTS.pkl', 'wb') as f:
        pickle.dump(TRTS, f)
    with open(f'TSTR.pkl', 'wb') as f:
        pickle.dump(TSTR, f)
    with open(f'TSTS.pkl', 'wb') as f:
        pickle.dump(TSTS, f)

    with open(f'TRTR_f1.pkl', 'wb') as f:
        pickle.dump(TRTR_f1, f)
    with open(f'TRTS_f1.pkl', 'wb') as f:
        pickle.dump(TRTS_f1, f)
    with open(f'TSTR_f1.pkl', 'wb') as f:
        pickle.dump(TSTR_f1, f)
    with open(f'TSTS_f1.pkl', 'wb') as f:
        pickle.dump(TSTS_f1, f)
