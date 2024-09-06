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
from sklearn.metrics import accuracy_score

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='withSD')
args = parser.parse_args()

syn_data = args.dataset

n_groups_1 = 3 
n_groups_2 = 3
n_groups = n_groups_1 + n_groups_2

# load eye detection data
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# real data
real_0 = pd.read_csv('../Dataset/stock_data_24.csv').iloc[:, :n_groups*24]

# real data initalization
data_real_0 = np.zeros((len(real_0), 24, n_groups))

# real data preprocessing
for instance in real_0.itertuples():
    for i in range(n_groups):
        data_real_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

# fake data preprocessing withSD
withSD_0 = pd.read_csv('withSD.csv').iloc[:, :n_groups*24]

data_withSD_0 = np.zeros((len(withSD_0), 24, n_groups))

for instance in withSD_0.itertuples():
    for i in range(n_groups):
        data_withSD_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]
 
# fake data preprocessing withoutSD
withoutSD_0 = pd.read_csv('withoutSD.csv').iloc[:, :n_groups*24]

data_withoutSD_0 = np.zeros((len(withoutSD_0), 24, n_groups))

for instance in withoutSD_0.itertuples():
    for i in range(n_groups):
        data_withoutSD_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

# fake data preprocessing withoutSD coscigan
coscigan_0 = pd.read_csv('coscigan.csv').iloc[:, :n_groups*24]

data_coscigan_0 = np.zeros((len(coscigan_0), 24, n_groups))

for instance in coscigan_0.itertuples():
    for i in range(n_groups):
        data_coscigan_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

# fake data preprocessing withoutSD coscigan
vflgan_0 = pd.read_csv('vflgan.csv').iloc[:, :n_groups*24]

data_vflgan_0 = np.zeros((len(vflgan_0), 24, n_groups))

for instance in vflgan_0.itertuples():
    for i in range(n_groups):
        data_vflgan_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

# fake data preprocessing DP
DP_0 = pd.read_csv('DP_v2.csv').iloc[:, :n_groups*24]

data_DP_0 = np.zeros((len(DP_0), 24, n_groups))

for instance in DP_0.itertuples():
    for i in range(n_groups):
        data_DP_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

datasets = {}
datasets['real'] = data_real_0
datasets['withSD'] = data_withSD_0
datasets['withoutSD'] = data_withoutSD_0
datasets['coscigan'] = data_coscigan_0
datasets['vflgan'] = data_vflgan_0
datasets['DP'] = data_DP_0

def eval():
    adam = Adam(learning_rate=0.001)
    TRTS_mae = {}
    TRTR_mae = {}
    TSTR_mae = {}
    TSTS_mae = {}

    len_test = int(3661*0.2)
    len_val = int(3661*0.8*0.2)
    len_train = 3361-len_test-len_val

    for iter in range(10):
        TRTS_mae[iter] = {}
        TRTR_mae[iter] = {}
        TSTR_mae[iter] = {}
        TSTS_mae[iter] = {}

        data_types = ['vflgan', 'withSD', 'withoutSD', 'coscigan', 'real', 'DP']
        train_data = {}
        train_label = {}
        test_data = {}
        test_label = {}
        val_data = {}
        val_label = {}

        models = {}
        chk = {}

        for data_type in data_types:
            idxs = np.random.choice(a=len(datasets[data_type]), size=3661, replace=False, p=None)
            dataset_tmp = datasets[data_type][idxs,:,:]
            train_data[data_type] = dataset_tmp[:len_train, :23, :]
            train_label[data_type] = dataset_tmp[:len_train, 23, :]
            val_data[data_type] = dataset_tmp[len_train:len_train+len_val, :23, :]
            val_label[data_type] = dataset_tmp[len_train:len_train+len_val, 23, :]
            test_data[data_type] = dataset_tmp[len_train+len_val:len_train+len_val+len_test, :23, :]
            test_label[data_type] = dataset_tmp[len_train+len_val:len_train+len_val+len_test, 23, :]

            chk[data_type] = ModelCheckpoint(f'eval_forecast/best_model_{data_type}_{iter}.pkl', monitor='val_loss', save_best_only=True, mode='max', verbose=0)
            models[data_type] = Sequential()
            models[data_type].add(LSTM(64, input_shape=(23, n_groups)))
            models[data_type].add(Dense(n_groups, activation='linear'))
            models[data_type].compile(loss='MSE', optimizer=adam)

            models[data_type].fit(train_data[data_type], train_label[data_type], batch_size=64, epochs=100, validation_data=(val_data[data_type], val_label[data_type]), callbacks=[chk[data_type]], verbose=0)

        for data_type in data_types:
            #loading the model and checking accuracy on the test data
            model = load_model(f'eval_forecast/best_model_{data_type}_{iter}.pkl')

            if data_type == 'real':          
                for test_type in data_types:
                    if test_type == 'real':
                        preds = model.predict(test_data['real'])
                        TRTR_mae[iter] = abs(preds-test_label['real']).mean()
                    else:
                        preds = model.predict(test_data[test_type])
                        TRTS_mae[iter][test_type] = abs(preds-test_label[test_type]).mean()
            else:
                preds = model.predict(test_data['real'])
                TSTR_mae[iter][data_type] = abs(preds-test_label['real']).mean()

                preds = model.predict(test_data[data_type])
                TSTS_mae[iter][data_type] = abs(preds-test_label[data_type]).mean()

        # print and save the final scores
        with open(f'TRTR_mae.pkl', 'wb') as f:
            pickle.dump(TRTR_mae, f)
        with open(f'TRTS_mae.pkl', 'wb') as f:
            pickle.dump(TRTS_mae, f)
        with open(f'TSTR_mae.pkl', 'wb') as f:
            pickle.dump(TSTR_mae, f)
        with open(f'TSTS_mae.pkl', 'wb') as f:
            pickle.dump(TSTS_mae, f)
      
eval()
