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

final_scores = {}
adam = Adam(learning_rate=0.001)

for iteration in range(10):
    final_scores[iteration] = {}
    # real data
    X_train = {}
    X_val = {}
    X_test = {}
    y_train = {}
    y_val= {}
    y_test = {}

    for label in [0]:
        dataset_fake = {}
        data_types = ['withSD', 'withoutSD', 'coscigan']
        
        idxs = np.random.choice(a=len(withSD_0), size=3661, replace=False, p=None)
        dataset_fake['withSD'] = data_withSD_0[idxs,:]
        idxs = np.random.choice(a=len(data_withoutSD_0), size=3661, replace=False, p=None)
        dataset_fake['withoutSD'] = data_withoutSD_0[idxs,:]
        idxs = np.random.choice(a=len(data_coscigan_0), size=3661, replace=False, p=None)
        dataset_fake['coscigan'] = data_coscigan_0[idxs,:]
        data_real = data_real_0
       
        for fake_type in data_types:
            dataset_all = np.vstack([data_real, dataset_fake[fake_type]])
            target_all = np.hstack([np.zeros(len(data_real)), np.ones(len(dataset_fake[fake_type]))])
            X_train[fake_type], X_test[fake_type], y_train[fake_type], y_test[fake_type] = train_test_split(dataset_all, target_all, test_size=0.2, random_state=iteration, stratify=target_all)
            X_train[fake_type], X_val[fake_type], y_train[fake_type], y_val[fake_type] = train_test_split(X_train[fake_type], y_train[fake_type], test_size=0.2, random_state=iteration, stratify=y_train[fake_type])

        models = {}
        for data_type in data_types:
            models[data_type] = Sequential()
            models[data_type].add(LSTM(64, input_shape=(24, n_groups)))
            models[data_type].add(Dense(1, activation='sigmoid'))
        
        chk = {}
        for data_type in data_types:
            chk[data_type] = ModelCheckpoint(f'metric_discriminative/best_model_{data_type}_{iteration}.pkl', monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)

        for data_type in data_types:
            models[data_type].compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
            models[data_type].fit(X_train[data_type], y_train[data_type], batch_size=64, epochs=20, validation_data=(X_val[data_type], y_val[data_type]), callbacks=[chk[data_type]], verbose=0)

        scores = {}
        for data_type in data_types:

            #loading the model and checking accuracy on the test data
            model = load_model(f'metric_discriminative/best_model_{data_type}_{iteration}.pkl')
            test_preds_real = model.predict(X_test[data_type])

            preds_real = np.zeros(test_preds_real.shape)
            preds_real[test_preds_real > 0.5] = 1
            scores[data_type] = accuracy_score(y_test[data_type], preds_real)

        final_scores[f'iter_{iteration}'] = scores

        # print and save the final scores
        print(iteration, scores)

with open(f'metric_discriminative/final_scores.pkl', 'wb') as f:
    pickle.dump(final_scores, f)
