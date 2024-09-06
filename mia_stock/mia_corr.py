import torch
import pandas as pd
import numpy as np
import random
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

class Features:
    def __init__(self, data):
        self.data = data

    def extract_corr(self, flatten=True):
        df = pd.DataFrame(self.data)
        corr = df.corr()
        corr = corr.fillna(0).values
        if flatten:
            mask = np.zeros_like(corr).astype(bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            corr = corr[mask].flatten()
        return corr

    def extract_mean(self):
        return self.data.mean(axis=0)
    

class MIAttackClassifier:
    def __init__(self, model, feature="corr"):
        self.model = model
        self.feature = feature

        self.trained = False

    def train(self, data, labels, n_splits=10):
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
        scores = cross_val_score(self.model, data, labels, cv=cv)

        return scores


def main():
    # 33914 record 1; 37592 record 2;
    shadow_path = "synthetic_datasets/shadow"
    # ano_path = "generated_data/vflgan_ClipBKD/"
    ano_path = f"synthetic_datasets/LOO_{LOO}"
    # print(shadow_path, ano_path)
    order_list = [i for i in range(128)]
    scores = []
    for _ in range(5):
        feature_shadow = []
        feature_ano = []
        random.shuffle(order_list)
        train_set = order_list[:90]
        # print(train_set)
        test_set = order_list[90:]
        # print(test_set)
        for i in train_set:
            shadow_dir = os.path.join(shadow_path, f'{i}.npy')
            data = np.load(shadow_dir)
            ano_dir =  os.path.join(ano_path, f'{i}.npy')
            data_ano = np.load(ano_dir)
            for j in range(1):  
                assert(len(data) != 0)    
                idxs = np.random.choice(a=data.shape[0], size=3661, replace=False, p=None)    
                F = Features(data[idxs,:])
                
                corr = F.extract_corr()
                feature_shadow += [corr]          

                idxs = np.random.choice(a=data.shape[0], size=3661, replace=False, p=None)  
                F_a = Features(data_ano[idxs,:])
                corr_a = F_a.extract_corr()
                feature_ano += [corr_a]

        features = feature_shadow + feature_ano
        features = np.array(features)
        # print(features.shape)
        labels = np.zeros(features.shape[0])
        labels[features.shape[0]//2:] = 1

        model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, 
                          criterion='gini', max_depth=3, max_features='sqrt',
                          min_impurity_decrease=0.0,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=1000,
                          n_jobs=None, oob_score=False, random_state=0,
                          verbose=0, warm_start=False)
        # model = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, 
        #                       hidden_layer_sizes=(24, 10), random_state=0,
        #                       max_iter=1000, verbose=False, learning_rate_init=.1)
                          
        model = model.fit(features, labels)

        feature_shadow = []
        feature_ano = []

        for i in test_set:
            shadow_dir = os.path.join(shadow_path, f'{i}.npy')
            data = np.load(shadow_dir)
            ano_dir = os.path.join(ano_path, f'{i}.npy')
            data_ano = np.load(ano_dir)
            for j in range(1):    
                idxs = np.random.choice(a=data.shape[0], size=3661, replace=False, p=None)    
                F = Features(data[idxs,:])
                corr = F.extract_corr()
                feature_shadow += [corr]          

                idxs = np.random.choice(a=data.shape[0], size=3661, replace=False, p=None)  
                F_a = Features(data_ano[idxs,:])
                corr_a = F_a.extract_corr()
                feature_ano += [corr_a]
        
        features = feature_shadow + feature_ano
        features = np.array(features)
        print(features.shape)
        labels = np.zeros(features.shape[0])
        labels[features.shape[0]//2:] = 1

        scores += [model.score(features, labels)]
    scores = np.array(scores)
    print('LOO: ', LOO)
    print("Accuracy: %f(+/- %f)" % (scores.mean(), scores.std()))
    # print(scores)


if __name__ == "__main__":
    LOOs = [688, 704, 1129]   # 688: proposed method; 704: Record 1 (groundhog day); 1129: Record 2.
    LOOs = [688]
    for LOO in LOOs:
        main()