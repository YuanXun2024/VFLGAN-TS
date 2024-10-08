import torch
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import json
from numpy import ndarray, array, linspace, all

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

from independent_histograms import NaiveFeatureSet

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

batch_size = 1599
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class MIAttackClassifier:
    def __init__(self, model, feature="corr"):
        self.model = model
        self.feature = feature

        self.trained = False

    def train(self, data, labels, n_splits=10):
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
        scores = cross_val_score(self.model, data, labels, cv=cv)

        return scores


def main(client:str):
    if client == 'C1':
        file_name = "/IF_C1.npy"
    elif client == 'C2':
        file_name = "/IF_C2.npy"
    else:
        NotImplementedError()

    shadow_path = "intermediate_data/shadow/"
    ano_path = "intermediate_data/LOO_809/"
    print(shadow_path, ano_path, client)

    order_list = [i for i in range(128)]
    scores = []

    F = NaiveFeatureSet(ndarray)

    for _ in range(5):
        feature_shadow = []
        feature_ano = []
        random.shuffle(order_list)
        train_set = order_list[:90]
        # print(train_set)
        test_set = order_list[90:]
        # print(test_set)
        for i in train_set:
            # shadow_dir = shadow_path + str(i) + file_name
            # data = pd.read_csv(shadow_dir)
            # ano_dir = ano_path + str(i) + file_name
            # data_ano = pd.read_csv(ano_dir)
            shadow_dir = shadow_path + str(i) + file_name
            data = np.load(shadow_dir)
            ano_dir = ano_path + str(i) + file_name
            data_ano = np.load(ano_dir)

            feature_shadow += [F.extract(data)]          
            feature_ano += [F.extract(data_ano)]

        features = feature_shadow + feature_ano
        features = np.array(features)
        print(features.shape)
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
            # shadow_dir = shadow_path + str(i) + file_name
            # data = pd.read_csv(shadow_dir)
            # ano_dir = ano_path + str(i) + file_name
            # data_ano = pd.read_csv(ano_dir)
            shadow_dir = shadow_path + str(i) + file_name
            data = np.load(shadow_dir)
            ano_dir = ano_path + str(i) + file_name
            data_ano = np.load(ano_dir)
            
            feature_shadow += [F.extract(data)]          
            feature_ano += [F.extract(data_ano)]
        
        features = feature_shadow + feature_ano
        features = np.array(features)
        print(features.shape)
        labels = np.zeros(features.shape[0])
        labels[features.shape[0]//2:] = 1

        scores += [model.score(features, labels)]
    scores = np.array(scores)
    print("Accuracy:%.2f(+/- %.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)


if __name__ == "__main__":
    main('C1')
    main('C2')