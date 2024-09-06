import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import argparse
import random
import tensorflow as tf


# from TimeGAN repository : https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/visualization_metrics.py
def visualization (ori_data, generated_data, analysis, data_type):
    """Using PCA or tSNE for generated and original data visualization.
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    #   anal_sample_no = min([1000, len(ori_data)])
    #   idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    anal_sample_no = len(ori_data)
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
    
    #   ori_data = ori_data[idx]
    #   generated_data = generated_data[idx]
    
    no, seq_len, dim = ori_data.shape  
    
    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                            np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
        
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        
        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
    
        ax.legend()  
        # plt.title('PCA plot')
        plt.xlabel('x-pca', fontdict={'size': 16})
        plt.ylabel('y_pca', fontdict={'size': 16})
        plt.yticks(size = 14)
        plt.xticks(size = 14)
        plt.legend(prop = {'size':16})
        # plt.xlim([-3.9, 10.5])
        # save figure as png
        plt.savefig(f'visualization/{data_type}_PCA.png', bbox_inches = 'tight')


    elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
        
        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)
        
        # Plotting
        f, ax = plt.subplots(1)
        
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
    
        ax.legend()        
        # plt.title('t-SNE plot')
        plt.xlabel('x-tsne', fontdict={'size': 16})
        plt.ylabel('y_tsne', fontdict={'size': 16})
        plt.yticks(size = 14)
        plt.xticks(size = 14)
        plt.legend(prop = {'size':16})
        # save figure as png
        plt.savefig(f'visualization/{data_type}_t-SNE_plot.png', bbox_inches = 'tight')

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.config.experimental.enable_op_determinism()

path = 'visualization/'
folder = os.path.exists(path)

if not folder:                   
    os.makedirs(path)  

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='withSD')
args = parser.parse_args()

syn_data = args.dataset

n_groups_1 = 3
n_groups_2 = 3
n_groups = n_groups_1 + n_groups_2

# load eye detection data
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# real data
real_0 = pd.read_csv('../Dataset/stock_data_24.csv').iloc[:, :n_groups*24]

# real data initalization
data_real_0 = np.zeros((len(real_0), 24, n_groups))

# real data preprocessing
for instance in real_0.itertuples():
    for i in range(n_groups):
        data_real_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

idxs_1 = np.random.choice(a=len(real_0), size=1024, replace=False, p=None)

data_real_0 = data_real_0[idxs_1,:,:]
# fake data preprocessing withSD
fake_data = {}
withSD_0 = pd.read_csv('withSD.csv').iloc[:, :n_groups*24]

data_withSD_0 = np.zeros((len(withSD_0), 24, n_groups))

for instance in withSD_0.itertuples():
    for i in range(n_groups):
        data_withSD_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]
 
idxs_1 = np.random.choice(a=len(withSD_0), size=1024, replace=False, p=None)

data_withSD_0 = data_withSD_0[idxs_1,:,:]

# fake data preprocessing withoutSD
withoutSD_0 = pd.read_csv('withoutSD.csv').iloc[:, :n_groups*24]

data_withoutSD_0 = np.zeros((len(withoutSD_0), 24, n_groups))

for instance in withoutSD_0.itertuples():
    for i in range(n_groups):
        data_withoutSD_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

idxs_1 = np.random.choice(a=len(withoutSD_0), size=1024, replace=False, p=None)

data_withoutSD_0 = data_withoutSD_0[idxs_1,:,:]
# fake data preprocessing withoutSD coscigan
coscigan_0 = pd.read_csv('coscigan.csv').iloc[:, :n_groups*24]

data_coscigan_0 = np.zeros((len(coscigan_0), 24, n_groups))

for instance in coscigan_0.itertuples():
    for i in range(n_groups):
        data_coscigan_0[instance[0], :, i] = instance[24*i+1:24*(i+1)+1]

idxs_1 = np.random.choice(a=len(coscigan_0), size=1024, replace=False, p=None)

data_coscigan_0 = data_coscigan_0[idxs_1,:,:]

ana = ['pca', 'tsne']

for a in ana:
    visualization(data_real_0, data_withSD_0, analysis=a, data_type='withSD')
    visualization(data_real_0, data_withoutSD_0, analysis=a, data_type='withoutSD')
    visualization(data_real_0, data_coscigan_0, analysis=a, data_type='coscigan')