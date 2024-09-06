# generate intermediate features
import pickle
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import torch

from vfl_shadow import DiscriminatorCentral


batch_size = 1024
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.manual_seed(0)
np.random.seed(0)

n_samples = 100
n_groups_1 = 3
n_groups_2 = 2
dataset = 'EEG_Eye_State_ZeroOne_chop_5best_1'

def generate_IF(param_path, data_path, epoch):
    try:
        with open('../Dataset/'+dataset+'.csv', 'rb') as fh:
            df = pd.read_csv(fh)
    except:
        with open('../Dataset/'+dataset+'.pkl', 'rb') as fh:
            df = pickle.load(fh)

    df = df.sample(frac=1.).reset_index(drop=True)

    input = Tensor(df.values)

    central_discriminator_1 = DiscriminatorCentral(n_samples = n_samples, n_groups=n_groups_1, alpha = 0.2).cuda()
    central_discriminator_2 = DiscriminatorCentral(n_samples = n_samples, n_groups=n_groups_2, alpha = 0.2).cuda()

    central_discriminator_1.load_state_dict(torch.load(param_path + f'/central_discriminator_1_epoch_{epoch}.pt'))
    central_discriminator_1.eval()
    central_discriminator_2.load_state_dict(torch.load(param_path + f'/central_discriminator_2_epoch_{epoch}.pt'))
    central_discriminator_2.eval()

    _, IF_C1 = central_discriminator_1(input[:,:n_samples*n_groups_1])
    _, IF_C2 = central_discriminator_2(input[:,n_samples*n_groups_1:])

    IF_C1 = IF_C1.cpu().detach().numpy()
    IF_C2 = IF_C2.cpu().detach().numpy()

    data_dir_1 = data_path + "/IF_C1.npy"
    data_dir_2 = data_path + "/IF_C2.npy"
    np.save(data_dir_1, IF_C1)
    np.save(data_dir_2, IF_C2)


if __name__ == "__main__":
    root = f'params_vfl/{dataset}/with_SD'
    folders = ['LOO_761', 'shadow']
    for f_1 in folders:
        path_1 = os.path.join(root, f_1)
        seeds = os.listdir(path_1)
        for seed in seeds:
            data_path = f'intermediate_data/{f_1}/{seed}'
            os.makedirs(data_path, exist_ok=True)
            param_path = os.path.join(path_1, str(seed))
            wd_file = os.path.join(param_path, 'wd_all_vfl.npy')
            wd = np.load(wd_file).tolist()
            epoch = wd.index(min(wd))*10
            generate_IF(param_path, data_path, epoch)
            