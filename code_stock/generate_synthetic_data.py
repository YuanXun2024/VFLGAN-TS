from vfl import LSTMGenerator

import torch
import numpy as np
import pandas as pd
from torch import nn
torch.manual_seed(111)
import warnings
warnings.filterwarnings('ignore')
import argparse
import pickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stock_data_24')
parser.add_argument('--nepochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsamples', type=int, default=24)
parser.add_argument('--LSTMG', type=bool, default=True)
parser.add_argument('--LSTMD', type=bool, default=False)
parser.add_argument('--criterion', type=str, default='BCE')
parser.add_argument('--Ngroups_1', type=int, default=3)
parser.add_argument('--Ngroups_2', type=int, default=3)
parser.add_argument('--noise_len', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument("--withSD", action="store_true")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

dataset = args.dataset
n_groups = args.Ngroups_1 + args.Ngroups_2
n_groups_1 = args.Ngroups_1
n_groups_2 = args.Ngroups_2
n_samples = args.nsamples
noise_len = args.noise_len
alpha = 0.2
with_SD=args.withSD
num = 3661*16

folder_0_withSD = 'params_vfl/stock_data_24/with_SD/'
folder_0_withoutSD = 'params_vfl/stock_data_24/without_SD/'
folder_0_coscigan= 'params_coscigan/stock_data_24/'

wd_all_withSD_0 = np.load(folder_0_withSD+'wd_all_vfl.npy').tolist()
wd_all_withoutSD_0 = np.load(folder_0_withoutSD+'wd_all_vfl.npy').tolist()
wd_all_coscigan_0 = np.load(folder_0_coscigan+'wd_all_vfl.npy').tolist()

# withSD_0
generator_1 = {}
generator_2 = {}
epoch = wd_all_withSD_0.index(min(wd_all_withSD_0))*10
# epoch = 90
for i in range(n_groups_1):
    generator_1[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
    params_path = f'params_vfl/stock_data_24/with_SD/Generator_1_channel_{i}_epoch_{epoch}.pt'
    generator_1[i].load_state_dict(torch.load(params_path))
    generator_1[i].eval()

for i in range(n_groups_2):
    generator_2[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
    params_path = f'params_vfl/stock_data_24/with_SD/Generator_2_channel_{i}_epoch_{epoch}.pt'
    generator_2[i].load_state_dict(torch.load(params_path))
    generator_2[i].eval()

new_noise = torch.randn((num, noise_len)).float().to(device)
synthetic_data = []
with torch.no_grad():
    for i in range(n_groups_1):
        output = generator_1[i](new_noise)
        synthetic_data.append(output)

    for i in range(n_groups_2):
        output = generator_2[i](new_noise)
        synthetic_data.append(output)

synthetic_data = torch.hstack(synthetic_data)
synthetic_data = synthetic_data.detach().cpu().numpy()
df = pd.DataFrame(synthetic_data)
df.to_csv('withSD.csv', index=False)

# withoutSD_0
generator_1 = {}
generator_2 = {}
epoch = wd_all_withoutSD_0.index(min(wd_all_withoutSD_0))*10
for i in range(n_groups_1):
    generator_1[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
    params_path = f'params_vfl/stock_data_24/without_SD/Generator_1_channel_{i}_epoch_{epoch}.pt'
    generator_1[i].load_state_dict(torch.load(params_path))
    generator_1[i].eval()

for i in range(n_groups_2):
    generator_2[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
    params_path = f'params_vfl/stock_data_24/without_SD/Generator_2_channel_{i}_epoch_{epoch}.pt'
    generator_2[i].load_state_dict(torch.load(params_path))
    generator_2[i].eval()

new_noise = torch.randn((num, noise_len)).float().to(device)
synthetic_data = []
with torch.no_grad():
    for i in range(n_groups_1):
        output = generator_1[i](new_noise)
        synthetic_data.append(output)

    for i in range(n_groups_2):
        output = generator_2[i](new_noise)
        synthetic_data.append(output)

synthetic_data = torch.hstack(synthetic_data)
synthetic_data = synthetic_data.detach().cpu().numpy()
df = pd.DataFrame(synthetic_data)
df.to_csv('withoutSD.csv', index=False)

# coscigan_0
generator = {}
epoch = wd_all_coscigan_0.index(min(wd_all_coscigan_0))*10
for i in range(n_groups):
    generator[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
    params_path = f'params_coscigan/stock_data_24/Generator_1_channel_{i}_epoch_{epoch}.pt'
    generator[i].load_state_dict(torch.load(params_path))
    generator[i].eval()

new_noise = torch.randn((num, noise_len)).float().to(device)
synthetic_data = []
with torch.no_grad():
    for i in range(n_groups):
        output = generator[i](new_noise)
        synthetic_data.append(output)

synthetic_data = torch.hstack(synthetic_data)
synthetic_data = synthetic_data.detach().cpu().numpy()
df = pd.DataFrame(synthetic_data)
df.to_csv('coscigan.csv', index=False)
