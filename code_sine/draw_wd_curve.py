from vfl import LSTMGenerator
from scipy.stats import wasserstein_distance as wd

import pandas as pd
import numpy as np
import torch
import os
import argparse
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data_frame_sine_normal')
parser.add_argument('--nepochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsamples', type=int, default=800)
parser.add_argument('--Ngroups_1', type=int, default=1)
parser.add_argument('--Ngroups_2', type=int, default=1)
parser.add_argument('--noise_len', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

dataset = args.dataset
n_groups = args.Ngroups_1 + args.Ngroups_2
n_groups_1 = args.Ngroups_1
n_groups_2 = args.Ngroups_2
n_samples = args.nsamples
noise_len = args.noise_len
num = 1024

def calculateWD_vfl(root='params_vfl/data_frame_sine_normal'):
    df = pd.read_pickle(f'../Dataset/{dataset}.pkl')
    real_samples = df.values

    path_dataset = root
    methods = os.listdir(path_dataset)
    

    for method in methods:
        paramfolder = os.path.join(path_dataset, method)
        save_path = os.path.join(paramfolder, 'wd_average.npy')
        epochs = [5*j for j in range(41)]
        wd_sum = []
        for epoch in epochs:
            generator_1 = {}
            generator_2 = {}

            for i in range(n_groups_1):
                generator_1[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
                params_path = os.path.join(paramfolder, f'Generator_0_epoch_{epoch}.pt')
                generator_1[i].load_state_dict(torch.load(params_path))
                generator_1[i].eval()

            for i in range(n_groups_2):
                generator_2[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
                params_path = os.path.join(paramfolder, f'Generator_1_epoch_{epoch}.pt')
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

            wd_all = 0
            for f in range(n_groups):
                for t in range(n_samples):
                    # cols = [i, 24+i, 48+i, 72+i, 96+i, 120+i]
                    input_1 = synthetic_data[:, n_samples*f+t]
                    input_2 = real_samples[:, n_samples*f+t]
                    wd_all += wd(input_1, input_2)
            wd_all = wd_all/(n_samples*n_groups)
            # print(f"Epoch {epoch}: WD all: {wd_all}")

            wd_sum.append(wd_all)
        np.save(save_path, np.array(wd_sum))


def calculateWD_coscigan(root='params_coscigan/data_frame_sine_normal/with_CD'):

    df = pd.read_pickle(f'../Dataset/{dataset}.pkl')
    real_samples = df.values
    
    paramfolder = root
    save_path = os.path.join(paramfolder, 'wd_average.npy')
    epochs = [5*j for j in range(41)]
    wd_sum = []
    for epoch in epochs:
        generator_1 = {}

        for i in range(n_groups):
            generator_1[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
            params_path = os.path.join(paramfolder, f'Generator_0_epoch_{epoch}.pt')
            generator_1[i].load_state_dict(torch.load(params_path))
            generator_1[i].eval()

        new_noise = torch.randn((num, noise_len)).float().to(device)
        synthetic_data = []
        with torch.no_grad():
            for i in range(n_groups):
                output = generator_1[i](new_noise)
                synthetic_data.append(output)

        synthetic_data = torch.hstack(synthetic_data)
        synthetic_data = synthetic_data.detach().cpu().numpy()

        wd_all = 0
        for f in range(n_groups):
            for t in range(n_samples):
                # cols = [i, 24+i, 48+i, 72+i, 96+i, 120+i]
                input_1 = synthetic_data[:, n_samples*f+t]
                input_2 = real_samples[:, n_samples*f+t]
                wd_all += wd(input_1, input_2)
        wd_all = wd_all/(n_samples*n_groups)
        # print(f"Epoch {epoch}: WD all: {wd_all}")

        wd_sum.append(wd_all)
    np.save(save_path, np.array(wd_sum))

if __name__ == '__main__':
    calculateWD_vfl()
    calculateWD_coscigan()
