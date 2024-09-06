from vfl_shadow import LSTMGenerator

import torch
import os
import numpy as np
torch.manual_seed(0)
import warnings
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='energy_data_24')
parser.add_argument('--nepochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsamples', type=int, default=24)
parser.add_argument('--Ngroups_1', type=int, default=14)
parser.add_argument('--Ngroups_2', type=int, default=14)
parser.add_argument('--noise_len', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:3')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

dataset = args.dataset
n_groups = args.Ngroups_1 + args.Ngroups_2
n_groups_1 = args.Ngroups_1
n_groups_2 = args.Ngroups_2
n_samples = args.nsamples
noise_len = args.noise_len
num = 19711 * 16


def generateData(root):
    folders = os.listdir(root)
    # folders = ['LOO_16240']
    for f in range(len(folders)):
        file_name = folders[f]
        path_1 = os.path.join(root, folders[f])
        seeds = os.listdir(path_1)

        for seed in seeds:
            os.makedirs(f'synthetic_datasets/{file_name}', exist_ok=True)
            file_path = f'synthetic_datasets/{file_name}/{seed}.npy'
            paramfolder = os.path.join(path_1, str(seed))
            wd_file = os.path.join(paramfolder, 'wd_all_vfl.npy')
            wd = np.load(wd_file).tolist()
            epoch = wd.index(min(wd))

            generator_1 = {}
            generator_2 = {}

            for i in range(n_groups_1):
                generator_1[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
                params_path = os.path.join(paramfolder, f'Generator_1_channel_{i}_epoch_{epoch}.pt')
                generator_1[i].load_state_dict(torch.load(params_path))
                generator_1[i].eval()

            for i in range(n_groups_2):
                generator_2[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples).to(device)
                params_path = os.path.join(paramfolder, f'Generator_2_channel_{i}_epoch_{epoch}.pt')
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
            np.save(file_path, synthetic_data)

generateData('params_vfl/energy_data_24/with_SD')