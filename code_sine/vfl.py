import torch
import os
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from numpy.linalg import norm
from scipy.stats import wasserstein_distance as wd
from scipy.stats import wasserstein_distance_nd as wd_nd
import warnings
import argparse

warnings.filterwarnings('ignore')

torch.manual_seed(111)


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--expId', type=int, default=57)
parser.add_argument('--dataset', type=str, default='data_frame_sine_normal')
parser.add_argument('--nepochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsamples', type=int, default=800)
parser.add_argument("--withSD", action="store_true")

parser.add_argument('--LSTMG', type=bool, default=True)
parser.add_argument('--LSTMD', type=bool, default=False)
parser.add_argument('--criterion', type=str, default='BCE')
parser.add_argument('--glr', type=float, default=0.0002)
parser.add_argument('--dlr', type=float, default=0.0002)
parser.add_argument('--cdlr', type=float, default=0.0001)
parser.add_argument('--Ngroups_1', type=int, default=1)
parser.add_argument('--Ngroups_2', type=int, default=1)
parser.add_argument('--real_data_fraction', type=float, default=10.0)
parser.add_argument('--SD_type', type=str, default='MLP')
parser.add_argument('--gamma_c', type=float, default=10.0)
parser.add_argument('--gamma_s', type=float, default=10.0)
parser.add_argument('--noise_len', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def amplitude_finder_minmax(signal):
    return (torch.max(signal, dim=1).values - torch.min(signal, dim=1).values)/2

def sinfunc(t, A, w, p, c):  
    return A * np.sin(w*t + p) + c

def fit_sin(yys, n_samples):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.arange(n_samples)
    amplitudes = []
    for yy in yys:
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        
        amplitudes.append(A)
    return np.array(amplitudes)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, n_samples, alpha):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(n_samples, 256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output_1 = self.layer_1(x)
        output_2 = self.layer_2(output_1)
        output = self.layer_3(output_2)
        return output, output_1

class LSTMDiscriminator(nn.Module):
    """Discriminator with LSTM"""
    def __init__(self, ts_dim, hidden_dim=256, inter_dim=64, num_layers=1):
        super(LSTMDiscriminator, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_1 = nn.Sequential(nn.Linear(hidden_dim, inter_dim), nn.LeakyReLU(0.1))
        self.linear_2 = nn.Sequential(nn.Linear(inter_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out, _ = self.lstm(x)
        inter_feature = self.linear_1(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = self.linear_2(inter_feature)
        out = out.view(x.size(0), x.size(1))
        return out, inter_feature
    
class DiscriminatorSever(nn.Module):
    def __init__(self, if_dim, n_groups, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(if_dim*n_groups, 256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])  # noise shape [batchsize, 1, n_sample]
        out, _ = self.lstm(x)  # [B, hidden]
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), self.ts_dim) 
        return out

class Generator(nn.Module):
    def __init__(self, noise_len, n_samples, alpha=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_len, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, n_samples)
        )

    def forward(self, x):
        output = self.model(x)
        return output

def main():
    dataset = args.dataset
    df = pd.read_pickle(f'../Dataset/{dataset}.pkl')
    data = df.sample(frac=1).reset_index(drop=True)

    train_id = torch.tensor(data['ID'].values.astype(np.float32))
    train = torch.tensor(data.drop('ID', axis = 1).values.astype(np.float32)) 
    train_tensor = torch.utils.data.TensorDataset(train, train_id) 

    n_groups = args.Ngroups_1 + args.Ngroups_2
    num_epochs = args.nepochs
    batch_size = args.batch_size
    n_samples = args.nsamples
    real_data_fraction = 1
    criterion='BCE'
    with_SD=args.withSD
    LSTM_G=args.LSTMG
    LSTM_D=args.LSTMD
    SD_type = args.SD_type
    generator_lr = args.glr
    discriminator_lr = args.dlr
    central_discriminator_lr = args.cdlr
    gamma_value=args.gamma_s
    noise_len = args.noise_len
    alpha = 0.2

    p1 = np.array([0,0])
    p2 = np.array([1,1])

    denum = norm(p2-p1)

    if with_SD:
        params_dir = f'params_vfl/{args.dataset}/with_SD'
    else:
        params_dir = f'params_vfl/{args.dataset}/without_SD'
    os.makedirs(params_dir, exist_ok=True)

    kwargs = {'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_tensor, batch_size=batch_size, shuffle=True, **kwargs
    )

    results_real_sin = pd.DataFrame()

    signals = train[:1024]

    signal_1 = signals[:, :n_samples]
    signal_2 = signals[:, n_samples:]

    results_real_sin['signal_1'] = fit_sin(signal_1.detach(), n_samples)
    results_real_sin['signal_2'] = fit_sin(signal_2.detach(), n_samples)

    discriminators = {}
    if LSTM_D:
        for i in range(n_groups):
            discriminators[i] = LSTMDiscriminator(ts_dim=n_samples)

    else:
        for i in range(n_groups):
            discriminators[i] = Discriminator(n_samples=n_samples, alpha=alpha).apply(initialize_weights)

    for i in range(n_groups):
        # discriminators[i] = nn.DataParallel(discriminators[i]).to(device)
        discriminators[i].to(device)

    generators = {}
    if LSTM_G:
        for i in range(n_groups):
            generators[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples)

    else:
        for i in range(n_groups):
            generators[i] = Generator(noise_len=noise_len, n_samples=n_samples).apply(initialize_weights)

    for i in range(n_groups):
        # generators[i] = nn.DataParallel(generators[i])
        generators[i].to(device)

    gamma = [gamma_value]*num_epochs
    if criterion == 'BCE':
        loss_function = nn.BCELoss()
    elif criterion == 'MSE':
        loss_function = nn.MSELoss()

    optimizers_D = {}
    optimizers_G = {}
    for i in range(n_groups):
        optimizers_D[i] = torch.optim.Adam(discriminators[i].parameters(), lr=discriminator_lr, betas=[0.5, 0.9])
        optimizers_G[i] = torch.optim.Adam(generators[i].parameters(), lr=generator_lr, betas=[0.5, 0.9])

    if with_SD:
        if SD_type == 'LSTM':
            central_discriminator = LSTMDiscriminator(ts_dim=n_samples, num_layers=n_groups)
        elif SD_type == 'MLP':
            central_discriminator = DiscriminatorSever(if_dim=256, n_groups=n_groups)
            central_discriminator = central_discriminator.apply(initialize_weights)
        else:
            NotImplementedError()

        central_discriminator.to(device)
        optimizer_central_discriminator = torch.optim.Adam(central_discriminator.parameters(), lr=central_discriminator_lr, betas=[0.5, 0.9])

    MSEs = []
    wd_1s = []
    wd_2s = []
    wd_alls = []
    for epoch in range(num_epochs):
        for n, (signals, ID) in enumerate(train_loader):
            signals = signals.to(device)
            n_signals = len(signals)

            signal_group = {}
            for i in range(n_groups):
                signal_group[i] = signals[:, i*n_samples:(i+1)*n_samples]

            shared_noise = torch.randn((n_signals, noise_len)).float().to(device)

            # Generating samples
            generated_samples = {}
            for i in range(n_groups):
                generated_samples[i] = generators[i](shared_noise).float()

            generated_samples_labels = torch.zeros((n_signals, 1)).to(device).float()
            real_samples_labels = torch.ones((n_signals, 1)).to(device).float()

            # Data for training the discriminators
            all_samples_group = {}
            for i in range(n_groups):
                all_samples_group[i] = torch.cat(
                    (signal_group[i], generated_samples[i])
                )

            inter_features_real = []
            inter_features_fake = []

            total_loss_D = 0

            for i in range(n_groups):
                optimizers_D[i].zero_grad()
                
                outputs_D, inter_f = discriminators[i](signal_group[i])
                inter_features_real.append(inter_f)
                total_loss_D += loss_function(outputs_D, real_samples_labels)
                outputs_D, inter_f = discriminators[i](generated_samples[i].float())
                inter_features_fake.append(inter_f)
                total_loss_D += loss_function(outputs_D, generated_samples_labels)

            inter_feature_real = torch.hstack(inter_features_real)
            inter_feature_fake = torch.hstack(inter_features_fake)

            if with_SD:
                # Training the discriminators and central discriminator
                optimizer_central_discriminator.zero_grad()
                output_server_discriminator = central_discriminator(inter_feature_real)
                total_loss_D += loss_function(output_server_discriminator, real_samples_labels)     
                output_server_discriminator = central_discriminator(inter_feature_fake)
                total_loss_D += loss_function(output_server_discriminator, generated_samples_labels) 

            total_loss_D.backward(retain_graph=False)
            for i in range(n_groups):
                optimizers_D[i].step()
            
            if with_SD:
                optimizer_central_discriminator.step()

            # Training the generators
            if epoch % 1 == 0:
                outputs_G = {}
                inter_features_G = []
                total_loss_G = 0
                # Generating samples
                generated_samples = {}
                inter_features_fake = []
                shared_noise = torch.randn((n_signals, noise_len)).float().to(device)
                for i in range(n_groups):
                    generated_samples[i] = generators[i](shared_noise).float()
                    
                for i in range(n_groups):
                    optimizers_G[i].zero_grad()
                    
                    outputs_G[i], inter_f = discriminators[i](generated_samples[i])
                    inter_features_G += [inter_f]
                    total_loss_G += loss_function(outputs_G[i], real_samples_labels)
                if with_SD:
                    inter_feature_G = torch.hstack(inter_features_G)
                    output_server = central_discriminator(inter_feature_G) 
                    total_loss_G += gamma[epoch] * loss_function(output_server, real_samples_labels)
                    
                else:
                    NotImplementedError()

                total_loss_G.backward(retain_graph=False)
                for i in range(n_groups):
                    optimizers_G[i].step()

        # Show loss
        if epoch % 5 == 0:
            for i in range(n_groups):
                torch.save(generators[i].state_dict(), params_dir + f'/Generator_{i}_epoch_{epoch}.pt')
        
            print(f"Epoch {epoch}: Loss D.: {total_loss_D}, Loss G.: {total_loss_G};")

            new_noise = torch.randn((1024, noise_len)).float().to(device)

            # Generating samples
            generated_samples_1_new = generators[0](new_noise).float()
            generated_samples_2_new = generators[1](new_noise).float()

            results_sin = pd.DataFrame()

            results_sin['signal_1'] = fit_sin(generated_samples_1_new.detach().cpu(), n_samples)
            results_sin['signal_2'] = fit_sin(generated_samples_2_new.detach().cpu(), n_samples)

            results_sin['distance'] = np.zeros(len(results_sin['signal_1']))

            for index, row in results_sin.iterrows():
                results_sin['distance'][index] = norm(np.cross(p2-p1, p1-np.array([row[0], row[1]])))/denum

            MSE = results_sin['distance'].mean()
            MSEs.append(MSE)

            wd_1 = wd(results_sin['signal_1'], results_real_sin['signal_1'])
            wd_2 = wd(results_sin['signal_2'], results_real_sin['signal_2'])

            results_all = np.concatenate((results_sin['signal_1'].values.reshape(-1,1),results_sin['signal_2'].values.reshape(-1,1)), axis=1)
            real_all = np.concatenate((results_real_sin['signal_1'].values.reshape(-1,1),results_real_sin['signal_2'].values.reshape(-1,1)), axis=1)
            
            idxs_1 = np.random.choice(a=1024, size=256, replace=False, p=None)
            idxs_2 = np.random.choice(a=1024, size=256, replace=False, p=None)

            wd_all = wd_nd(results_all[idxs_1, :], real_all[idxs_2, :])

            print(f"Epoch {epoch}: MSE: {MSE}, WD signal 1: {wd_1}, WD signal 2: {wd_2}, WD all: {wd_all}")

            wd_1s.append(wd_1)
            wd_2s.append(wd_2)
            wd_alls.append(wd_all)
            
    print(wd_alls.index(min(wd_alls)), min(wd_alls))
    np.save(params_dir+'/wd_1_vfl.npy', np.array(wd_1s))
    np.save(params_dir+'/wd_2_vfl.npy', np.array(wd_2s))
    np.save(params_dir+'/wd_all_vfl.npy', np.array(wd_alls))
    np.save(params_dir+'/MSEs_vfl.npy', np.array(MSEs))

if __name__=='__main__':
    main()