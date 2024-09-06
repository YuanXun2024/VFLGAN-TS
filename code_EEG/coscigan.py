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
import pickle as pickle

warnings.filterwarnings('ignore')

torch.manual_seed(111)


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--expId', type=int, default=57)
parser.add_argument('--dataset', type=str, default='EEG_Eye_State_ZeroOne_chop_5best_0')
parser.add_argument('--nepochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsamples', type=int, default=100)
# parser.add_argument('--withSD', type=bool, default=True)
parser.add_argument("--withSD", action="store_true")

parser.add_argument('--LSTMG', type=bool, default=True)
parser.add_argument('--LSTMD', type=bool, default=False)
parser.add_argument('--criterion', type=str, default='BCE')
parser.add_argument('--glr', type=float, default=0.0005)
parser.add_argument('--dlr', type=float, default=0.0005)
parser.add_argument('--cdlr', type=float, default=0.0002)
parser.add_argument('--sdlr', type=float, default=0.0002)
parser.add_argument('--Ngroups_1', type=int, default=3)
parser.add_argument('--Ngroups_2', type=int, default=2)
parser.add_argument('--real_data_fraction', type=float, default=10.0)
parser.add_argument('--SD_type', type=str, default='MLP')
parser.add_argument('--gamma_c', type=float, default=5.0)
parser.add_argument('--gamma_s', type=float, default=5.0)
parser.add_argument('--noise_len', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, n_samples, alpha):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(n_samples, 128),
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
        output_2 = self.layer_1(x)
        # output_2 = self.layer_2(output_1)
        output = self.layer_3(output_2)
        return output

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
        return out
    
class DiscriminatorSever(nn.Module):
    def __init__(self, if_dim, n_clients=2, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(if_dim*n_clients, 256),
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

class DiscriminatorCentral(nn.Module):
    def __init__(self, n_samples, n_groups, alpha):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(n_samples*n_groups, 256),
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
        return output, output_2
    
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
    
dataset = args.dataset
n_groups = args.Ngroups_1 + args.Ngroups_2
n_groups_1 = args.Ngroups_1
n_groups_2 = args.Ngroups_2
num_epochs = args.nepochs
batch_size = args.batch_size
n_samples = args.nsamples
real_data_fraction = 1
criterion='BCE'
with_SD=args.withSD
with_CD=True
CD_type = 'MLP'
LSTM_G=args.LSTMG
LSTM_D=args.LSTMD
SD_type = args.SD_type
generator_lr = args.glr
discriminator_lr = args.dlr
central_discriminator_lr = args.cdlr
server_discriminator_lr = args.sdlr
gamma_s_value=args.gamma_s
gamma_c_value=args.gamma_c
noise_len = args.noise_len
alpha = 0.2

# Prepare data
try:
    with open('../Dataset/'+dataset+'.csv', 'rb') as fh:
        df = pd.read_csv(fh)
except:
    with open('../Dataset/'+dataset+'.pkl', 'rb') as fh:
        df = pickle.load(fh)

data = df.sample(frac=real_data_fraction).reset_index(drop=True)

if 'ID' not in data.columns:
    data['ID'] = np.zeros(len(data))

train_id = torch.tensor(data['ID'].values.astype(np.float32))
train = torch.tensor(data.drop('ID', axis = 1).values.astype(np.float32)) 
train_tensor = torch.utils.data.TensorDataset(train, train_id) 

if with_SD:
    params_dir = f'params_coscigan/{args.dataset}'
else:
    params_dir = f'params_coscigan/{args.dataset}'
os.makedirs(params_dir, exist_ok=True)

kwargs = {'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    train_tensor, batch_size=batch_size, shuffle=True, **kwargs
)

# Define discriminators
discriminators = {}
if LSTM_D:
    for i in range(n_groups):
        discriminators[i] = LSTMDiscriminator(ts_dim=n_samples)

else:
    for i in range(n_groups):
        discriminators[i] = Discriminator(n_samples=n_samples, alpha=alpha).apply(initialize_weights)

for i in range(n_groups):
    discriminators[i].to(device) 

# Define generators
generators = {}
if LSTM_G:
    for i in range(n_groups):
        generators[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples)

else:
    for i in range(n_groups):
        generators[i] = Generator(noise_len=noise_len, n_samples=n_samples, alpha=alpha).apply(initialize_weights)

for i in range(n_groups):
    generators[i].to(device)

# Define loss
if criterion == 'BCE':
    loss_function = nn.BCELoss()
elif criterion == 'MSE':
    loss_function = nn.MSELoss()

# Define optimizers
optimizers_D = {}
for i in range(n_groups):
    optimizers_D[i] = torch.optim.Adam(discriminators[i].parameters(), lr=discriminator_lr, betas=[0.5, 0.9])

optimizers_G = {}
for i in range(n_groups):
    optimizers_G[i] = torch.optim.Adam(generators[i].parameters(), lr=generator_lr, betas=[0.5, 0.9])

if with_CD:
    if CD_type == 'LSTM':
        NotImplementedError()
    elif CD_type == 'MLP':
        central_discriminator = DiscriminatorCentral(n_samples = n_samples, n_groups=n_groups, alpha = alpha)
        central_discriminator = central_discriminator.apply(initialize_weights)
    else:
        NotImplementedError()

    central_discriminator.to(device)
    optimizer_central_discriminator = torch.optim.Adam(central_discriminator.parameters(), lr=central_discriminator_lr, betas=[0.5, 0.9])

# Training
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
        total_loss_D = 0

        for i in range(n_groups):
            optimizers_D[i].zero_grad()
            outputs_D = discriminators[i](signal_group[i].float())
            total_loss_D += loss_function(outputs_D, real_samples_labels)
            outputs_D = discriminators[i](generated_samples[i].float())
            total_loss_D += loss_function(outputs_D, generated_samples_labels)

        if with_CD:
            optimizer_central_discriminator.zero_grad()
            temp_generated = generated_samples[0]
            for j in range(1,n_groups):
                temp_generated = torch.hstack((temp_generated, generated_samples[j]))

            temp_real = signal_group[0]
            for j in range(1,n_groups):
                temp_real = torch.hstack((temp_real, signal_group[j]))

            outputs_D, inter_r = central_discriminator(temp_real)
            total_loss_D += loss_function(outputs_D, real_samples_labels)

            outputs_D, inter_f = central_discriminator(temp_generated)
            total_loss_D += loss_function(outputs_D, generated_samples_labels)
            
        total_loss_D.backward(retain_graph=False)
        for i in range(n_groups):
            optimizers_D[i].step()
        optimizer_central_discriminator.step()

        # Training the generators
        total_loss_G = 0
        # Generating samples
        generated_samples = {}
        shared_noise = torch.randn((n_signals, noise_len)).float().to(device)
        for i in range(n_groups):
            generated_samples[i] = generators[i](shared_noise).float()
            
        for i in range(n_groups):
            optimizers_G[i].zero_grad()
            outputs_G = discriminators[i](generated_samples[i])
            total_loss_G += loss_function(outputs_G, real_samples_labels)

        if with_CD:
            temp_generated = generated_samples[0]
            for j in range(1,n_groups):
                temp_generated = torch.hstack((temp_generated, generated_samples[j]))
            output, inter_f = central_discriminator(temp_generated)
            total_loss_G += gamma_c_value * loss_function(output, real_samples_labels)            

        total_loss_G.backward(retain_graph=False)
        for i in range(n_groups):
            optimizers_G[i].step()

    # Show loss
    if epoch % 10 == 0:
        for i in range(n_groups):
            torch.save(generators[i].state_dict(), params_dir + f'/Generator_1_channel_{i}_epoch_{epoch}.pt')
      
        print(f"Epoch {epoch}: Loss D.: {total_loss_D}, Loss G.: {total_loss_G};")

        new_noise = torch.randn((1024, noise_len)).float().to(device)

        # Generating samples
        generated_samples = []
        for i in range(n_groups):
            output_tmp = generators[i](new_noise).float()
            generated_samples.append(output_tmp)

        generated_samples = torch.hstack(generated_samples)
        generated_samples = generated_samples.detach().cpu().numpy()

        real_samples = train.detach().cpu().numpy()

        idxs_1 = np.random.choice(a=1024, size=128, replace=False, p=None)
        idxs_2 = np.random.choice(a=1024, size=128, replace=False, p=None)

        wd_all = 0
        for i in range(n_samples):
            cols = [i, 100+i, 200+i, 300+i, 400+i]
            input_1 = generated_samples[idxs_1, :]
            input_1 = input_1[:, cols]
            input_2 = real_samples[idxs_2, :]
            input_2 = input_2[:, cols]
            wd_all += wd_nd(input_1, input_2)
        wd_all = wd_all/n_samples
        print(f"Epoch {epoch}: WD all: {wd_all}")

        wd_alls.append(wd_all)
        
print(wd_alls.index(min(wd_alls)), min(wd_alls))
np.save(params_dir+'/wd_all_vfl.npy', np.array(wd_alls))