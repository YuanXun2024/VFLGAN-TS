import torch
import os
from torch import nn
from torch.utils.data import Dataset

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
import random
from tqdm import tqdm
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=16)
parser.add_argument('--loo', type=int, default=688)  # 688: proposed method; 396: proposed v4; 704: Record 1 (groundhog day); 1129: Record 2.

parser.add_argument('--expId', type=int, default=57)
parser.add_argument('--dataset', type=str, default='stock_data_24')
parser.add_argument('--nepochs', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nsamples', type=int, default=24)
# parser.add_argument('--withSD', type=bool, default=True)
parser.add_argument("--withSD", action="store_true")

parser.add_argument('--LSTMG', type=bool, default=True)
parser.add_argument('--LSTMD', type=bool, default=False)
parser.add_argument('--criterion', type=str, default='BCE')
parser.add_argument('--glr', type=float, default=0.0002)
parser.add_argument('--dlr', type=float, default=0.0002)
parser.add_argument('--cdlr', type=float, default=0.0001)
parser.add_argument('--sdlr', type=float, default=0.0002)
parser.add_argument('--Ngroups_1', type=int, default=3)
parser.add_argument('--Ngroups_2', type=int, default=3)
parser.add_argument('--real_data_fraction', type=float, default=10.0)
parser.add_argument('--SD_type', type=str, default='MLP')
parser.add_argument('--gamma_c', type=float, default=0)
parser.add_argument('--gamma_s', type=float, default=5.0)
parser.add_argument('--noise_len', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epsilon', type=int, default=10)

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
        return output, output_1
    
class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])  # noise shape [batchsize, 1, n_sample]
        out, _ = self.lstm(x)  # [B, hidden]
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        # out = self.sigmoid(out)
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

SENSITIVITY = 2
n_steps = int(3661 / batch_size)
if args.epsilon == 10:
    sigma = 1.51
elif args.epsilon==5:
    sigma = 3.21
else:
    NotImplementedError()

LOO = args.loo
GLOBAL_SEED = 1

def dp_mechamism(grad):
    # print(grad.shape)
    clip_norm = 1. / batch_size
    grad_input_norm = torch.norm(grad, p=2, keepdim=True)      
    clip_coef = clip_norm / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    grad = grad*clip_coef + clip_norm * SENSITIVITY * sigma * torch.randn([1]).cuda().float()
    return grad

class CustomDataset(Dataset):
    def __init__(self, data, label, delet_target=None, transform=None):
        if delet_target is None:
            self.train_set = data
            self.train_labels = label
        else:
            self.train_set = np.delete(data, delet_target, axis=0)
            self.train_labels = np.delete(label, delet_target)
        self.transform = transform

    def __getitem__(self, index):
        img = np.array(self.train_set[index])
        if self.transform is not None:
            img = self.transform(img)
        return torch.from_numpy(img)

    def __len__(self):
        return len(self.train_set)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def main(seed):
    # Prepare data
    try:
        with open('../Dataset/'+dataset+'.csv', 'rb') as fh:
            df = pd.read_csv(fh)
    except:
        with open('../Dataset/'+dataset+'.pkl', 'rb') as fh:
            df = pickle.load(fh)
    df.drop(LOO, inplace=True)
    data = df.sample(frac=real_data_fraction).reset_index(drop=True)
    print(df.shape, data.shape)

    if 'ID' not in data.columns:
        data['ID'] = np.zeros(len(data))

    train_id = torch.tensor(data['ID'].values.astype(np.float32))
    train = torch.tensor(data.drop('ID', axis = 1).values.astype(np.float32)) 
    # train_tensor = torch.utils.data.TensorDataset(train, train_id) 
    train_tensor = CustomDataset(train, train_id)

    if with_SD:
        params_dir = f'params_vfl/{args.dataset}/with_SD/LOO_{LOO}/{seed}'
    else:
        NotImplementedError()
    os.makedirs(params_dir, exist_ok=True)

    kwargs = {'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_tensor, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, **kwargs
    )

    # Define discriminators
    discriminators_1 = {}
    discriminators_2 = {}
    if LSTM_D:
        for i in range(n_groups_1):
            discriminators_1[i] = LSTMDiscriminator(ts_dim=n_samples)
        for i in range(n_groups_2):
            discriminators_2[i] = LSTMDiscriminator(ts_dim=n_samples)

    else:
        for i in range(n_groups_1):
            discriminators_1[i] = Discriminator(n_samples=n_samples, alpha=alpha).apply(initialize_weights)
        for i in range(n_groups_2):
            discriminators_2[i] = Discriminator(n_samples=n_samples, alpha=alpha).apply(initialize_weights)

    param_D1 = {}
    param_D2 = {}
    for i in range(n_groups_1):
        discriminators_1[i].to(device) 
        param_D1[i] = discriminators_1[i].parameters().__next__()
        param_D1[i].register_hook(dp_mechamism) 
    for i in range(n_groups_2):
        discriminators_2[i].to(device)  
        param_D2[i] = discriminators_2[i].parameters().__next__()
        param_D2[i].register_hook(dp_mechamism) 

    # Define generators
    generators = {}
    generators_1 = {}
    generators_2 = {}
    if LSTM_G:
        for i in range(n_groups_1):
            generators_1[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples)
        for i in range(n_groups_2):
            generators_2[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples)

    else:
        for i in range(n_groups_1):
            generators_1[i] = Generator(noise_len=noise_len, n_samples=n_samples, alpha=alpha).apply(initialize_weights)
        for i in range(n_groups_2):
            generators_2[i] = Generator(noise_len=noise_len, n_samples=n_samples, alpha=alpha).apply(initialize_weights)

    for i in range(n_groups_1):
        generators_1[i].to(device)
    for i in range(n_groups_2):
        generators_2[i].to(device)

    # Define loss
    if criterion == 'BCE':
        loss_function = nn.BCELoss()
    elif criterion == 'MSE':
        loss_function = nn.MSELoss()

    # Define optimizers
    optimizers_D_1 = {}
    optimizers_D_2 = {}
    for i in range(n_groups_1):
        optimizers_D_1[i] = torch.optim.Adam(discriminators_1[i].parameters(), lr=discriminator_lr, betas=[0.5, 0.9])
    for i in range(n_groups_2):
        optimizers_D_2[i] = torch.optim.Adam(discriminators_2[i].parameters(), lr=discriminator_lr, betas=[0.5, 0.9])

    optimizers_G_1 = {}
    optimizers_G_2 = {}
    for i in range(n_groups_1):
        optimizers_G_1[i] = torch.optim.Adam(generators_1[i].parameters(), lr=generator_lr, betas=[0.5, 0.9])
    for i in range(n_groups_2):
        optimizers_G_2[i] = torch.optim.Adam(generators_2[i].parameters(), lr=generator_lr, betas=[0.5, 0.9])

    if with_CD:
        if CD_type == 'LSTM':
            NotImplementedError()
        elif CD_type == 'MLP':
            central_discriminator_1 = DiscriminatorCentral(n_samples = n_samples, n_groups=n_groups_1, alpha = alpha)
            central_discriminator_1 = central_discriminator_1.apply(initialize_weights)
            central_discriminator_2 = DiscriminatorCentral(n_samples = n_samples, n_groups=n_groups_2, alpha = alpha)
            central_discriminator_2 = central_discriminator_2.apply(initialize_weights)
        else:
            NotImplementedError()

        central_discriminator_1.to(device)
        param_C1 = central_discriminator_1.parameters().__next__()
        param_C1.register_hook(dp_mechamism) 
        central_discriminator_2.to(device)
        param_C2 = central_discriminator_2.parameters().__next__()
        param_C2.register_hook(dp_mechamism) 
        optimizer_central_discriminator_1 = torch.optim.Adam(central_discriminator_1.parameters(), lr=central_discriminator_lr, betas=[0.5, 0.9])
        optimizer_central_discriminator_2 = torch.optim.Adam(central_discriminator_2.parameters(), lr=central_discriminator_lr, betas=[0.5, 0.9])

    if with_SD:
        server_discriminator = DiscriminatorSever(if_dim=256).apply(initialize_weights)
        server_discriminator.to(device)
        optimizer_server_discriminator = torch.optim.Adam(server_discriminator.parameters(), lr=server_discriminator_lr, betas=[0.5, 0.9])

    # Training
    wd_alls = []
    for epoch in range(num_epochs):
        for n in range(n_steps):
            index = random.sample(range(train_tensor.__len__()), batch_size)
            signals = train_tensor.__getitem__(index=index)
            signals = signals.to(device)
            n_signals = len(signals)
            
            signal_group_1 = {}
            signal_group_2 = {}
            for i in range(n_groups_1):
                signal_group_1[i] = signals[:, i*n_samples:(i+1)*n_samples]
            for i in range(n_groups_2):
                signal_group_2[i] = signals[:, (i+n_groups_1)*n_samples:(i+1+n_groups_1)*n_samples]

            shared_noise = torch.randn((n_signals, noise_len)).float().to(device)

            # Generating samples
            generated_samples_1 = {}
            generated_samples_2 = {}
            for i in range(n_groups_1):
                generated_samples_1[i] = generators_1[i](shared_noise).float()
            for i in range(n_groups_2):
                generated_samples_2[i] = generators_2[i](shared_noise).float()

            generated_samples_labels = torch.zeros((n_signals, 1)).to(device).float()
            real_samples_labels = torch.ones((n_signals, 1)).to(device).float()

            # Data for training the discriminators
            inter_features_real = []
            inter_features_fake = []

            total_loss_D = 0

            for i in range(n_groups_1):
                optimizers_D_1[i].zero_grad()
                outputs_D = discriminators_1[i](signal_group_1[i].float())
                total_loss_D += loss_function(outputs_D, real_samples_labels)
                outputs_D = discriminators_1[i](generated_samples_1[i].float())
                total_loss_D += loss_function(outputs_D, generated_samples_labels)
            for i in range(n_groups_2):
                optimizers_D_2[i].zero_grad()
                outputs_D = discriminators_2[i](signal_group_2[i].float())
                total_loss_D += loss_function(outputs_D, real_samples_labels)
                outputs_D = discriminators_2[i](generated_samples_2[i].float())
                total_loss_D += loss_function(outputs_D, generated_samples_labels)

            if with_CD:
                optimizer_central_discriminator_1.zero_grad()
                optimizer_central_discriminator_2.zero_grad()
                temp_generated_1 = generated_samples_1[0]
                for j in range(1,n_groups_1):
                    temp_generated_1 = torch.hstack((temp_generated_1, generated_samples_1[j]))
                temp_generated_2 = generated_samples_2[0]
                for j in range(1,n_groups_2):
                    temp_generated_2 = torch.hstack((temp_generated_2, generated_samples_2[j]))

                temp_real_1 = signal_group_1[0]
                for j in range(1,n_groups_1):
                    temp_real_1 = torch.hstack((temp_real_1, signal_group_1[j]))
                temp_real_2 = signal_group_2[0]
                for j in range(1,n_groups_2):
                    temp_real_2 = torch.hstack((temp_real_2, signal_group_2[j]))

                outputs_D, inter_r = central_discriminator_1(temp_real_1)
                inter_features_real.append(inter_r)
                # total_loss_D += loss_function(outputs_D, real_samples_labels)

                outputs_D, inter_f = central_discriminator_1(temp_generated_1)
                inter_features_fake.append(inter_f)
                # total_loss_D += loss_function(outputs_D, generated_samples_labels)

                outputs_D, inter_r = central_discriminator_2(temp_real_2)
                inter_features_real.append(inter_r)
                # total_loss_D += loss_function(outputs_D, real_samples_labels)

                outputs_D, inter_f = central_discriminator_2(temp_generated_2)
                inter_features_fake.append(inter_f)
                # total_loss_D += loss_function(outputs_D, generated_samples_labels)
                
            if with_SD:
                inter_feature_real = torch.hstack(inter_features_real)
                inter_feature_fake = torch.hstack(inter_features_fake)
                # print(inter_feature_real.shape)
                # Training the discriminators and central discriminator
                optimizer_server_discriminator.zero_grad()
                output_server_discriminator = server_discriminator(inter_feature_real)
                total_loss_D += loss_function(output_server_discriminator, real_samples_labels)     
                output_server_discriminator = server_discriminator(inter_feature_fake)
                total_loss_D += loss_function(output_server_discriminator, generated_samples_labels) 

                total_loss_D.backward(retain_graph=False)
                for i in range(n_groups_1):
                    optimizers_D_1[i].step()
                for i in range(n_groups_2):
                    optimizers_D_2[i].step()
                optimizer_central_discriminator_1.step()
                optimizer_central_discriminator_2.step()     
                optimizer_server_discriminator.step()    
            else:
                total_loss_D.backward(retain_graph=False)
                for i in range(n_groups_1):
                    optimizers_D_1[i].step()
                for i in range(n_groups_2):
                    optimizers_D_2[i].step()
                optimizer_central_discriminator_1.step()
                optimizer_central_discriminator_2.step()

            # Training the generators
            inter_features_G = []
            total_loss_G = 0
            # Generating samples
            generated_samples_1 = {}
            generated_samples_2 = {}
            inter_features_fake = []
            shared_noise = torch.randn((n_signals, noise_len)).float().to(device)
            for i in range(n_groups_1):
                generated_samples_1[i] = generators_1[i](shared_noise).float()
            for i in range(n_groups_2):
                generated_samples_2[i] = generators_2[i](shared_noise).float()
                
            for i in range(n_groups_1):
                optimizers_G_1[i].zero_grad()
                outputs_G = discriminators_1[i](generated_samples_1[i])
                total_loss_G += loss_function(outputs_G, real_samples_labels)
            for i in range(n_groups_2):
                optimizers_G_2[i].zero_grad()
                outputs_G = discriminators_2[i](generated_samples_2[i])
                total_loss_G += loss_function(outputs_G, real_samples_labels)

            if with_CD:
                temp_generated_1 = generated_samples_1[0]
                for j in range(1,n_groups_1):
                    temp_generated_1 = torch.hstack((temp_generated_1, generated_samples_1[j]))
                output, inter_f = central_discriminator_1(temp_generated_1)
                inter_features_G.append(inter_f)
                total_loss_G += gamma_c_value * loss_function(output, real_samples_labels)

                temp_generated_2 = generated_samples_2[0]
                for j in range(1,n_groups_2):
                    temp_generated_2 = torch.hstack((temp_generated_2, generated_samples_2[j]))
                output, inter_f = central_discriminator_2(temp_generated_2)
                inter_features_G.append(inter_f)
                total_loss_G += gamma_c_value * loss_function(output, real_samples_labels)

            if with_SD:
                inter_feature_G = torch.hstack(inter_features_G)
                output_server = server_discriminator(inter_feature_G) 
                total_loss_G += gamma_s_value * loss_function(output_server, real_samples_labels)
                

            total_loss_G.backward(retain_graph=False)
            for i in range(n_groups_1):
                optimizers_G_1[i].step()
            for i in range(n_groups_2):
                optimizers_G_2[i].step()

        # Show loss
        if epoch % 10 == 0:
            for i in range(n_groups_1):
                torch.save(generators_1[i].state_dict(), params_dir + f'/Generator_1_channel_{i}_epoch_{epoch}.pt')
            for i in range(n_groups_2):
                torch.save(generators_2[i].state_dict(), params_dir + f'/Generator_2_channel_{i}_epoch_{epoch}.pt')
            torch.save(central_discriminator_1.state_dict(), params_dir + f'/central_discriminator_1_epoch_{epoch}.pt')
            torch.save(central_discriminator_2.state_dict(), params_dir + f'/central_discriminator_2_epoch_{epoch}.pt')
        
            # print(f"Epoch {epoch}: Loss D.: {total_loss_D}, Loss G.: {total_loss_G};")

            new_noise = torch.randn((3661, noise_len)).float().to(device)

            # Generating samples
            generated_samples_1 = {}
            generated_samples_2 = {}
            generated_samples = []
            for i in range(n_groups_1):
                generated_samples_1[i] = generators_1[i](new_noise).float()
                generated_samples.append(generated_samples_1[i])
            for i in range(n_groups_2):
                generated_samples_2[i] = generators_2[i](new_noise).float()
                generated_samples.append(generated_samples_2[i])

            generated_samples = torch.hstack(generated_samples)
            generated_samples = generated_samples.detach().cpu().numpy()

            real_samples = train.detach().cpu().numpy()

            # idxs_1 = np.random.choice(a=3661, size=3072, replace=False, p=None)
            # idxs_2 = np.random.choice(a=3661, size=3072, replace=False, p=None)

            wd_all = 0
            for f in range(n_groups):
                for t in range(n_samples):
                    # cols = [i, 24+i, 48+i, 72+i, 96+i, 120+i]
                    input_1 = generated_samples[:, 24*f+t]
                    input_2 = real_samples[:, 24*f+t]
                    wd_all += wd(input_1, input_2)
            wd_all = wd_all/(n_samples*n_groups)
            # print(f"Epoch {epoch}: WD all: {wd_all}")

            wd_alls.append(wd_all)
            
    # print(wd_alls.index(min(wd_alls)), min(wd_alls))
    np.save(params_dir+'/wd_all_vfl.npy', np.array(wd_alls))

if __name__=="__main__":
    begin = args.begin
    end = args.end
    for i in range(begin, end):
        print(i)
        set_seed(i)
        main(i)
