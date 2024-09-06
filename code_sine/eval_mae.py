import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)

n_groups_1 = 1
n_groups_2 = 1
n_groups = n_groups_1 + n_groups_2
n_samples = 800
f_1 = 0.01
f_2 = 0.005
f = [f_1, f_2]
x = np.arange(n_samples)
amp = np.ones([1024,1])
sine = {}
for i in range(n_groups):
    sine_tmp = np.sin(2 * np.pi * f[i] * x)
    sine[i] = amp * sine_tmp
    print(sine[i].shape)

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

# fake data preprocessing withSD
withSD = pd.read_csv('withSD_v2.csv').iloc[:, :n_groups*n_samples]
idxs = np.random.choice(a=len(withSD), size=1024, replace=False, p=None)
data_withSD = np.zeros((len(withSD), n_samples, n_groups))

for instance in withSD.itertuples():
    for i in range(n_groups):
        data_withSD[instance[0], :, i] = instance[n_samples*i+1:n_samples*(i+1)+1]

data_withSD = data_withSD[idxs,:,:]
mae_withSD = 0
mae_withSD_c = 0

for i in range(n_groups):
    data_tmp = data_withSD[:,:,i]
    amplitudes = fit_sin(data_tmp, n_samples)
    if i==0:
        amplitudes_c = amplitudes
    else:
        amplitudes_c += amplitudes
    ground_truth = amplitudes.reshape(-1,1) * sine[i]
    mae_withSD += abs(data_tmp-ground_truth).mean()
mae_withSD = mae_withSD / n_groups

amplitudes_c = amplitudes_c/n_groups
for i in range(n_groups):
    data_tmp = data_withSD[:,:,i]
    ground_truth_c = amplitudes_c.reshape(-1,1) * sine[i]
    mae_withSD_c += abs(data_tmp-ground_truth_c).mean()
mae_withSD_c = mae_withSD_c / n_groups

# fake data preprocessing withoutSD
withoutSD = pd.read_csv('withoutSD_v2.csv').iloc[:, :n_groups*n_samples]
idxs = np.random.choice(a=len(withoutSD), size=1024, replace=False, p=None)
data_withoutSD = np.zeros((len(withoutSD), n_samples, n_groups))

for instance in withoutSD.itertuples():
    for i in range(n_groups):
        data_withoutSD[instance[0], :, i] = instance[n_samples*i+1:n_samples*(i+1)+1]
        
data_withoutSD = data_withoutSD[idxs,:,:]
mae_withoutSD = 0
mae_withoutSD_c = 0
for i in range(n_groups):
    data_tmp = data_withoutSD[:,:,i]
    amplitudes = fit_sin(data_tmp, n_samples)
    if i==0:
        amplitudes_c = amplitudes
    else:
        amplitudes_c += amplitudes
    ground_truth = amplitudes.reshape(-1,1) * sine[i]
    mae_withoutSD += abs(data_tmp-ground_truth).mean()
mae_withoutSD = mae_withoutSD / n_groups

amplitudes_c = amplitudes_c/n_groups
for i in range(n_groups):
    data_tmp = data_withoutSD[:,:,i]
    ground_truth_c = amplitudes_c.reshape(-1,1) * sine[i]
    mae_withoutSD_c += abs(data_tmp-ground_truth_c).mean()
mae_withoutSD_c = mae_withoutSD_c / n_groups

# fake data preprocessing withoutSD coscigan
coscigan = pd.read_csv('coscigan_v2.csv').iloc[:, :n_groups*n_samples]
idxs = np.random.choice(a=len(coscigan), size=1024, replace=False, p=None)
data_coscigan = np.zeros((len(coscigan), n_samples, n_groups))

for instance in coscigan.itertuples():
    for i in range(n_groups):
        data_coscigan[instance[0], :, i] = instance[n_samples*i+1:n_samples*(i+1)+1]
data_coscigan = data_coscigan[idxs,:,:]
mae_coscigan = 0
mae_coscigan_c = 0

for i in range(n_groups):
    data_tmp = data_coscigan[:,:,i]
    amplitudes = fit_sin(data_tmp, n_samples)
    if i==0:
        amplitudes_c = amplitudes
    else:
        amplitudes_c += amplitudes
    ground_truth = amplitudes.reshape(-1,1) * sine[i]
    mae_coscigan += abs(data_tmp-ground_truth).mean()
mae_coscigan = mae_coscigan / n_groups

amplitudes_c = amplitudes_c/n_groups
for i in range(n_groups):
    data_tmp = data_coscigan[:,:,i]
    ground_truth_c = amplitudes_c.reshape(-1,1) * sine[i]
    mae_coscigan_c += abs(data_tmp-ground_truth_c).mean()
mae_coscigan_c = mae_coscigan_c / n_groups


print('mae_withSD:',mae_withSD, 'mae_withoutSD:', mae_withoutSD, 'mae_coscigan:', mae_coscigan)

print('mae_withSD_c:',mae_withSD_c, 'mae_withoutSD_c:', mae_withoutSD_c, 'mae_coscigan_c:', mae_coscigan_c)