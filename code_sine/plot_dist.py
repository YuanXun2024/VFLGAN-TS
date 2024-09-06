import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

n_samples = 800

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

df = pd.read_pickle('../Dataset/data_frame_sine_normal.pkl')
data = df.sample(frac=1).reset_index(drop=True)[:]
signals = data.drop('ID', axis = 1).values

attribute_11 = signals[:, :n_samples]
attribute_21 = signals[:, n_samples:]

results_real_sin = {}
results_real_sin['attribute_11'] = fit_sin(attribute_11, n_samples)
results_real_sin['attribute_21'] = fit_sin(attribute_21, n_samples)

plt.figure(0)
plot = sns.jointplot(data=results_real_sin, x="attribute_11", y="attribute_21")
plot.set_axis_labels("attribute_11", "attribute_21", fontsize=16)
plot.ax_joint.set_xlim(0,1)
plot.ax_joint.set_ylim(0,1)
plot.ax_joint.plot([0,1], [0,1], 'b-', linewidth = 2)
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.savefig('dist_real.png', bbox_inches = 'tight')


n_groups = 2
withSD = pd.read_csv('withSD_v2.csv').iloc[:, :n_groups*n_samples].values
idxs = np.random.choice(a=len(withSD), size=1024, replace=False, p=None)
withSD = withSD[idxs, :]
attribute_11 = withSD[:, :n_samples]
attribute_21 = withSD[:, n_samples:]

results_real_sin = {}
results_real_sin['attribute_11'] = fit_sin(attribute_11, n_samples)
results_real_sin['attribute_21'] = fit_sin(attribute_21, n_samples)

plt.figure(1)
plot = sns.jointplot(data=results_real_sin, x="attribute_11", y="attribute_21")
plot.set_axis_labels("attribute_11", "attribute_21", fontsize=16)
plot.ax_joint.set_xlim(0,1)
plot.ax_joint.set_ylim(0,1)
plot.ax_joint.plot([0,1], [0,1], 'b-', linewidth = 2)
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.savefig('dist_withSD_v2.png', bbox_inches = 'tight')


withoutSD = pd.read_csv('withoutSD_v2.csv').iloc[:, :n_groups*n_samples].values
idxs = np.random.choice(a=len(withoutSD), size=1024, replace=False, p=None)
withoutSD = withoutSD[idxs, :]
attribute_11 = withoutSD[:, :n_samples]
attribute_21 = withoutSD[:, n_samples:]

results_real_sin = {}
results_real_sin['attribute_11'] = fit_sin(attribute_11, n_samples)
results_real_sin['attribute_21'] = fit_sin(attribute_21, n_samples)

plt.figure(2)
plot = sns.jointplot(data=results_real_sin, x="attribute_11", y="attribute_21")
plot.set_axis_labels("attribute_11", "attribute_21", fontsize=16)
plot.ax_joint.set_xlim(0,1)
plot.ax_joint.set_ylim(0,1)
plot.ax_joint.plot([0,1], [0,1], 'b-', linewidth = 2)
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.savefig('dist_withoutSD_v2.png', bbox_inches = 'tight')


coscigan = pd.read_csv('coscigan_v2.csv').iloc[:, :n_groups*n_samples].values
idxs = np.random.choice(a=len(coscigan), size=1024, replace=False, p=None)
coscigan = coscigan[idxs, :]
attribute_11 = coscigan[:, :n_samples]
attribute_21 = coscigan[:, n_samples:]

results_real_sin = {}
results_real_sin['attribute_11'] = fit_sin(attribute_11, n_samples)
results_real_sin['attribute_21'] = fit_sin(attribute_21, n_samples)

plt.figure(3)
plot = sns.jointplot(data=results_real_sin, x="attribute_11", y="attribute_21")
plot.set_axis_labels("attribute_11", "attribute_21", fontsize=16)
plot.ax_joint.set_xlim(0,1)
plot.ax_joint.set_ylim(0,1)
plot.ax_joint.plot([0,1], [0,1], 'b-', linewidth = 2)
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.savefig('dist_coscigan_v2.png', bbox_inches = 'tight')
