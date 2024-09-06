import numpy as np
import matplotlib.pyplot as plt

wd_path_SD = 'params_vfl/EEG_Eye_State_ZeroOne_chop_5best_0/with_SD/shadow/0/wd_all_vfl.npy'

wd_SD = np.load(wd_path_SD)


min_wd_sd = round(wd_SD.min(),3)


epochs = [10*i for i in range(len(wd_SD))]

plt.figure(0)
plt.plot(epochs, wd_SD, label=f'VFLGAN-TS (min={min_wd_sd})')
plt.xlabel('epoch', fontdict={'size': 16})
plt.ylabel('AWD', fontdict={'size': 16})
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.legend(prop = {'size':16})
plt.savefig('wd_curve_eeg_0_average.png', bbox_inches = 'tight')
