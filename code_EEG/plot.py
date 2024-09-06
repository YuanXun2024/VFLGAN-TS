import numpy as np
import matplotlib.pyplot as plt

wd_path_SD = 'params_vfl/EEG_Eye_State_ZeroOne_chop_5best_0/with_SD/wd_average.npy'
wd_path_woSD = 'params_vfl/EEG_Eye_State_ZeroOne_chop_5best_0/without_SD/wd_average.npy'
wd_path_coscigan = 'params_coscigan/EEG_Eye_State_ZeroOne_chop_5best_0/wd_average.npy'
wd_path_vflgan = '../vflgan_eeg/params/EEG_Eye_State_ZeroOne_chop_5best_0/wd_average.npy'

wd_SD = np.load(wd_path_SD)
wd_woSD = np.load(wd_path_woSD)
wd_coscigan = np.load(wd_path_coscigan)
wd_vflgan = np.load(wd_path_vflgan)


min_wd_sd = round(wd_SD.min(),3)
min_wd_wosd = round(wd_woSD.min(),3)
min_wd_coscigan = round(wd_coscigan.min(),3)
min_wd_vflgan = round(wd_vflgan.min(),3)


epochs = [10*i for i in range(len(wd_SD))]

plt.figure(0)
plt.plot(epochs, wd_SD, label=f'VFLGAN-TS (min={min_wd_sd})')
plt.plot(epochs, wd_vflgan, label=f'VFLGAN (min={min_wd_vflgan})')

plt.plot(epochs, wd_woSD, label=f'V-CosciGAN (min={min_wd_wosd})')
plt.plot(epochs, wd_coscigan, label=f'C-CosciGAN (min={min_wd_coscigan})')
plt.xlabel('epoch', fontdict={'size': 16})
plt.ylabel('AWD', fontdict={'size': 16})
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.legend(prop = {'size':16})
plt.savefig('wd_curve_eeg_0_average.png', bbox_inches = 'tight')
