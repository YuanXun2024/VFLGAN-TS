import os
import sys
import numpy as np
from autodp import rdp_acct, rdp_bank

sys.path.insert(0, '../source')


# red-wine: 1599; (epsilon 10 delta 0.0005: batch size 16, n_steps 100, epoch 150, sigma 1.39) (epsilon 5 delta 0.0005: batch size 16, n_steps 100, epoch 150, sigma 2.348)
# white-wine: 4898; (epsilon 10 delta 0.0002: batch size 32, n_steps 150, epoch 200, sigma 1.34)  (epsilon 5 delta 0.0002: batch size 32, n_steps 150, epoch 200, sigma 2.253)
# adult: 42960; (epsilon 10 delta 1e-5: batch size 64, n_steps 671, epoch 60, sigma 0.6128); (epsilon 5 delta 1e-5: batch size 64, n_steps 671, epoch 60, sigma 0.8)
# mnist: 60000; (epsilon 10 delta 1e-5: batch size 64, n_steps 937, epoch 300, sigma 0.802);

# credit: 1000; (epsilon 10 delta 0.001: batch size 16, n_steps 100, epoch 50, sigma 1.286)
# HCVEGY: 1385; (epsilon 10 delta 0.0008: batch size 16, n_steps 100, epoch 50, sigma 1.428)

def main():
    delta = 0.0008
    batch_size = 16
    prob = batch_size / 1000  # subsampling rate
    n_steps = 100  # training iterations
    epoch = 65
    sigma = 1.428  # noise scale
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    for _ in range(1):
        acct.compose_subsampled_mechanism(func, prob, coeff=n_steps*epoch)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))


if __name__ == '__main__':
    main()
