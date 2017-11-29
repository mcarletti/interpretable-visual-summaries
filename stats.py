import csv
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default='results/alexnet_5k_double_opt/results.csv')

args = parser.parse_args()


def read_csv(filename):
    assert os.path.exists(filename)
    return np.genfromtxt(filename, delimiter=',', dtype=None)


data = read_csv(args.csv_file)
header = data[0, 1:].astype(str)
fnames = data[1:, 0].astype(str)
data = data[1:, 1:].astype(np.float32)

'''
0       target_prob
1-4     smooth_mask_prob    smooth_drop     smooth_blurred_prob     smooth_p
5-8     sharp_mask_prob     sharp_drop      sharp_blurred_prob      sharp_p
9-12   spx_mask_prob       spx_drop        spx_blurred_prob        spx_p
'''

N = data.shape[0]
target_prob = data[:, 0]
smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p = data[:, 5], data[:, 6], data[:, 7], data[:, 8]
spx_mask_prob, spx_drop, spx_blurred_prob, spx_p = data[:, 9], data[:, 10], data[:, 11], data[:, 12]

print('Prediction drop after masking')
print('smooth:', (np.mean(smooth_drop), np.var(smooth_drop)))
print('sharp: ', (np.mean(sharp_drop), np.var(sharp_drop)))

print('\nNormalized softmax score')
print('smooth:', (np.mean(smooth_p), np.var(smooth_p)))
print('sharp: ', (np.mean(sharp_p), np.var(sharp_p)))

print('\nMinimum drop difference')
drop_diff = sharp_drop - smooth_drop
min_drop_diff = np.min(drop_diff)
msg = 'positive' if min_drop_diff >= 0. else 'negative'
print(min_drop_diff, 'is', msg)

import matplotlib.pyplot as plt


plt.figure()
plt.grid(True)
xx = np.linspace(1, N, N)
plt.title('Prediction drop after masking')
plt.plot(xx, smooth_drop, color='b', label='smooth_drop')
plt.plot(xx, sharp_drop, color='r', label='sharp_drop')
#plt.plot(xx, spx_drop, color='g', label='spx_drop')
plt.figure()
plt.grid(True)
plt.title('Drop difference (positive is good)')
plt.plot(xx, drop_diff, color='r', label='drop_diff')
plt.legend()
plt.show()

