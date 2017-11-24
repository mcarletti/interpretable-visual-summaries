import csv
import os
import numpy as np


def read_csv(filename):
    assert os.path.exists(filename)
    return np.genfromtxt(filename, delimiter=',', dtype=None)


data = read_csv('results/alexnet_original/results.csv')
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


import matplotlib.pyplot as plt


plt.figure()
plt.title('Prediction drop after masking')
plt.grid(True)
xx = np.linspace(1, N, N)
plt.scatter(xx, smooth_drop, color='b', label='smooth_drop')
plt.scatter(xx, sharp_drop, color='r', label='sharp_drop')
plt.legend()
plt.show()
