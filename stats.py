import csv
import os
import sys
import numpy as np


def read_csv(filename):
    assert os.path.exists(filename)
    return np.genfromtxt(filename, delimiter=',', dtype=None)


data = read_csv(sys.argv[1])
head = data[0, 1:].astype(str)
pths = data[1:, 0].astype(str)
data = data[1:, 1:].astype(np.float32)

N = data.shape[0]
smooth_drop = data[:, 2]
sharp_drop = data[:, 6]
smooth_p = data[:, 4]
sharp_p = data[:, 8]

print('Prediction drop after masking')
print('smooth:', (np.mean(smooth_drop), np.var(smooth_drop)))
print('sharp: ', (np.mean(sharp_drop), np.var(sharp_drop)))

print('Normalized softmax score')
print('smooth:', (np.mean(smooth_p), np.var(smooth_p)))
print('sharp: ', (np.mean(sharp_p), np.var(sharp_p)))
