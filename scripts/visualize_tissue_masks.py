#!/usr/bin/env python3

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_files", nargs="+")
options = parser.parse_args()

n = len(options.data_files)
c = int(np.ceil(n ** 0.5))
r = int(np.ceil(n / c))
fig, axs = plt.subplots(r, c)
if n == 1:
    axs = np.array([axs])
for filename, ax in zip(options.data_files, axs.flatten()):
    with h5py.File(filename, "r") as data:
        img = (data["image"][()] + 1) / 2
        mask = data["label"][()] == 1
        img[mask] = 0.25 * img[mask]
        ax.imshow(img)
plt.tight_layout()
plt.show()
