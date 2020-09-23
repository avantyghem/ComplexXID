#!/usr/bin/env python

import sys
import numpy as np
from matplotlib import pyplot as plt
import pyink as pu

if len(sys.argv) != 2:
    print("USAGE: {} som_file".format(sys.argv[0]))
    sys.exit(-1)

som_file = sys.argv[1]
plot_file = som_file.replace(".bin", "_plots")

# SOM shape: (width, height, depth)
# Neuron shape: (depth, height, width)

som = pu.SOM(som_file)
shape = som.som_shape[:2]

base_size = [10, 10]
base_size[np.argmin(shape)] = (
    base_size[np.argmax(shape)] * np.min(shape) / np.max(shape)
)
figsize = tuple(base_size)

for channel in ["radio", "ir"]:
    chan = {"radio": 0, "ir": 1}[channel]

    # Plotting the entire layer all at once
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    cmap = ax.imshow(som.data[chan, :, :], cmap="bwr")
    # fig.colorbar(cmap)
    marks_x = np.arange(0, som.som_shape[0] + 1, 1) * som.neuron_shape[1]
    marks_y = np.arange(0, som.som_shape[1] + 1, 1) * som.neuron_shape[2]
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.axvline(m - 0.5, c="k", ls="-") for m in marks_x]
    [ax.axhline(m - 0.5, c="k", ls="-") for m in marks_y]
    ax.set_xlim(xmax=marks_x[-1])
    ax.set_ylim(ymin=marks_y[-1])
    plt.savefig(f"{plot_file}_{channel}.png", dpi=shape[0] * 25)
    # plt.show()

"""
# Looping over individual neurons
fig, axes = plt.subplots(som.som_shape[0], som.som_shape[1], figsize=(10,10))

fig.subplots_adjust(hspace=0, wspace=0)
plt.setp(ax.set_xticks([]) for ax in axes.flatten())
plt.setp(ax.set_yticks([]) for ax in axes.flatten())

for x in range(som.som_shape[0]):
    for y in range(som.som_shape[1]):
        axes[y,x].imshow(som.neurons[x,y,0,:,:,chan], cmap="bwr")
plt.show()
"""

