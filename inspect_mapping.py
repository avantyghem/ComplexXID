#!/usr/bin/env python

import sys
import string
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils as pu
from collections import Counter

base = "FIRST_only"
imread = pu.ImageReader(f"{base}_imgs.bin")
df = pd.read_csv(f"{base}.csv")
mapping = pu.Mapping(f"{base}/som_5_Similarity.bin")
som_height = mapping.som_shape[0]
som_shape = mapping.som_shape

ind = mapping.best_matching_neuron
df["best_match_ind"] = tuple(np.transpose(ind))
df["best_match"] = mapping.best_matching_letter_code
df["best_match_distance"] = mapping.data[tuple([np.arange(mapping.header[3]), *ind])]

i = np.random.randint(0, imread.header[3])
code = df["best_match"].iloc[i]
print(f"Testing image {i}")
print(f"Best matching neuron: {code}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(imread.data[i, :, :, 0], cmap="viridis")
axes[1].imshow(mapping.data[i, :, :, 0], cmap="gray")

axes[1].set_yticks(np.arange(0, som_height))
axes[1].set_yticklabels(list(string.ascii_uppercase)[:5])
axes[1].set_ylim(som_height - 0.5, -0.5)

plt.show()


fig, axes = plt.subplots(5, 5, sharex=True)
for col in range(som_shape[0]):
    for row in range(som_shape[1]):
        ax = axes[col, row]
        code = f"{string.ascii_uppercase[col]}{row}"
        ax.hist(df[df["best_match"] == code]["best_match_distance"])
        ax.text(0.8, 0.8, code, transform=ax.transAxes)
plt.show()


count = Counter(df["best_match"])
freq = np.reshape(sorted(count.items()), (5, 5, 2))
# freq = freq.swapaxes(1,2)
labels = freq[:, :, 0]
values = np.array(freq[:, :, 1], dtype=float)

plt.imshow(values, cmap="viridis")
plt.colorbar()

for row in range(5):
    for col in range(5):
        let = string.ascii_uppercase[row]
        code = f"{let}{col}"
        plt.text(
            row,
            col,
            f"{100*count[code]/mapping.header[3]:0.1f}%",
            color="r",
            horizontalalignment="center",
            verticalalignment="center",
        )

plt.yticks(np.arange(0, som_height))
plt.gca().set_yticklabels(list(string.ascii_uppercase)[:5])
plt.ylim(som_height - 0.5, -0.5)
plt.show()
