"""SOM diagnostics"""

import os, sys
import numpy as np
from scipy.ndimage import rotate
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pyink as pu
from typing import Callable, Iterator, Union, List, Set, Dict, Tuple, Optional


def inv_transform(img, transform):
    flip, angle = transform
    if flip == 1:
        img = img[::-1]
    img = rotate(img, np.rad2deg(-angle), reshape=False)
    return img


def plot_image(
    imbin,
    idx=None,
    df=None,
    somset=None,
    apply_transform=False,
    fig=None,
    show_bmu=False,
):
    """Plot the radio+IR channels of a single image.
    """
    if idx is None:
        if df is not None:
            # If df is supplied, choose a random index from the df
            idx = np.random.choice(df.index)
        else:
            # Choose a random index from the image binary
            idx = np.random.randint(imbin.data.shape[0])
    img = imbin.data[idx]

    if apply_transform or show_bmu:
        # Need bmu_idx, which requires either a df or somset
        if somset is not None:
            bmu_idx = somset.mapping.bmu(idx)
            tkey = somset.transform.data[(idx, *bmu_idx)]
        elif df is not None:
            bmu_idx = df.loc[idx]["bmu_tup"]
            tkey = df.loc[idx][["flip", "angle"]]
        else:
            raise ValueError("apply_transform requires either a df or somset")

    if apply_transform:
        img = pu.pink_spatial_transform(img, tkey)
        # img = np.array([inv_transform(c_img, tkey) for c_img in img])

    if fig is None:
        if show_bmu:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    else:
        axes = fig.axes
    axes[0].imshow(img[0])
    axes[1].imshow(img[1])
    axes[0].set_title(f"index = {idx}")

    if show_bmu:
        if somset is None:
            raise ValueError("Cannot show the bmu with no somset provided")
        axes[2].imshow(somset.som[bmu_idx][0])
        axes[2].set_title(f"Best-matching neuron: {bmu_idx}", fontsize=12)
        # axes[2].imshow(somset.som[bmu_idx][1])

    if df is not None:
        fig.suptitle(df.loc[idx]["Component_name"], fontsize=16)


if len(sys.argv) != 3:
    print("USAGE: {} sample_file som_file".format(sys.argv[0]))
    sys.exit(-1)

sample_file = sys.argv[1]
df = pd.read_csv(sample_file)

imbin = pu.ImageReader(sample_file.replace(".csv", "_imgs.bin"))

som_file = sys.argv[2]
som = pu.SOM(som_file)

map_file = som_file.replace(".bin", "_Similarity.bin")
mapping = pu.Mapping(map_file)

trans_file = som_file.replace(".bin", "_Transform.bin")
transform = pu.Transform(trans_file)

somset = pu.SOMSet(som=som, mapping=mapping, transform=transform)

### Update the component catalogue ###

bmus = mapping.bmu()
df["bmu_x"] = bmus[:, 0]
df["bmu_y"] = bmus[:, 1]
df["bmu_tup"] = mapping.bmu(return_tuples=True)
df["bmu_ed"] = mapping.bmu_ed()

trans = transform.data[np.arange(transform.data.shape[0]), bmus[:, 0], bmus[:, 1]]
df["angle"] = trans["angle"]
df["flip"] = trans["flip"]

### Plot number of images for each neuron ###

bmu_counts = mapping.bmu_counts()  # 2D array

# som_shape: (width, height, depth)
w, h, d = som.som_shape
Y, X = np.mgrid[:w, :h]
neuron_stats = pd.DataFrame(dict(row=Y.flatten(), col=X.flatten()))
# neuron_stats = pd.DataFrame(np.array([Y.flatten(), X.flatten()]).T, columns=("row", "col"))
neuron_stats["freq"] = np.array(bmu_counts.flatten(), dtype=int)

plt.imshow(np.log10(bmu_counts), cmap="viridis")
plt.colorbar()

for row in range(mapping.data.shape[1]):
    for col in range(mapping.data.shape[2]):
        plt.text(
            col,
            row,
            f"{bmu_counts[row, col]:.0f}",
            color="r",
            horizontalalignment="center",
            verticalalignment="center",
        )

### Compare an image to its bmu ###

# i = np.random.randint(len(df))
i = 4187
bmu_idx = df.iloc[i]["bmu_tup"]
tkey = transform.data[(i, *bmu_idx)]

fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

img_radio = imbin.data[i, 0, :, :]
img_ir = imbin.data[i, 1, :, :]

axes[0, 0].imshow(img_radio)
axes[1, 0].imshow(img_ir)
axes[0, 0].set_title("Input image (original)")

img_radio = pu.pink_spatial_transform(img_radio, tkey)
img_ir = pu.pink_spatial_transform(img_ir, tkey)

axes[0, 1].imshow(img_radio)
axes[1, 1].imshow(img_ir)
axes[0, 1].set_title("Input image (flipped+rotated)")

axes[0, 2].imshow(som[bmu_idx][0])
axes[1, 2].imshow(som[bmu_idx][1])
axes[0, 2].set_title("Best-matching Neuron")


### Histogram of distances ###

fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
ax.hist(df["bmu_ed"], bins=100)
ax.loglog()
ax.set_xlabel("Euclidean Distance", fontsize=16)
ax.set_ylabel(r"$N$", fontsize=16)

### Get all components for a single neuron ###

neuron = (9, 1)
selection = df[df.bmu_tup == neuron]

### Histogram of distances for a single neuron ###

neuron = (9, 1)
selection = df[df.bmu_tup == neuron]
dist = selection["bmu_ed"]
fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
ax.hist(dist, bins=100)
if len(dist) > 100:
    ax.loglog()
ax.set_xlabel("Euclidean Distance", fontsize=16)
ax.set_ylabel(r"$N$", fontsize=16)

### Select outliers ###

bad = mapping.data.reshape(mapping.data.shape[0], -1)
bad = bad.min(axis=1)
args = bad.argsort()[::-1]
bad_df = df.iloc[args[:100]]

# for i, idx in enumerate(args[::-1]):
#     if i > 10:
#         break

#     fig, ax1 = plt.subplots(1,1)
#     ax1.imshow(imgs.data[idx, 0])
#     fig.savefig(f"{path.Weird}/weird_{i}.pdf")

### Random image matching a given neuron ###

neuron = (0, 0)
selection = df[df.bmu_tup == neuron]
idx = np.random.choice(selection.index)
plot_image(imbin, df=df, somset=somset, apply_transform=True, show_bmu=True)

### Worst matches for a given neuron ###

frac = 1
neuron = (0, 0)
selection = df[df.bmu_tup == neuron]
keep = max(1, int(frac * len(selection)))
selection = selection.sort_values("bmu_ed", ascending=False).iloc[:keep]

plot_image(imbin, df=df, somset=somset, show_bmu=True, idx=selection.index[1])
