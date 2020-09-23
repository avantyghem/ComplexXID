#!/usr/bin/env python

import os, sys
import pandas as pd
from matplotlib import pyplot as plt
import utils as pu
import pdb

def plot_images(im_radio, im_ir):
    fig, axes = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
    axes[0].imshow(im_radio, origin="lower")
    axes[1].imshow(im_ir, origin="lower")
    plt.setp(axes, xticks=[], yticks=[])

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} image_bin")
    sys.exit(-1)

# base = sys.argv[1].rstrip("_imgs.bin")
base = sys.argv[1].split("_imgs.bin")[0]
imbin = pu.ImageReader(sys.argv[1])
nchan = imbin.data.shape[-1]  # number of image channels (i.e. radio + IR)

outdir = f"{base}_images"
if not os.path.exists(outdir):
    os.makedirs(outdir)

df = pd.read_csv(f"{base}.csv")

for i in range(imbin.num_images):
    fname = df.iloc[i]["filename"]
    outname = os.path.join(outdir, f"{fname}.png")
    print(f"Creating image {i}/{imbin.num_images}: {outname}")
    im_radio = imbin.data[i, :, :, 0]
    if nchan == 2:
        im_ir = imbin.data[i, :, :, 1]
        plot_images(im_radio, im_ir)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5,5), constrained_layout=True)
        ax.imshow(im_radio, origin="lower")
        plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(outname)
    plt.close()