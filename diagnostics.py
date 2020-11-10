"""SOM diagnostics"""

import os, sys
from collections import Counter
import argparse
from itertools import product
from typing import Callable, Iterator, Union, List, Set, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import pyink as pu

# from collation import create_wcs


def binary_names(tile_id, path=""):
    imbin_file = os.path.join(path, f"IMG_{tile_id}.bin")
    map_file = os.path.join(path, f"MAP_{tile_id}.bin")
    trans_file = os.path.join(path, f"TRANSFORM_{tile_id}.bin")
    return imbin_file, map_file, trans_file


def create_wcs(ra, dec, imgsize, pixsize):
    pixsize = u.Quantity(pixsize, u.deg)
    hdr = fits.Header()
    hdr["CRPIX1"] = imgsize // 2 + 0.5
    hdr["CRPIX2"] = imgsize // 2 + 0.5
    hdr["CDELT1"] = -pixsize.value
    hdr["CDELT2"] = pixsize.value
    hdr["PC1_1"] = 1
    hdr["PC2_2"] = 1
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    return WCS(hdr)


def neuron_img_comp(somset, imgs, sampler, outpath=""):
    path = pu.PathHelper(outpath)
    logfile = open(f"{path.path}/info.txt", "w")

    bmu_ed = somset.mapping.bmu_ed()

    img_shape = imgs.data.shape[-1]
    neuron_shape = somset.som.neuron_shape[-1]
    b1 = (neuron_shape - img_shape) // 2
    b2 = b1 + img_shape

    for neuron, ind in enumerate(sampler.points):
        # Plot neuron
        somset.som.plot_neuron(ind)
        f1 = plt.gcf()
        plt.xticks([])
        plt.yticks([])

        radio_img = somset.som[ind][0]
        levels = np.linspace(0.25 * radio_img.max(), radio_img.max(), 4)
        f1.axes[1].contour(radio_img, levels=levels, colors="white", linewidths=0.5)

        f1.axes[0].set_xlim([b1, b2])
        f1.axes[0].set_ylim([b2, b1])
        for ax in f1.axes:
            ax.axvline(neuron_shape / 2, c="r", ls="--", lw=1)
            ax.axhline(neuron_shape / 2, c="r", ls="--", lw=1)

        f1.savefig(f"{path.path}/neuron_{neuron}.png")
        plt.close(f1)

        # Plot images
        matches = somset.mapping.images_with_bmu(ind)
        dist = bmu_ed[matches]
        idx1 = matches[np.argmin(dist)]
        idx2 = matches[np.argsort(dist)[len(matches) // 2]]
        for i, idx in enumerate([idx1, idx2]):
            plot_image(
                imgs,
                idx=idx,
                somset=somset,
                apply_transform=True,
                show_index=False,
                grid=True,
            )
            f2 = plt.gcf()
            plt.xticks([])
            plt.yticks([])
            for ax in f2.axes:
                ax.axvline(img_shape / 2, c="r", ls="--", lw=1)
                ax.axhline(img_shape / 2, c="r", ls="--", lw=1)

            bmu_idx = somset.mapping.bmu(idx)
            tkey = somset.transform.data[(idx, *bmu_idx)]
            radio_img = imgs.data[idx, 0]
            radio_img = pu.pink_spatial_transform(radio_img, tkey)
            levels = np.linspace(0.25 * radio_img.max(), radio_img.max(), 4)
            f2.axes[1].contour(radio_img, levels=levels, colors="white", linewidths=0.5)

            f2.savefig(f"{path.path}/neuron_{neuron}_img{i}.png")
            plt.close(f2)

        # Print to a log file
        print(f"Neuron {neuron}: {ind}", file=logfile)
        print(f"Number in neuron: {len(matches)}", file=logfile)
        print(f"img0 ind: {idx1}", file=logfile)
        print(f"img1 ind: {idx2}", file=logfile)
        print("------------------\n", file=logfile)


def source_from_catalogue(
    src_cat,
    radio_cat,
    imbin_path,
    som=None,
    show_nearby=False,
    show_bmu=False,
    idx=None,
):
    if idx is None:
        # idx = np.random.randint(len(src_cat))
        idx = src_cat[src_cat.N_components > 1].sample(1).index[0]
    src = src_cat.loc[idx]

    comp_names = src.Component_names.split(";")
    tile_id = radio_cat[radio_cat.Component_name == comp_names[0]].Tile.iloc[0]
    radio_tile = radio_cat[radio_cat.Tile == tile_id].reset_index(drop=True)
    comps = radio_tile[radio_tile["Component_name"].isin(comp_names)]

    radio_ra = np.array(src.RA_components.split(";"), dtype=float)
    radio_dec = np.array(src.DEC_components.split(";"), dtype=float)
    radio_pos = SkyCoord(radio_ra, radio_dec, unit=u.deg)

    if src.N_host_candidates > 0:
        ir_ra = np.array(src.RA_host_candidates.split(";"), dtype=float)
        ir_dec = np.array(src.DEC_host_candidates.split(";"), dtype=float)
        ir_pos = SkyCoord(ir_ra, ir_dec, unit=u.deg)

    img_file, map_file, trans_file = binary_names(tile_id, imbin_path)
    imgs = pu.ImageReader(img_file)
    mapping = pu.Mapping(map_file)
    transform = pu.Transform(trans_file)

    if som is not None:
        somset = pu.SOMSet(som, mapping, transform)

    img_idx = comps.index[0]
    comp = comps.loc[img_idx]
    npix = imgs.data.shape[-1]
    wcs = create_wcs(comp.RA, comp.DEC, npix, 3 * u.arcmin / npix)

    if show_bmu:
        plot_image(
            imgs,
            img_idx,
            somset=somset,
            wcs=wcs,
            transform_neuron=True,
            show_bmu=show_bmu,
        )
    else:
        plot_image(imgs, img_idx, wcs=wcs, show_bmu=False)
    axes = plt.gcf().axes

    if show_nearby:
        posn = SkyCoord(comp.RA, comp.DEC, unit=u.deg)
        coords = SkyCoord(radio_cat.RA, radio_cat.DEC, unit=u.deg)
        nearby = coords[posn.separation(coords) < 1.5 * u.arcmin]
        for ax in plt.gcf().axes:
            ax.scatter(
                nearby.RA, nearby.DEC, c="w", transform=ax.get_transform("world")
            )

    # for ax in plt.gcf().axes:
    #     ax.scatter(comps.RA, comps.DEC, c="r", transform=ax.get_transform("world"))

    axes[0].scatter(
        radio_pos.ra, radio_pos.dec, c="r", transform=axes[0].get_transform("world")
    )
    if src.N_host_candidates > 0:
        axes[1].scatter(
            ir_pos.ra, ir_pos.dec, c="w", transform=axes[1].get_transform("world")
        )

    plt.suptitle(f"Source ID: {idx}")


def trim_neuron(neuron, img_shape=None):
    som_shape = neuron.shape[0]
    if img_shape is None:
        img_shape = int(np.floor(som_shape / np.sqrt(2)))
    b1 = (som_shape - img_shape) // 2
    b2 = b1 + img_shape
    return neuron[b1:b2, b1:b2]


def plot_neuron(som, idx, fig=None):
    cmaps = ["viridis", "plasma", "inferno", "cividis"]
    nchan = som.data.shape[0]
    if fig is None:
        fig, axes = plt.subplots(
            1, nchan, figsize=(5 * nchan, 5), sharex=True, sharey=True
        )
    for chan, ax in enumerate(axes):
        ax.imshow(som[idx][chan], cmap=cmaps[chan])
    plt.show()


def plot_neuron_grid(som, start=(0, 0), dim=5, cross=False):
    fig, axes = plt.subplots(
        dim, dim, figsize=(1.5 * dim, 1.5 * dim), sharex=True, sharey=True,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    y1, x1 = start
    for yi in range(y1, y1 + dim):
        for xi in range(x1, x1 + dim):
            ax = axes[yi, xi]
            ax.imshow(som[yi, xi][0], cmap="viridis")
            ax.contour(
                som[yi, xi][1],
                colors="w",
                linewidths=0.5,
                levels=0.05 * np.arange(0.5, 1, 0.1),
            )
            if cross:
                ax.axhline(0.5 * som.header[-1][-2], lw=0.5, c="r", ls="--")
                ax.axvline(0.5 * som.header[-1][-1], lw=0.5, c="r", ls="--")


def plot_som(som, channel=0, fig=None, show_cbar=False, outfile=None):
    shape = som.som_shape[:2]

    if fig is None:
        base_size = [10, 10]
        base_size[np.argmin(shape)] = (
            base_size[np.argmax(shape)] * np.min(shape) / np.max(shape)
        )
        if show_cbar:
            base_size[1] += 2
        figsize = tuple(base_size)
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    cmap = ax.imshow(som.data[channel, :, :], cmap="bwr")
    if show_cbar:
        fig.colorbar(cmap)
    marks_x = np.arange(0, som.som_shape[0] + 1, 1) * som.neuron_shape[1]
    marks_y = np.arange(0, som.som_shape[1] + 1, 1) * som.neuron_shape[2]
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.axvline(m - 0.5, c="k", ls="-") for m in marks_x]
    [ax.axhline(m - 0.5, c="k", ls="-") for m in marks_y]
    ax.set_xlim(xmax=marks_x[-1])
    ax.set_ylim(ymin=marks_y[-1])
    if outfile is not None:
        plt.savefig(outfile, dpi=shape[0] * 25)
    else:
        plt.show()


def plot_image(
    imbin,
    idx=None,
    df=None,
    somset=None,
    apply_transform=False,
    transform_neuron=False,
    fig=None,
    show_bmu=False,
    show_index=True,
    wcs=None,
    grid=False,
):
    """Plot an image from the image set.

    Args:
        imbin (pu.ImageReader): Image binary
        idx (int, optional): Index of the image to plot. Defaults to None.
        df (pandas.DataFrame, optional): Table with information on the sample. Defaults to None.
        somset (pu.SOMSet, optional): Container holding the SOM, mapping, and transform. Defaults to None.
        apply_transform (bool, optional): Option to transform the image to match the neuron. Defaults to False.
        fig (pyplot.Figure, optional): Preexisting Figure for plotting. Defaults to None.
        show_bmu (bool, optional): Display the best-matching neuron alongside the image. Defaults to False.

    Raises:
        ValueError: Transform requested without the required information.
        ValueError: bmu requested with no SOMSet
    """
    cmaps = ["viridis", "plasma", "inferno", "cividis"]
    if idx is None:
        # Choose an index randomly from either the df (which can
        # be trimmed to a smaller sample) or the pu.ImageReader.
        if df is not None:
            idx = np.random.choice(df.index)
        else:
            idx = np.random.randint(imbin.data.shape[0])
    img = imbin.data[idx]

    nchan = imbin.data.shape[1]
    img_shape = imbin.data.shape[2]

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
        nrow = 2 if show_bmu else 1
        cl = False if wcs is not None else True
        fig, axes = plt.subplots(
            nrow,
            nchan,
            figsize=(nchan * 4, nrow * 4),
            sharex=True,
            sharey=True,
            constrained_layout=cl,
            subplot_kw=dict(projection=wcs),
        )
    else:
        axes = fig.axes

    for ax, chan in zip(axes.flatten(), range(nchan)):
        ax.imshow(img[chan], cmap=cmaps[chan])
        if grid:
            ax.grid()

    if show_index:
        axes.flatten()[0].set_title(f"index = {idx}")

    if show_bmu:
        if somset is None:
            raise ValueError("Cannot show the bmu with no somset provided")
        for chan in range(nchan):
            neuron_img = somset.som[bmu_idx][chan]
            if transform_neuron:
                neuron_img = pu.pink_spatial_transform(neuron_img, tkey, reverse=True)
            axes[1][chan].imshow(trim_neuron(neuron_img, img_shape), cmap=cmaps[chan])
            if grid:
                ax.grid()
        axes[1, 0].set_title(f"Best-matching neuron: {bmu_idx}", fontsize=12)

    if df is not None:
        fig.suptitle(df.loc[idx]["Component_name"], fontsize=16)


def plot_source(imgs, somset, radio_cat, idx, components, ir_srcs, pixsize):
    # Plot the image with the positions of the related radio components
    # and candidate IR hosts
    pixsize = u.Quantity(pixsize, u.arcsec)
    src = radio_cat.loc[idx]
    plot_image(imgs, somset=somset, show_bmu=True, transform_neuron=True, idx=src_idx)
    fig = plt.gcf()
    wcs = create_wcs(src.RA, src.DEC, imgs.data.shape[2], pixsize)

    # Plot radio components
    x, y = wcs.all_world2pix(components.ra, components.dec, 0)
    for ax in plt.gcf().axes:
        ax.scatter(x, y, color="red")

    # Plot IR sources
    x, y = wcs.all_world2pix(ir_srcs.ra, ir_srcs.dec, 0)
    for ax in plt.gcf().axes:
        ax.scatter(x, y, color="white")


def inspect_components(imgs, somset, positions, matches, idx, pixsize=0.36):
    # Transform an image all nearby components to the BMU frame.
    # Check that the components align with the BMU signal.
    # bmu_keys = somset.mapping.bmu(return_idx=True, squeeze=True)
    # bz, by, bx = bmu_keys.T

    pixsize = u.Quantity(pixsize, u.arcsec)
    bmu = somset.mapping.bmu(idx)
    by, bx = bmu

    radio_positions, ir_positions = positions
    radio_matches, ir_matches = matches

    center_pos = radio_positions[idx]

    trans_key = (idx, *bmu)
    flip, angle = somset.transform.data[trans_key]
    src_transform = (flip, angle)

    src_mask = idx == radio_matches[0]
    src_matches = radio_matches[1][src_mask]

    spatial_radio_pos = pu.CoordinateTransformer(
        center_pos, radio_positions[src_matches], src_transform, pixel_scale=pixsize,
    )

    src_img = imgs.data[idx, 0].copy()
    transform_img = pu.pink_spatial_transform(src_img, src_transform)
    cen_pix = src_img.shape[0] // 2

    fig, axes = plt.subplots(
        2, 3, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True
    )

    axes[0, 0].imshow(src_img)
    axes[0, 1].imshow(transform_img)
    axes[0, 2].imshow(trim_neuron(somset.som[bmu][0], src_img.shape[0]))

    axes[0, 0].plot(
        spatial_radio_pos.coords["offsets-pixel"][0].value + cen_pix,
        spatial_radio_pos.coords["offsets-pixel"][1].value + cen_pix,
        "ro",
    )

    axes[0, 1].plot(
        spatial_radio_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_radio_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    axes[0, 2].plot(
        spatial_radio_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_radio_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    # Plotting the IR channel
    src_img = imgs.data[idx, 1].copy()
    transform_img = pu.pink_spatial_transform(src_img, src_transform)

    src_mask = idx == ir_matches[0]
    src_matches = ir_matches[1][src_mask]

    spatial_ir_pos = pu.CoordinateTransformer(
        center_pos, ir_positions[src_matches], src_transform, pixel_scale=pixsize,
    )

    axes[1, 0].imshow(src_img)
    axes[1, 1].imshow(transform_img)
    axes[1, 2].imshow(trim_neuron(somset.som[bmu][1], src_img.shape[0]))

    axes[1, 0].plot(
        spatial_ir_pos.coords["offsets-pixel"][0].value + cen_pix,
        spatial_ir_pos.coords["offsets-pixel"][1].value + cen_pix,
        "ro",
    )

    axes[1, 1].plot(
        spatial_ir_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_ir_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    axes[1, 2].plot(
        spatial_ir_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_ir_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    axes[0, 0].set(title=f"Original Image")
    axes[0, 1].set(title=f"flip, rot: {src_transform}")
    axes[0, 2].set(title=f"Neuron: {bmu}")


"""
def accumulate(path, k, bmu_keys, som, sky_positions, sky_matches, close=True):
    # Average all images that match to a specified neuron.
    bz, by, bx = bmu_keys.T
    mask = (k[0] == by) & (k[1] == bx)
    argmask = np.argwhere(mask)

    if np.sum(mask) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(np.sqrt(som[k][0]), cmap="Greys")
    ax2.imshow(np.sqrt(som[k][0]), cmap="Greys")

    ax1.set(title=f"{k} - {np.sum(mask)}")

    divider = make_axes_locatable(ax2)
    axHistx = divider.append_axes("top", size="23%", pad="2%", sharex=ax2)
    axHisty = divider.append_axes("right", size="23%", pad="2%", sharey=ax2)

    spatial_pos = []

    for j, src in enumerate(argmask):
        src = src[0]

        center_pos = sky_positions[src]
        src_mask = src == sky_matches[0]
        src_matches = sky_matches[1][src_mask]

        src_transform = transform.data[(src, *k)]
        flip, angle = src_transform

        spatial_emu_pos = pu.CoordinateTransformer(
            center_pos,
            sky_positions[src_matches],
            src_transform,
            pixel_scale=2 * u.arcsecond,
        )

        spatial_pos.append(spatial_emu_pos)

    px = (
        np.concatenate(
            [pos.coords["offsets-neuron"][0].value for pos in spatial_pos]
        ).flatten()
        + 213 / 2
    )
    py = (
        np.concatenate(
            [pos.coords["offsets-neuron"][1].value for pos in spatial_pos]
        ).flatten()
        + 213 / 2
    )

    ax2.plot(px, py, "ro", markersize=2.0, alpha=0.5)

    axHistx.hist(px, bins=30, density=True, histtype="stepfilled", alpha=0.7)
    axHisty.hist(
        py,
        bins=30,
        density=True,
        orientation="horizontal",
        histtype="stepfilled",
        alpha=0.7,
    )
    axHistx.set(ylabel="Density")
    axHisty.set(xlabel="Density")

    # no labels
    plt.setp(axHistx.get_xticklabels(), visible=False)
    plt.setp(axHisty.get_yticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(f"{path.Cluster}/{k}_cluster.png")
"""


def plot_filter_idx(imgs, filters, idx):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, sharex=ax3, sharey=ax3)

    filter_size = filters.filters[0][idx].neuron.filters[0].shape
    img_size = imgs.data[idx, 0].shape

    filters.filters[0][idx].plot(axes=ax1)
    filters.filters[1][idx].plot(axes=ax2)

    ax3.imshow(imgs.data[idx, 0])
    ax4.imshow(imgs.data[idx, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(which="major", axis="both", color="white", alpha=0.6)
    for ax in [ax1, ax2]:
        ax.axvline(filter_size[0] / 2, color="white", ls="--", alpha=0.5)
        ax.axhline(filter_size[1] / 2, color="white", ls="--", alpha=0.5)
    for ax in [ax3, ax4]:
        ax.axvline(img_size[0] / 2, color="white", ls="--", alpha=0.5)
        ax.axhline(img_size[1] / 2, color="white", ls="--", alpha=0.5)


def get_src_img(pos, survey, angular=None, level=0, **kwargs):
    if level > 5:
        print("Failing")
        raise ValueError("Too many failed attempts. ")

    sv_survey = {"first": "VLA FIRST (1.4 GHz)", "wise": "WISE 3.4"}
    survey = sv_survey[survey] if survey in sv_survey else survey

    if angular is None:
        FITS_SIZE = 5 * u.arcmin
    else:
        FITS_SIZE = (angular).to(u.arcsecond)

    CELL_SIZE = 2.0 * u.arcsec / u.pix
    imsize = FITS_SIZE.to("pix", equivalencies=u.pixel_scale(CELL_SIZE))

    try:
        images = SkyView.get_images(
            pos,
            survey,
            pixels=int(imsize.value),
            width=FITS_SIZE,
            height=FITS_SIZE,
            coordinates="J2000",
        )
    except:
        import time

        time.sleep(4)
        get_src_img(pos, survey, angular=angular, level=level + 1)

    return images[0]


def som_counts(somset, show_counts=False):
    # Add an outfile
    counts = somset.mapping.bmu_counts()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    cim = ax.imshow(counts)

    cbar = fig.colorbar(cim)
    cbar.set_label("Counts per Neuron", fontsize=16)
    # cbar.ax.tick_params(labelsize=16)

    if show_counts:
        for row in range(somset.mapping.data.shape[1]):
            for col in range(somset.mapping.data.shape[2]):
                ax.text(
                    col,
                    row,
                    f"{counts[row, col]:.0f}",
                    color="r",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    return


def update_component_catalogue(df, somset):
    # Update the component catalogue
    bmus = somset.mapping.bmu()
    df["bmu_x"] = bmus[:, 0]
    df["bmu_y"] = bmus[:, 1]
    df["bmu_tup"] = somset.mapping.bmu(return_tuples=True)
    df["bmu_ed"] = somset.mapping.bmu_ed()

    trans = somset.transform.data[
        np.arange(somset.transform.data.shape[0]), bmus[:, 0], bmus[:, 1]
    ]
    df["angle"] = trans["angle"]
    df["flip"] = trans["flip"]
    return df


def dist_hist(
    somset,
    df=None,
    neuron=None,
    density=False,
    bins=100,
    log=True,
    ax=None,
    labels=True,
):
    # if neuron is not None:
    #     df = df[df.bmu_tup == neuron]
    #     bmu_ed = df["bmu_ed"]

    bmu = somset.mapping.bmu()
    bmu_ed = somset.mapping.bmu_ed()
    if neuron is not None:
        mask = np.array([bmu_i == neuron for bmu_i in bmu])
        bmu_ed = bmu_ed[mask]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

    if log:
        bmu_ed = np.log10(bmu_ed)
    ax.hist(bmu_ed, bins=bins, density=density)
    ax.set_yscale("log")
    if labels:
        xlabel = "Euclidean Distance" if not log else f"log(Euclidean Distance)"
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(r"$N$", fontsize=16)


def dist_hist_2d(df, somset, bins=100, loglog=False):
    w, h, d = somset.som.som_shape
    fig, axes = plt.subplots(h, w, sharey=True, sharex=True, figsize=(11, 10))
    for row in range(h):
        for col in range(w):
            ax = axes[row, col]
            dist_hist(
                df, neuron=(row, col), density=False, loglog=loglog, ax=ax, labels=False
            )
    fig.subplots_adjust(hspace=0, wspace=0)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Euclidean Distance", fontsize=16)
    plt.ylabel(r"$N$", fontsize=16)


def dist_stats(somset):
    w, h, d = somset.som.som_shape
    neurons = list(product(range(h), range(w)))
    Y, X = np.mgrid[:w, :h]

    dists = np.zeros(Y.shape)
    stds = np.zeros(Y.shape)

    for neuron in neurons:
        # print(neuron)
        inds = somset.mapping.images_with_bmu(neuron)
        if len(inds) == 0:
            continue
        ed = somset.mapping.bmu_ed(inds)
        dists[neuron[0], neuron[1]] = np.percentile(ed, 50)
        stds[neuron[0], neuron[1]] = np.percentile(ed, 67) - np.percentile(ed, 33)

    return dists, stds


def distance_sampling(somset, N=10):
    inds = np.argsort(somset.mapping.bmu_ed())
    low = inds[:N]
    med = inds[len(inds) // 2 - N // 2 : len(inds) // 2 + N // 2]
    high = inds[-N:][::-1]
    return low, med, high


def plot_dist_stats(somset):
    dists, stds = dist_stats(somset)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    p1 = axes[0].imshow(dists)
    cb1 = fig.colorbar(p1, ax=axes[0], aspect=50, shrink=0.95)
    cb1.set_label("Median distance to neuron", size=16)

    p2 = axes[1].imshow(stds)
    cb2 = fig.colorbar(p2, ax=axes[1], aspect=50, shrink=0.95)
    cb2.set_label(r"1$\sigma$ dispersion in distance", size=16)


def neuron_stats(somset):
    # Only tracks the number of images in each neuron so far
    w, h, d = somset.som.som_shape
    Y, X = np.mgrid[:w, :h]
    bmu_counts = somset.mapping.bmu_counts()  # 2D array
    neuron_stats = pd.DataFrame(dict(row=Y.flatten(), col=X.flatten()))
    neuron_stats["freq"] = np.array(bmu_counts.flatten(), dtype=int)
    neuron_stats["idx"] = list(zip(neuron_stats.row.values, neuron_stats.col.values))
    return neuron_stats


def total_image_flux(imgs):
    summed = imgs.data.reshape(-1, 2, np.product(imgs.data.shape[-2:]))
    return summed.sum(axis=2)


def worst_matches(df=None, somset=None, N=None, frac=None, neuron=None):
    # Select images poorly represented by SOM
    # N supercedes frac
    # Require a df, even though this could be done without one.
    if neuron is not None:
        if bmu_tup not in df:
            df = update_component_catalogue(df, somset)
        df = df[df.bmu_top == neuron]

    if N is None:
        if frac is None:
            N = len(df)
        else:
            N = max(1, int(frac * len(df)))

    # Perform the same operation using only the pu.Mapping instance
    # bad = mapping.data.reshape(mapping.data.shape[0], -1)
    # bad = bad.min(axis=1)
    # args = bad.argsort()[::-1]
    # bad_df = df.iloc[args[:N]]

    df = df.sort_values("bmu_ed", ascending=False).iloc[:N]
    return df


def init_somset(som_file, map_file=None, trans_file=None):
    """Initialize the SOMSet, which contains the SOM, mapping,
    and transform files.

    Args:
        som_file (str): Name of the SOM file.
        map_file (str, optional): Name of the mapping file. Defaults to None, 
        in which case it follows the default naming scheme.
        trans_file (str, optional): Name of the transform file. Defaults to None.

    Returns:
        pu.SOMSet: A container holding the SOM, mapping, and transform files.
    """
    som = pu.SOM(som_file)

    if map_file is None:
        map_file = som_file.replace("SOM", "MAP")
    mapping = pu.Mapping(map_file)

    if trans_file is None:
        trans_file = map_file.replace("MAP", "TRANSFORM")
    transform = pu.Transform(trans_file)

    somset = pu.SOMSet(som=som, mapping=mapping, transform=transform)
    return somset


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Download VLASS and unWISE cutouts.")
    # parser.add_argument("sample_file", help="Name of the radio sample table")
    parser.add_argument("imbin_file", help="Name of the image binary")
    parser.add_argument("som_file", help="Name of the SOM file")
    parser.add_argument(
        "-m",
        "--map",
        dest="map_file",
        help="Name of the mapping file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--transform",
        dest="trans_file",
        help="Name of the mapping file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="outpath",
        help="Path to use when creating files",
        default="Plots",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Initialize the sample DataFrame, ImageReader, and SOM
    # files = set_filenames(args.som_file, args.map_file, args.trans_file, args.outbase)

    # df = pd.read_csv(args.sample_file)
    imgs = pu.ImageReader(args.imbin_file)
    somset = init_somset(args.som_file, args.map_file, args.trans_file)
    path = pu.PathHelper(args.outpath)

    # Only keep the records that preprocessed successfully
    # df = df.loc[imgs.records].reset_index(drop=True)
    # df = update_component_catalogue(df, somset)

    som_counts(somset, show_counts=True)
    plt.savefig(f"{path}/bmu_frequency.png")
    # plot_image(imgs, df=df, somset=somset, show_bmu=True, apply_transform=True)

    # dist_hist(df, labels=True)
    # plt.savefig(f"{path}/euc_distance_hist.png")

    # dist_hist_2d(df, somset, bins=100, loglog=False)
    # plt.savefig(f"{path}/euc_distance_hist_2d.png", dpi=somset.som.som_shape[0] * 25)

    # # Restrict sample to a specific bmu
    # neuron = (9, 1)
    # selection = df[df.bmu_tup == neuron]
    # plot_image(imgs, df=selection, somset=somset, apply_transform=True, show_bmu=True)

    # nstats = neuron_stats(somset)
    # worst = worst_matches(df)
    # worst.to_csv(f"{path}/worst_matches.csv")

    total_flux = total_image_flux(imgs)
    bmu_ed = somset.mapping.bmu_ed()

    for i in range(total_flux.shape[1]):
        plt.clf()
        chan_tot = total_flux[:, i]
        name = ["Radio", "IR"][i]
        plt.hist2d(np.log10(chan_tot), np.log10(bmu_ed), bins=100)
        plt.xlabel(f"Sum of {name} pixels")
        plt.ylabel("log(Euc Dist)")
        plt.colorbar()
        plt.savefig(f"EucDist_{name}_hist2d.png")
        plt.close()

    fig, ax = plt.subplots(1, 1)
    dist_hist(somset, ax=ax, log=True, labels=True)
    plt.savefig(f"EucDist_hist_loglog.png")
