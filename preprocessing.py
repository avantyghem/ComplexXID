"""Preprocess a sample of VLASS components with corresponding unWISE
images. Outputs an image binary file suitable for PINK.
"""

import os
import pickle
from multiprocessing import Pool, cpu_count
from glob import glob
from collections import Counter
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from tqdm import tqdm as tqdm
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from reproject import reproject_interp

import pyink as pu

# from vos import Client
# from legacy_survey_cutout_fetcher import grab_vlass_unwise_cutouts

import logging

# logger = logging.getLogger("pyink.binwrap")
# logger.addHandler(logging.StreamHandler().setLevel(logging.DEBUG))

np.seterr(all="ignore")

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning, append=True)


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Just take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def radio_preprocess(data, lower=3):
    remove_zeros = data[data != 0]  # data.flatten()
    noise = pu.rms_estimate(remove_zeros, mode="mad", clip_rounds=2)

    # empty = ~np.isfinite(data)
    # data[empty] = np.random.normal(0, noise, np.sum(empty))

    img_scale = np.zeros_like(data)
    for i, mask in enumerate(pu.island_segmentation(data, 3 * noise)):
        if np.sum(mask) <= 5:
            continue

        img_scale[mask] = pu.minmax(np.log10(data[mask]))

    img_scale[~np.isfinite(img_scale)] = 0

    return img_scale.astype(np.float32)


def ir_preprocess(data):
    med = np.median(data.flatten())
    mad = np.median(np.abs(data.flatten() - med))

    noise = med + 3 * mad

    img_scale = np.zeros_like(data)
    for i, mask in enumerate(pu.island_segmentation(data, noise)):
        if np.sum(mask) <= 5:
            continue
        img_scale[mask] = pu.minmax(np.log10(data[mask]))

    img_scale[~np.isfinite(img_scale)] = 0

    return img_scale.astype(np.float32)


def vlass_preprocessing(
    idx,
    tab,
    ir=True,
    img_size=(2, 150, 150),
    radio_path="",
    ir_path="",
    radio_fname_col="filename",
    ir_fname_col="ir_filename",
    ir_weight=0.05,
):
    """Preprocess a single VLASS image. 
    Do not worry about parallelization yet.
    """

    radio_file = tab.loc[idx][radio_fname_col]
    radio_file = os.path.join(radio_path, radio_file)

    try:
        dlist = fits.open(radio_file)
        d = dlist[0]
    except:
        print(f"Failed to open image {radio_file}")
        return None

    if d.header["NAXIS"] == 0:
        print(f"Empty image: {radio_file}")
        return None

    d_wcs = WCS(d.header).celestial
    d.data = np.squeeze(d.data)
    d.header = d_wcs.to_header()

    if ir:
        ir_file = tab.loc[idx][ir_fname_col]
        ir_file = os.path.join(ir_path, ir_file)

        try:
            wlist = fits.open(ir_file)
            w = wlist[0]
        except:
            print(f"Failed to open image {ir_file}")
            return None

        reproject_wcs = w.header.copy()
    else:
        reproject_wcs = d.header.copy()

    # Recentering the reference pixel
    reproject_wcs["CRVAL1"] = tab.loc[idx]["RA"]
    reproject_wcs["CRVAL2"] = tab.loc[idx]["DEC"]
    reproject_wcs["CRPIX1"] = img_size[1] // 2 + 0.5
    reproject_wcs["CRPIX2"] = img_size[2] // 2 + 0.5
    reproject_wcs = WCS(reproject_wcs).celestial

    e_new_data, e_new_footprint = reproject_interp(
        d, reproject_wcs, shape_out=img_size[1:]
    )

    e_new_data = radio_preprocess(e_new_data)
    dlist.close()

    if np.sum(~np.isfinite(e_new_data)) > 0:
        print(f"Skipping index {idx} due to too many NaN")
        return None
    if np.max(e_new_data) <= 0:
        print(f"Skipping index {idx} due to no positive values")
        return None

    if ir is False:
        return (idx, np.array([e_new_data]))

    w_new_data, w_new_footprint = reproject_interp(
        w, reproject_wcs, shape_out=img_size[1:]
    )

    w_new_data = ir_preprocess(w_new_data)
    wlist.close()

    if np.sum(~np.isfinite(w_new_data)) > 0:
        print(f"Skipping index {idx} due to too many NaN in IR channel")
        return None
    if np.max(w_new_data) <= 0:
        print(f"Skipping index {idx} due to no positive values in IR channel")
        return None

    return (idx, np.array((e_new_data * (1 - ir_weight), w_new_data * ir_weight)))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Preprocess a sample of images and output a SOM image binary."
    )
    parser.add_argument(
        dest="sample", help="Data sample to use", type=str,
    )
    parser.add_argument(
        dest="outfile", help="Name for the output image binary file", type=str,
    )
    parser.add_argument(
        "-r",
        "--radio_survey",
        dest="radio_survey",
        help="The radio survey to be processed",
        default="VLASS",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--ir_survey",
        dest="ir_survey",
        help="The optical/IR survey to be processed",
        default="unWISE-NEO4",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--image_path",
        dest="img_path",
        help="Path to the directory containing the raw input images",
        default="images",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        help="Number of threads to use for multiprocessing",
        default=cpu_count(),
        type=int,
    )
    args = parser.parse_args()
    return args


def main(df, outfile, img_size, img_path, threads=None):
    if threads is None:
        threads = cpu_count()

    pool = Pool(processes=threads)

    kwargs = dict(ir=True, img_size=img_size, radio_path=img_path, ir_path=img_path)
    with pu.ImageWriter(outfile, 0, img_size, clobber=True) as pk_img:
        results = [
            pool.apply_async(vlass_preprocessing, args=(idx, df), kwds=kwargs)
            for idx in df.index
        ]
        for res in tqdm(results):
            out = res.get()
            if out is not None:
                pk_img.add(out[1], attributes=out[0])


if __name__ == "__main__":
    args = parse_args()

    # Feed DataFrame directly into preprocessing routine
    # Requires filename columns
    df = pd.read_csv(args.sample)

    if "filename" not in df:
        df["filename"] = df["Component_name"].apply(
            add_filename, survey=args.radio_survey
        )
        df["ir_filename"] = df["Component_name"].apply(
            add_filename, survey=args.ir_survey
        )

    # interested in `complex` objects
    # outfile = "PINK_Binaries/Example_PINK_Binary.bin"
    # emu = df[df["n_components"] > 1].index
    main(
        df,
        args.outfile,
        img_size=(2, 500, 500),
        img_path=args.img_path,
        threads=args.threads,
    )
