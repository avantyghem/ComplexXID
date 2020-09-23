#!/usr/bin/env python
"""Script to apply preprocessing steps to images and create a PINK binary.

Important to remember images are (y,x) when in numpy arrays. Also, critical
to properly write out images to binary file in correct row/column order expected
by PINK. 
"""
import os
import argparse
import struct as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from utils import ImageWriter

np.random.seed(42) # Order is needed in the Universe


def background(img: np.ndarray, region_size: int=10):
    """Create slices to segment out the inner region of an image
    
    Arguments:
        img {np.ndarray} -- Image to work out background statistics
        region_size {int} -- Size of the inner most region. Equal size dimensions only. 
    """
    img_size = np.array(img.shape)
    center   = img_size // 2
    region   = region_size // 2
    return (slice(center[0]-region, center[0]+region),
            slice(center[1]-region, center[1]+region))


def background_stats(img: np.ndarray, slices: tuple):
    """Calculate and return background statistics. Procedure is following
    from Pink_Experiments repo written originally by EH/KP. 
    
    Arguments:
        img {np.ndarray} -- Image to derive background statistics for
        slices {tuple} -- Tuple of slices returned from background
    """
    empty = ~np.isfinite(img)
    mask  = np.full(img.shape, True) 
    
    mask[empty] = False 
    mask[slices] = False

    return {'mean': np.mean(img[mask]),
            'std': np.std(img[mask]),
            'min': np.min(img[mask]),
            'empty': empty}

def first_process(*args, **kwargs):
    return radio_process(*args, **kwargs)

def radio_process(img: np.ndarray, *args, survey: str="FIRST", inner_frac: int=5, 
                  clip_level: int=1, weight: float=1., adaptive_max: float=None, 
                  adaptive_damp: float=1., upper_sig: float=None, **kwargs):
    """Procedure to preprocess FIRST data, following the original set of steps from 
    EH/KP
    
    Arguments:
        img {nd.ndarray} -- Image to preprocess
        inner_fact {int} -- Fraction of the inner region to extract
        clip_level {int} -- Clip pixels below this background threshold
        weight {float} -- Weight to apply to the channel
        adaptive_max {float} -- Threshold to use to enable adaptive clip level. No clipping by default. 
        adaptive_damp {float} -- Tuning parameter to control aggressiveness of adaptive clipping. 
        upper_sig {float} -- Upper clipping level in units of std. No limit by default. 
    """
    size = img.shape[0] # Lets assume equal pixel sizes
    slices = background(img, region_size=int(size/2))
    bstats = background_stats(img, slices)

    # Try to drop out weak sources to avoid including them
    # onto the map. This is an experiment. 
    # if np.max(img[slices]) < 5*bstats['std']:
    #     raise ValueError('FIRST source is weak')

    # Replace empty pixels
    img[bstats['empty']] = np.random.normal(loc=bstats['mean'], scale=bstats['std'],
                                            size=np.sum(bstats['empty'].flatten()))
    # Clip out noise pixels and scale image
    if adaptive_max is not None:
        clip_level += ((adaptive_max - min(img.max()/bstats['std'], adaptive_max))/adaptive_damp)

    noise_level = clip_level*bstats['std']
    img = img - bstats['mean']

    upper_level = 1e10 if upper_sig is None else upper_sig*bstats['std']
    img = np.clip(img, noise_level, upper_level)
    img = img - noise_level
    
    # MinMax scale the image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Appply channel weight
    img = img * weight

    # Final check to flag out any images with a non-finite
    if np.sum(~np.isfinite(img)) > 0:
        raise ValueError(F"FIRST contains non-finite values.")

    return img


def wise_process(*args, **kwargs):
    return ir_process(*args, **kwargs)


def ir_process(img: np.ndarray, *args, inner_frac: int=5,
               weight: float=1., log10: bool=True, nan_limit: int=20):
    """Procedure to preprocess images from the optical/IR survey

    Arguments:
        img {np.ndarray} -- [description]
    
    Keyword Arguments:
        weight {float} -- Weight to apply to the channel (default: {1.})
        inner_fact {int} -- Fraction of the inner region to extract
        log10 {bool} -- log10 the data before returning (default: {True})
        nan_limit {int} -- Limit on how many pixels are allowed to be blank before rejecting (default: {20})
    """
    size = img.shape[0] # Lets assume equal pixel sizes
    slices = background(img, region_size=size//5)
    bstats = background_stats(img, slices)

    # NEED A WAY TO MASK OUT STARS
    # CURRENTLY LOSING 20% OF THE SAMPLE DUE TO TOO MANY NAN
    if np.sum(bstats['empty']) > nan_limit:
        raise ValueError('IR image has to many NaN pixels to replace')

    # Replace empty pixels
    img[bstats['empty']] = np.random.normal(loc=bstats['mean'], scale=bstats['std'],
                                          size=np.sum(bstats['empty'].flatten()))
    # Clip out nasty values to allow the log10 to work
    # PINK does not like having nans or infs
    img = np.clip(img, 0.0001, 1e10)

    if log10:
        img = np.log10(img)

    # MinMax scale the image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # apply channel weight
    img = img * weight

    # Final check to flag out any images with a non-finite
    if np.sum(~np.isfinite(img)) > 0:
        raise ValueError(F"IR image contains non-finite values.")

    return img


def get_fits(f: str, ext: int=0):
    """Read in the FITS data for FIRST

    f {str} -- The filename to read in
    """
    with fits.open(f) as hdu1:
        img = hdu1[0].data.copy()
        while len(img.shape) > 2:
            img = img[0]

    return img


def main(files: list, ir_files, out_path: str, *args, 
         radio_path: str='../first',
         ir_path: str='../wise_reprojected', 
         radio_weight: float=0.95,
         ir_weight: float=0.05,
         radio_survey="FIRST",
         ir_survey="WISE-w1",
         **kwargs):
    """Run the preprocessing on the set of files
    
    Arguments:
        files {list} -- Iterable with list of radio filenames to process
        ir_files {list} -- Iterable with list of IR filenames to process
        out_path {str} -- Name of the file to create
        radio_path {str} -- Relative path containing the first fits images
        ir_path {str} -- Relative path containing the wise reprojected fits images
        radio_weight (float) -- Weighting applied to first channel
        ir_weight {float} -- Weighting applied to wise channel

    Raises:
        Exception -- Catch all used for removing files that fail preprocessing
    """

    radio_shape = get_fits(f"{radio_path}/{files[0]}").shape
    ir_shape = get_fits(f"{radio_path}/{files[0]}").shape[-2:] # In case of 3D
    reproject = (ir_shape != radio_shape) # reproject if pixel scale does not match
    if reproject:
        print(f"Reprojecting radio images from a shape of {radio_shape} to {ir_shape}")
    height, width = ir_shape

    print(f'Derived height, width: {height}, {width}')

    # File handler
    imwriter = ImageWriter(out_path, 0, (2,width,height))
    # imwriter = ImageWriter(out_path, 0, (width,height))

    success = []
    failed  = [] 
    reason  = []
    for f, f_ir in tqdm(zip(files, ir_files)):
        try:
            ir_full = os.path.join(ir_path, f_ir)
            radio_full = os.path.join(radio_path, f)

            img_ir = get_fits(ir_full)
            img_ir = ir_process(img_ir, *args, weight=ir_weight, **kwargs)
            
            if img_ir.shape != ir_shape:
                raise ValueError("Image is not the correct size.")

            if reproject:
                hdu_ir = fits.open(ir_full)
                hdu_radio = fits.open(radio_full)
                # img_radio, footprint = reproject_interp(hdu_radio, hdu_ir[0].header)
                wcs = WCS(hdu_ir[0].header)
                img_radio, footprint = reproject_interp(hdu_radio, wcs.celestial, ir_shape)
            else:
                img_radio = get_fits(radio_full)
            img_radio = radio_process(img_radio, *args, weight=radio_weight, 
                                      adaptive_max=None, adaptive_damp=None, 
                                      upper_sig=9., **kwargs)

            if img_radio.shape != (height, width):
                raise ValueError("Image is not the correct size.")
            
            imwriter.add(np.stack([img_radio, img_ir]))

            # Need to keep track of successfully written file names
            success.append(f)
        
        except ValueError as ve:
            print(f"{f}: {len(failed)} {ve}")
            failed.append(f)
            reason.append(ve)
        except FileNotFoundError as fe:
            print(f"{f}: {len(failed)} {fe}")
            failed.append(f)
            reason.append(fe)

        except Exception as e:
            raise e

    print(f"Have written out {len(success)}/{len(files)} files")

    return success, failed, reason

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Run ClaRAN Source ID on a sample')
    parser.add_argument('--sample', dest='sample', help='Data sample to use',
                        default='sample.csv', type=str)
    parser.add_argument('--survey', dest='radio_survey', help='The radio survey to be processed',
                        default='FIRST', type=str)
    parser.add_argument('--ir_survey', dest='ir_survey', help='The optical/IR survey to be processed',
                        default=None, type=str)
    parser.add_argument('--image_path', dest='img_path',
                        help='Path to the directory containing the raw input images',
                        default="images", type=str)
    parser.add_argument('--output', dest='base', help='Base name for the output files',
                        default='test_prepro', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    base = args.base

    # df = pd.read_csv('../FIRST_Cata_Images_Rescanned_Success.csv')
    df = pd.read_csv(args.sample)
    # Would prefer to have a basename column
    # df["filename"] = df["island_name"]
    radio_files = df['filename'].values
    ir_files = df['ir_filename'].values

    ir_companion = {"VLASS": "unWISE-NEO4", "FIRST": "WISE-w1", "EMU_Pilot": "unWISE"}
    ir_survey = args.ir_survey
    if ir_survey is None:
        ir_survey = ir_companion[args.radio_survey]

    imgs = main(radio_files, ir_files, f'{base}_imgs.bin', 
                radio_path=args.img_path, ir_path=args.img_path,
                radio_survey=args.radio_survey, ir_survey=ir_survey)
    success_imgs, failed_imgs, reason = imgs

    # If images were not successfully dumped, then they should be excluded
    # from the catalogue. Until a newer format is supported by PINK, we 
    # have to handle this in this manner.
    sub_df = df[df['filename'].isin(success_imgs)]

    sub_df.to_csv(f'{base}.csv')
    sub_df.to_pickle(f'{base}.pkl')
    sub_df.to_json(f'{base}.json')

    sub_df = df[df['filename'].isin(failed_imgs)]
    sub_df['Fail_Reason'] = reason

    sub_df.to_csv(f'{base}_Failed.csv')
    sub_df.to_pickle(f'{base}_Failed.pkl')
    sub_df.to_json(f'{base}_Failed.json')

