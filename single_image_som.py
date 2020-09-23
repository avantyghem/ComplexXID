"""Hack to create a SOM from a single image.
Intended to search for images with a similar morphology.
"""
import os, sys
import subprocess
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

import pyink as pu
import preprocessing


def preprocess(img_file):
    hdu = fits.open(img_file)
    img = np.squeeze(hdu[0].data)
    pimg = preprocessing.radio_preprocess(img, lower=3)
    return pimg


# image_file = sys.argv[1]
image_file = (
    "/home/adrian/CIRADA/Data/VLASS_QL/image_cutouts/J120937.64+222115.4_VLASS.fits"
)
outfile = "IMG_test_single_img_som.bin"

# Make the image binary
with pu.ImageWriter(outfile, 0, (500, 500), clobber=True) as pk_img:
    data = preprocess(image_file)
    pk_img.add(data, attributes=0)

# Hack together a SOM
# Pink training requires width >1
# Forcing som-width=2 trains properly, but messes with the dimensions for pu.SOM

# Do the mapping in a separate task
