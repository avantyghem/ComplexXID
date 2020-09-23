"""Download and stack unWISE catalogues
"""

import os, sys
import numpy as np
import pandas as pd
from astropy.table import Table
import wget
import pdb


def unwise_cat_url(coadID, band=1):
    # Build the url string for one unwise target
    caturl = (
        f"https://faun.rc.fas.harvard.edu/unwise/release/cat/{coadID}.{band}.cat.fits"
    )
    return caturl


coad_info = pd.read_csv("unWISE_coad_directory.csv")
# mask = (
#     (coad_info.cenRA >= 185)
#     & (coad_info.cenRA <= 265)
#     & (coad_info.cenDec >= 21)
#     & (coad_info.cenDec <= 39)
# )
# coads = coad_info[mask]["Coad_ID"].values
coads = coad_info["Coad_ID"].values

# cols = ['x', 'y', 'flux', 'dx', 'dy', 'dflux', 'qf', 'rchi2', 'fracflux',
#        'sky', 'ra', 'dec']

for i, coad in enumerate(coads):
    if i % 25 == 0:
        print(f"Iteration {i}/{len(coads)}")

    url = unwise_cat_url(coad, band=1)

    coad_file = f"{coad}.1.cat.fits"
    if not os.path.exists(coad_file):
        wget.download(url)

    new_df = Table.read(coad_file).to_pandas()
    # new_df = new_df[cols]
    try:
        df = pd.concat([df, new_df])
    except NameError:
        df = new_df.copy()
    # os.remove(coad_file)

df.to_csv("../unwise_cat.csv", index=False)
# Table.from_pandas(df).write("unwise_cat.fits")
