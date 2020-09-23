import os, sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm import tqdm


def bad_source_name(ra, dec, prefix="VLASS1QLCIR"):
    # Rounded
    ra, dec = np.array(ra), np.array(dec)

    cat = SkyCoord(ra=ra, dec=dec, unit="deg")

    astring = cat.ra.to_string(sep="", unit="hour", precision=2, pad=True)
    dstring = cat.dec.to_string(sep="", precision=1, pad=True, alwayssign=True)

    sname = [
        f"{prefix} J{ra_str}{dec_str}" for ra_str, dec_str in zip(astring, dstring)
    ]

    return sname


def good_source_name(ra, dec, aprec=2, dp=5, prefix="VLASS1QLCIR"):
    # Truncated
    ra, dec = np.array(ra), np.array(dec)
    cat = SkyCoord(ra=ra, dec=dec, unit="deg")

    astring = cat.ra.to_string(sep="", unit="hour", precision=dp, pad=True)
    dstring = cat.dec.to_string(sep="", precision=dp, pad=True, alwayssign=True)

    ###truncation index
    tind = aprec + 7

    sname = [
        prefix + " J" + astring[i][:tind] + dstring[i][:tind]
        for i in range(len(astring))
    ]

    return sname


def make_filename(objname, survey="VLASS", format="fits"):
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


if __name__ == "__main__":
    # Run from the directory containing the cutouts
    df = pd.read_csv(sys.argv[1])
    df["bad"] = bad_source_name(df.RA, df.DEC)
    df["bad_filename"] = df["bad"].apply(make_filename)
    df["good"] = good_source_name(df.RA, df.DEC)
    df["good_filename"] = df["good"].apply(make_filename)
    missing = ~df["good_filename"].apply(os.path.exists)

    # to_rename = df[(df.good != df.bad)]
    df[missing][["RA", "DEC", "Component_name", "good_filename"]].to_csv("missing.csv")

    for i, row in tqdm(df[missing].iterrows()):
        if not os.path.exists(row.bad_filename):
            print(f"File {row.bad_filename} not found.")
        else:
            os.rename(row.bad_filename, row.good_filename)
