import os, sys
import numpy as np
from astropy import units as u
from astropy.table import Table
import pandas as pd


def build_ir_cat(ir_tile_info, ir_data_path, ra_bounds, dec_bounds):
    # 861k rows in a 5x5 degree region
    cols = ["ra", "dec", "unwise_detid", "flux"]
    ra_min, ra_max = ra_bounds
    dec_min, dec_max = dec_bounds
    info = pd.read_csv(ir_tile_info)
    mask = (
        (info.cenRA >= ra_min)
        & (info.cenRA <= ra_max)
        & (info.cenDec >= dec_min)
        & (info.cenDec <= dec_max)
    )
    info = info[mask]
    coads = info.Coad_ID.values
    for coad in coads:
        coad_file = os.path.join(ir_data_path, f"{coad}.1.cat.fits")
        new_df = Table.read(coad_file).to_pandas()
        try:
            df = pd.concat([df, new_df])
        except NameError:
            df = new_df.copy()
    return df


def sky_chunk_something(
    ir_tile_info, ir_data_path, ra_range=(190, 195), dec_range=(24, 29)
):
    """Load radio and IR catalogues for a small region on the sky. 
    The ImageReader binary file will (probably) correspond to the
    catalogue covering the entire sky. Will need to be
    careful about sources at the edge of each chunk.
    Could provide a 1.5' overlap in the chunk edges to handle these
    cases.
    Start off with a fixed RA/Dec limit that is manually chosen 
    and has a similar size as the VLASS QL images.
    Then update to use only components within a single QL image.
    Consider creating Image Binaries for each tile.
    """
    ra_min, ra_max = ra_range
    dec_min, dec_max = dec_range
    return radio_cat, ir_cat


def sky_chunk(df, ra_range=(190, 195), dec_range=(24, 29)):
    """Restrict a DataFrame to an RA and Dec range
    """
    ra_min, ra_max = ra_range
    dec_min, dec_max = dec_range
    bounds = (
        (df.RA > ra_min) & (df.RA < ra_max) & (df.DEC > dec_min) & (df.DEC < dec_max)
    )
    df = df[bounds]
    return df


def load_tile(tile_id):
    """Load the VLASS and unWISE catalogues for a single VLASS tile.
    """
    return


def vlass_tile(subtiles):
    def tile_info(subtiles, tile):
        t = subtiles[subtiles.Tile == tile]
        return dict(
            Tile=tile,
            min_ra=np.min(t.min_ra),
            max_ra=np.max(t.max_ra),
            min_dec=np.min(t.min_dec),
            max_dec=np.max(t.max_dec),
        )

    st = subtiles[
        [
            "Tile",
            "NAXIS1",
            "NAXIS2",
            "CRVAL1",
            "CDELT1",
            "CRPIX1",
            "CRVAL2",
            "CDELT2",
            "CRPIX2",
        ]
    ].copy()

    st["min_ra"] = st.CRVAL1 + (0 - st.CRPIX1) * np.abs(st.CDELT1)
    st["max_ra"] = st.CRVAL1 + (st.NAXIS1 - st.CRPIX1) * np.abs(st.CDELT1)
    st["min_dec"] = st.CRVAL2 + (0 - st.CRPIX2) * np.abs(st.CDELT2)
    st["max_dec"] = st.CRVAL2 + (st.NAXIS2 - st.CRPIX2) * np.abs(st.CDELT2)

    tiles = [tile_info(st, tile) for tile in np.unique(st.Tile)]
    tiles = pd.DataFrame(tiles)
    return tiles


def load_vlass_catalogue(catalog, complex=True, **kwargs):
    fmt = "fits" if catalog.endswith("fits") else "csv"
    df = Table.read(catalog, format=fmt).to_pandas()
    # df = df[(df.Quality_flag == 0) & (df.Duplicate_flag < 2)]
    df = df[df.Total_flux >= df.Peak_flux]
    df = df[df.Peak_flux >= 5 * df.Isl_rms]
    df = df[df.Duplicate_flag < 2]
    df["SNR"] = np.array(df.Total_flux / df.Isl_rms)
    if complex:
        df = complex_vlass(df, **kwargs)
    return df


def complex_vlass(df, NN_dist=72, SNR_min=None):
    """Algorithm to select complex VLASS components.

    Args:
        df ([pandas.DataFrame]): Table of VLASS components.
        NN_dist ([float]], optional): Maximum distance (in arcsec) to the nearest component to be considered complex. Defaults to 72.
        SNR_min ([float]], optional): Minimum signal-to-noise of the component. Defaults to None.

    Returns:
        pandas.DataFrame: Subset of the input DataFrame containing only the complex components.
    """
    mask = (df.S_Code == "S") & (df.NN_dist < NN_dist)
    mask |= df.S_Code == "M"
    mask |= df.S_Code == "C"
    df = df[mask]

    if SNR_min is not None:
        df = df[df.SNR >= SNR_min]

    return df
