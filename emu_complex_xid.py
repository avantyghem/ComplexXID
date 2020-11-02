"""Pipeline for the EMUCat Complex-XID process

Input:
- A radio component catalogue
    - One catalogue per pipeline run? Leave chunking to the pipeline?
- A WISE catalogue covering the same field
    - Can I submit an SQL request to get one?
- A SOM and its annotations, created separately
- EMU tiles, which cutouts are extracted from

Intermediate files:
- EMU and WISE image cutouts
- Preprocessed image binary
- Map and transform files from mapping the prepro files through the SOM
- Source table for each chunk that the pipeline must loop through

Output:
- A file containing the annotation information
- A component table, containing:
    - Component name (links to the main component catalogue)
    - Best-matching neuron (x and y ID)
    - Transformations to match to best-matching neuron (flip, angle)
    - Source name (links to the source table)
- A source table, containing:
    - A source name, based on the source position
    - A list of component names belonging to the source
    - Source position, based on either the radio components or host position
    - Host galaxy ID
    - Source flux, area, extent, and major axis, 
      determined from the image using the SOM transformations
    - Source morphology (labels TBD)
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import votable

from legacy_survey_cutout_fetcher import grab_cutouts
import pyink as pu
import vlass_data_loader as vdl
import complex_xid_core as cxc

from vos import Client as VOSpace


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def src_cat_name(tile_id, path=""):
    cname = f"EMUCat_complex_xid_v1_{tile_id}.xml"
    return os.path.join(path, cname)


def comp_cat_name(tile_id, path=""):
    cname = f"EMUCat_component_info_{tile_id}.xml"
    return os.path.join(path, cname)


def read_table(infile, pandas=False):
    ext = infile.split(".")[-1]
    format = {"xml": "votable", "csv": "ascii"}
    tab = Table.read(infile, format=format[ext])
    if pandas:
        tab = tab.to_pandas()
    return tab


def stack_catalogues(stack, path=""):
    # Files to stack are in VOTable format
    for tile_cat in stack:
        new_tab = read_table(tile_cat)
        try:
            # final_cat = pd.concat([final_cat, new_tab])
            final_cat = vstack([final_cat, new_tab])
        except NameError:
            final_cat = new_tab.copy()

    # final_cat = final_cat.reset_index(drop=True)
    return final_cat


def load_tile_catalogues(
    out_cat_path, radio_cat, tile_cat, coad_summary, ir_data_path, tile_id
):
    radio_sample = None
    ir_cat = None
    return radio_sample, ir_cat


def prepare_tile_catalogues():
    return None, None


# /// Input Info \\\
overwrite = True

# Run from `Pipeline` directory
pipe_dir = os.getcwd()
base_dir = os.path.split(pipe_dir)[0]
data_path = os.path.join(base_dir, "Data")
model_path = os.path.join(base_dir, "SOM")

# Radio Input Info
radio_path = os.path.join(data_path, "EMU")
ir_cat_path = os.path.join(data_path, "WISE")
cutout_path = os.path.join(data_path, "image_cutouts")
out_cat_path = os.path.join(pipe_dir, "catalogues")
out_bin_path = os.path.join(pipe_dir, "binaries")
for out_dir in [out_cat_path, out_bin_path]:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

radio_component_catalogue = "CIRADA_VLASS1QL_table1_components_v01.csv.gz"
vlass_subtiles = "CIRADA_VLASS1QL_table3_subtile_info_v01.csv.gz"
ir_data_path = os.path.join(ir_cat_path, "coads")
coad_summary = os.path.join(ir_cat_path, "unWISE_coad_summary.csv")

# SOM Input Info
som_file = os.path.join(model_path, "complex_300px", "SOM_B3_h35_w35_vlass.bin")
annotations_file = f"{som_file}.results.pkl"

# /// Select Sample (single tile) \\\
print("Loading the VLASS catalogue, specifying complex components")
radio_cat_file = os.path.join(radio_path, radio_component_catalogue)
radio_cat = vdl.load_vlass_catalogue(
    radio_cat_file, complex=True, NN_dist=72, SNR_min=None
)

print("Creating the tile catalogue")
subtile_file = os.path.join(radio_path, vlass_subtiles)
subtile_cat = Table.read(subtile_file, format="csv").to_pandas()
tile_cat = vdl.vlass_tile(subtile_cat)

print("Loading the SOM and annotations")
som = pu.SOM(som_file)
annotation = pu.Annotator(som.path, results=annotations_file)

# sdss = sky_chunk(df, (120, 240), (-10, 50))
tiles = tile_cat.Tile.values
tiles = tiles[:10]

for tile_id in tiles:
    # Prepare data -- Make sure catalogues are saved and cutouts exist
    imbin_file, map_file, trans_file = cxc.binary_names(tile_id, path=out_bin_path)
    radio_sample, ir_cat = prepare_tile_catalogues()

    # /// Acquire Data \\\
    tile_cutout_path = os.path.join(cutout_path, tile_id)
    if not os.path.exists(imbin_file):
        if not os.path.exists(tile_cutout_path):
            os.makedirs(tile_cutout_path)

        print(
            f"Downloading any required image cutouts to the directory: {tile_cutout_path}"
        )
        grab_cutouts(
            radio_sample,
            output_dir=tile_cutout_path,
            name_col="Component_name",
            ra_col="RA",
            dec_col="DEC",
            imgsize_arcmin=5.0,
            imgsize_pix=150,
        )

for tile_id in tiles:
    # Process everything
    print(f"Processing tile {tile_id}")
    imbin_file, map_file, trans_file = cxc.binary_names(tile_id, path=out_bin_path)

    # Load EMU and WISE catalogues
    radio_sample, ir_cat = load_tile_catalogues(
        out_cat_path, radio_cat, tile_cat, coad_summary, ir_data_path, tile_id
    )

    # Preprocess, map, and collate
    comp_cat, src_cat = cxc.run_all(
        (radio_sample, ir_cat),
        som_file,
        tile_id,
        tile_cutout_path,
        bin_path=out_bin_path,
        img_size=(2, 150, 150),
        numthreads=10,
        annotation=annotation,
        remove_tile_cutouts=False,
        cpu_only=False,
        sorter_mode="area_ratio",
        pix_scale=2.0,
    )

    # /// Write to file \\\
    print("Writing the results to a file")
    comp_tab = Table.from_pandas(comp_cat)
    comp_tab.write(comp_cat_name(tile_id, path=out_cat_path))

    src_tab = Table.from_pandas(src_cat)
    src_tab.write(src_cat_name(tile_id, path=out_cat_path))


# /// Memory cleanup \\\
del radio_sample, ir_cat, src_cat
del radio_cat, som, annotation


# /// Stack source catalogues \\\
final_file = "CIRADA_VLASS1QL_table4_complex_sources_v01.xml"
tile_cats = [src_cat_name(tile_id, path=out_cat_path) for tile_id in tiles]
final_cat = stack_catalogues(tile_cats, path="")
final_cat.write(final_file, format="votable")

# /// Stack component catalogues \\\
final_file = "CIRADA_VLASS1QL_table1_components_som_v01.xml"
tile_cats = [comp_cat_name(tile_id, path=out_cat_path) for tile_id in tiles]
final_cat = stack_catalogues(tile_cats, path="")
final_cat.write(final_file, format="votable")
