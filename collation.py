"""Collate a sample into a source table and update the component catalogue
with the appropriate source name.

Future considerations:
- Build a table of neuron properties
- Need to decide on the list of global annotations (i.e. the morphologies
and any labels to describe neuron quality).
- How will non-global annotations (e.g. filters based on sidelobes) be
incorporated into the tables.

Still to add:
- Best host
- Host name
- Replace host coords with names for the non-best IDs
- VLASS/EMU tile ID?
"""

import os, sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.nddata.utils import Cutout2D
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from astroquery.skyview import SkyView
import networkx as nx
import pyink as pu
import pdb

# from diagnostics import plot_image, plot_source, plot_filter_idx


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


def source_name(ra, dec, aprec=2, dp=5, prefix="VLASS1QLCIR"):
    # Truncated
    ra, dec = np.array(ra), np.array(dec)
    cat = SkyCoord(ra=ra, dec=dec, unit="deg")

    astring = cat.ra.to_string(sep="", unit="hour", precision=dp, pad=True)
    dstring = cat.dec.to_string(sep="", precision=dp, pad=True, alwayssign=True)

    # Truncation index
    tind = aprec + 7
    sname = [f"{prefix} J{r[:tind]}{d[:tind]}" for r, d in zip(astring, dstring)]
    return sname


def retrieve_source(group, radio_positions, ir_positions, idx):
    G = group.graph
    subg = sorted(nx.connected_components(G), key=lambda x: len(x), reverse=True)[idx]
    subg = list(subg)

    data = [d for d in G.edges(nbunch=subg, data=True)]
    data = min(data, key=lambda x: x[2]["count"])

    ir_hosts = data[2]["IR Host"]
    bmu = tuple(data[2]["bmu"])

    components = radio_positions[np.array(subg)]
    ir_srcs = ir_positions[ir_hosts]
    # mean_component = SkyCoord(components.cartesian.mean())

    node_data = [G.nodes(data=True)[d] for d in subg]
    # onejet = any([d["onejet"] for d in node_data if "onejet" in d.keys()])
    return data, bmu, components, ir_srcs


def source_info(
    group,
    subg,
    radio_cat,
    ir_cat,
    radio_positions,
    ir_positions,
    radio_name="Component_name",
    host_name="unwise_detid",
):
    G = group.graph
    somset = group.sorter.som_set
    subg = list(subg)

    data = [d for d in G.edges(nbunch=subg, data=True)]
    data = min(data, key=lambda x: x[2]["count"])

    src_idx = data[2]["src_idx"]
    ir_hosts = data[2]["IR Host"]
    bmu = tuple(data[2]["bmu"])

    # Radio position info
    components = radio_positions[np.array(subg)]
    mean_component = SkyCoord(components.cartesian.mean())
    comp_ra = [rs.ra.value for rs in components]
    comp_dec = [rs.dec.value for rs in components]
    radio_comps = radio_cat.iloc[subg]  # networkx resets numbering
    comp_names = list(radio_comps[radio_name])

    # IR position info
    ir_srcs = ir_positions[ir_hosts]
    ir_src_ra = [irs.ra.value for irs in ir_srcs]
    ir_src_dec = [irs.dec.value for irs in ir_srcs]
    ir_host_id = list(ir_cat[host_name].iloc[ir_hosts])

    # Derived info
    total_flux = radio_comps["Total_flux"].sum()
    e_total_flux = total_flux * np.sum(
        (radio_comps.E_Total_flux / radio_comps.Total_flux) ** 2
    )
    peak_comp = radio_comps.Peak_flux.idxmax()
    euc_dist = somset.mapping.bmu_ed(src_idx)[0]

    # node_data = [G.nodes(data=True)[d] for d in subg]

    source = OrderedDict(
        RA_source=mean_component.ra.value,
        DEC_source=mean_component.dec.value,
        N_components=len(components),
        Component_names=comp_names,
        RA_components=comp_ra,
        DEC_components=comp_dec,
        N_host_candidates=len(ir_srcs),
        Host_candidates=ir_host_id,
        RA_host_candidates=ir_src_ra,
        DEC_host_candidates=ir_src_dec,
        Best_neuron=bmu,
        Euc_dist=euc_dist,
        Total_flux=total_flux,
        E_Total_flux=e_total_flux,
        Peak_flux=radio_comps.Peak_flux.loc[peak_comp],
        E_Peak_flux=radio_comps.E_Peak_flux.loc[peak_comp],
    )
    return source


def collate(group, *args, **kwargs):
    G = group.graph
    for i, subg in enumerate(
        sorted(nx.connected_components(G), key=lambda x: len(x), reverse=True)
    ):
        yield source_info(group, subg, *args, **kwargs)


def best_host(components, ir_srcs, sep_limit):
    # A host is most likely when it aligns with a radio components
    likely_host = False
    sep_limit = u.Quantity(sep_limit, u.arcsec)

    sky_matches = search_around_sky(components, ir_srcs, seplimit=15 * u.arcsecond)
    if len(sky_matches[0]) > 0:
        best_match = np.argmin(sky_matches[2])
        best_ir_match = ir_srcs[sky_matches[1][best_match]]

        # Obtained from cross-matching a shifted catalogue to WISE
        if sky_matches[2][best_match] < 3.4 * u.arcsecond:
            # ax.scatter(
            #     best_ir_match.ra,
            #     best_ir_match.dec,
            #     transform=ax.get_transform("world"),
            #     color="white",
            #     s=22,
            # )
            likely_host = True
    return likely_host


def main(
    radio_cat, ir_cat, imgs, somset, annotation, sorter_mode="area_ratio", pix_scale=0.6
):
    pix_scale = u.Quantity(pix_scale, u.arcsec)
    seplimit = 0.5 * np.max(imgs.data.shape[-2:]) * pix_scale

    radio_cat = radio_cat.loc[imgs.records]

    radio_positions = SkyCoord(radio_cat.RA, radio_cat.DEC, unit=u.deg)
    ir_positions = SkyCoord(ir_cat.ra, ir_cat.dec, unit=u.deg)

    # radio_matches = search_around_sky(
    #     radio_positions, radio_positions, seplimit=3 * u.arcmin / 2
    # )
    ir_matches = search_around_sky(radio_positions, ir_positions, seplimit=seplimit)

    # Projecting the filters
    filters = pu.FilterSet(
        radio_positions,
        (radio_positions, ir_positions),
        annotation,
        somset,
        seplimit=seplimit,
        progress=True,
        pixel_scale=pix_scale,
    )
    print(f"Number of matches are: {[len(fs[0]) for fs in filters.sky_matches]}")

    # Inspect the assigned and unassigned components for a given index
    sorter = pu.Sorter(somset, annotation, mode=sorter_mode)

    actions = pu.LabelResolve(
        {
            "Follow up": pu.Action.PASS,
            "Related Radio": pu.Action.LINK,
            "IR Host": pu.Action.DATA_ATTACH,
        }
    )
    # LINK, UNLINK, RESOLVE, FLAG, PASS, NODE_ATTACH, DATA_ATTACH, TRUE_ATTACH, FALSE_ATTACH, ISOLATE

    def src_fn(idx):
        return {"idx": idx * 2, "bmu": somset.mapping.bmu(idx)}

    group = pu.Grouper(
        filters, annotation, actions, sorter, src_stats_fn=src_fn, progress=True
    )

    source_cat = pd.DataFrame(
        collate(group, radio_cat, ir_cat, radio_positions, ir_positions)
    )
    names = source_name(source_cat.RA_source, source_cat.DEC_source, prefix="VLASS1QLC")
    source_cat.insert(0, "Source_name", names)
    # source_cat["Source_name"] = names

    return source_cat


def component_table(radio_cat, somset, name_col="Component_name"):
    bmu = somset.mapping.bmu()
    euc_dist = somset.mapping.bmu_ed()
    ind = np.arange(somset.transform.data.shape[0])
    trans = somset.transform.data[ind, bmu[:, 0], bmu[:, 1]]
    flip = trans["flip"]
    angle = trans["angle"]
    comp_tab = pd.DataFrame(
        {
            name_col: radio_cat[name_col],
            "bmu_y": bmu[:, 0],
            "bmu_x": bmu[:, 1],
            "ed": euc_dist,
            "flip": flip,
            "angle": angle,
        }
    )
    return comp_tab


def neuron_table():
    # Create neuron table
    # bmu, global text labels, any properties derived from filters
    return


if __name__ == "__main__":

    # unWISE catalogue available here:
    # http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=II/363/unwise
    radio_cat = Table.read("vlass_RA_185_190_DEC_21_26.csv", format="csv").to_pandas()
    ir_cat = Table.read("unwise_RA_185_190_DEC_21_26.csv", format="csv").to_pandas()

    # The mapping file does not necessarily correspond to the imbin used to train the SOM.
    imbin_file = "imgs_RA_185_190_DEC_21_26_imgs.bin"
    som_file = "June2_sample/unity_som_10w10h_5.bin"

    imgs = pu.ImageReader(imbin_file)
    som = pu.SOM(som_file)
    mapping = pu.Mapping("MAP_RA_185_190_DEC_21_26.bin")
    transform = pu.Transform("TRANSFORM_RA_185_190_DEC_21_26.bin")
    somset = pu.SOMSet(som, mapping, transform)

    annotation = pu.Annotator(
        som.path, results="June2_sample/unity_som_10w10h_5.bin.results.pkl"
    )

    source_cat = main(
        radio_cat, ir_cat, imgs, somset, annotation, sorter_mode="area_ratio"
    )


"""
random_idxs = np.random.randint(0, somset.mapping.data.shape[0], size=5)
best_idxs = sorter[:6]
worst_idxs = sorter[-6:]
plot_filter_idx(imgs, filters, worst_idxs[0])
"""

# data, bmu, components, ir_srcs = retrieve_source(
#     group, radio_positions, ir_positions, 0
# )
# plot_source(imgs, somset, radio_cat, data[2]["src_idx"], components, ir_srcs, 0.36)


# ant = annotation.results[bmu]
# ax = plt.imshow(ant.neuron[0], cmap="bwr")

