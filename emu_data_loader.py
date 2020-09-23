"""Load EMU catalogues and obtain image cutouts."""

import os, sys
import numpy as np
from astropy import units as u
from astropy.table import Table
import pandas as pd
import vos


class EMUPilotSurveyHandler:
    def __init__(self, type="Component"):
        self.client = vos.Client("/cirada")
        # self.catalogues = client.listdir("vos:/cirada/data/EMU/catalogues")
        self.catalogues = list(
            client.iglob("vos:/cirada/data/EMU/catalogues/AS101*{type}*csv")
        )


def retrieve_pilot_tile(ssid, taylor=0):
    """Fetch an image cube from the CANFAR VOSpace"""
    # image.v.SB9437.cont.taylor.0.restored.fits
    filename = f"image.i.SB{ssid}.cont.taylor.{taylor}.restored.fits"
    return


def extract_cutout(image_cube):
    return
