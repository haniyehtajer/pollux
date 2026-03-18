# -*- coding: utf-8 -*-
"""
@author: behmardaida, 12/9/2025

- library of functions for processing Gaia RVS spectra
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import os

from astropy.io import fits
#from sortedcontainers import SortedList
from scipy import interpolate
from heapq import nsmallest


def process_spectra(wl,flux,flux_err):

    # following Angelo+24 for handling nan flux values
    nans = np.isnan(flux)
    flux[nans] = np.interp(wl[nans], wl[~nans], flux[~nans])
    flux_err[nans] = 999

    # mask out Ca triplet
    ca_centers = [850, 854.4, 866.5]
    offset = 0.1
    #ca1 = np.logical_and(wl > 849.7, wl < 850)
    ca1 = np.logical_and(wl > ca_centers[0] - offset, wl < ca_centers[0] + offset)
    ca2 = np.logical_and(wl > ca_centers[1] - offset, wl < ca_centers[1] + offset)
    ca3 = np.logical_and(wl > ca_centers[2] - offset, wl < ca_centers[2] + offset)

    #ca1 = np.logical_and(wl > 849.7, wl < 849.9)
    #ca2 = np.logical_and(wl > 854.1, wl < 854.3)
    #ca3 = np.logical_and(wl > 866.1, wl < 866.3)
    ca_triplet = np.logical_or(np.logical_or(ca1, ca2), ca3)
    flux_err[ca_triplet] = 999

    return flux,flux_err

