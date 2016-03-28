# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
import math
import warnings
import numpy as np
from .constants import SPECTRAL_HT20_NUM_BINS


def get_psd_vector_ht20(hdr, buf, debug=False):

    # check input
    if (len(buf) != (SPECTRAL_HT20_NUM_BINS)):
        warnings.warn('Invalid PSD buffer length detected.', RuntimeWarning, stacklevel=2)
        return np.array([], dtype=complex)
    elif buf.dtype != np.dtype(np.uint8):
        warnings.warn('Invalid PSD buffer data type detected.', RuntimeWarning, stacklevel=2)
        return np.array([], dtype=complex)
    else:
        buf = buf.tolist()
        max_exp = int(hdr['max_exp'])
        noise = int(hdr['noise'])
        rssi = int(hdr['rssi'])
        psd_vector = np.full((SPECTRAL_HT20_NUM_BINS), (np.nan), dtype=np.float64)

    # calculate unscaled (by max_exp) sum power over all FFT bins
    sum_pwr = 10*math.log10(sum([(x << max_exp)**2 for x in buf]))

    # iterate over all FFT bins (subcarriers)
    for sc_cnt, sc_dat in enumerate(buf):
        if ( sc_dat == 0 ):
            sc_dat = 1
        if debug:
            print("Subcarrier Id %d -> noise=%d, rssi=%d, max_exp=%d, sc_dat=%d, sum_power=%d" % (sc_cnt, noise, rssi, max_exp, sc_dat, sum_pwr))
        sc_pwr = noise + rssi + 10*math.log10((sc_dat << max_exp)**2) - sum_pwr
        psd_vector[sc_cnt] = sc_pwr

    return psd_vector


def get_psd_vector_ht20_40(hdr, buf):
    pass


def get_psd_vector(hdr, buf):

    GET_PSD_VECTOR = {
        1 : get_psd_vector_ht20,
        2 : get_psd_vector_ht20_40,
    }

    psd_vector = GET_PSD_VECTOR[int(hdr['type'])](hdr, buf)
    return psd_vector
