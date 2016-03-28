# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:20:59 2015

@author: olbrich
"""
import numpy as np


# 802.11n parameters
SPECTRAL_HT20_FFT_SIZE = 64
SPECTRAL_HT20_NUM_BINS = 56
SPECTRAL_HT20_40_NUM_BINS = 128
SPECTRAL_HT20_40_FFT_SIZE = 128
#NL80211_CHAN_NO_HT = 0
#NL80211_CHAN_HT20 = 1
#NL80211_CHAN_HT40MINUS = 2
#NL80211_CHAN_HT40PLUS = 3


# ath9k spectral sample tlv type for numpy
DTYPE_PSD_TLV = np.dtype([
    ("type", np.uint8),
    ("length", np.uint16),
]).newbyteorder('>')


# ath9k spectral sample ht20 header type for numpy
DTYPE_PSD_HDR_HT20 = np.dtype([
    ("type", np.uint8),
    ("length", np.uint16),
    ("max_exp", np.uint8),
    ("freq", np.uint16),
    ("rssi", np.int8),
    ("noise", np.int8),
    ("max_magnitude", np.uint16),
    ("max_index", np.uint8),
    ("bitmap_weight", np.uint8),
    ("tsf", np.uint64),
]).newbyteorder('>')


# ath9k spectral sample ht40 header type for numpy
DTYPE_PSD_HDR_HT20_40 = np.dtype([
    ("type", np.uint8),
    ("length", np.uint16),
    ("channel_type", np.uint8),
    ("freq", np.uint16),
    ("lower_rssi", np.int8),
    ("upper_rssi", np.int8),
    ("tsf", np.uint64),
    ("lower_noise", np.int8),
    ("upper_noise", np.int8),
    ("lower_max_magnitude", np.uint16),
    ("upper_max_magnitude", np.uint16),
    ("lower_max_index", np.uint8),
    ("upper_max_index", np.uint8),
    ("lower_bitmap_weight", np.uint8),
    ("upper_bitmap_weight", np.uint8),
    ("max_exp", np.uint8),
]).newbyteorder('>')


# define sample type mappings
SAMPLE_HDR_DTYPE = {
    1: DTYPE_PSD_HDR_HT20,
    2: DTYPE_PSD_HDR_HT20_40,
}

SAMPLE_NUM_BINS = {
    1: SPECTRAL_HT20_NUM_BINS,
    2: SPECTRAL_HT20_40_NUM_BINS,
}

