# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:20:59 2015

@author: olbrich
"""
import numpy as np


# CSI packet header type for numpy
DTYPE_CSI_HDR = np.dtype([
    ("tstamp", np.uint64),
    ("csi_len", np.uint16),
    ("channel", np.uint16),
    ("phyerr", np.uint8),
    ("noise", np.uint8),
    ("rate", np.uint8),
    ("chanbw", np.uint8),
    ("num_tones", np.uint8),
    ("nr", np.uint8),
    ("nc", np.uint8),
    ("rssi", np.uint8),
    ("rssi_0", np.uint8),
    ("rssi_1", np.uint8),
    ("rssi_2", np.uint8),
    ("pld_len", np.uint8),
])

# CSI bandwidth codes
CSI_BWS = [20, 40]


# CSI HT20 rate codes (800ns GI)
CSI_HT20_RATES = [6.5, 13, 19.5, 26, 39, 52, 58.5, 65,
                  13, 26, 39, 52, 78, 104, 117, 130,
                  19.5, 39, 58.5, 78, 117, 156, 175.5, 195]


# CSI HT40 rate codes (800ns GI)
CSI_HT40_RATES = [13.5, 27, 40.5, 54, 81, 108, 121.5, 135,
                  27, 54, 81, 108, 162, 216, 243, 270,
                  40.5, 81, 121.5, 162, 243, 324, 364.5, 405]


# CSI number streams codes
CSI_NUM_STREAMS = [1, 1, 1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2, 2, 2,
                   3, 3, 3, 3, 3, 3, 3, 3]


# CSI PHY error codes
CSI_PHY_ERRS = ['success',
                'timing error',
                'illegal parity',
                'illegal rate',
                'illegal length',
                'radar detect',
                'illegal service',
                'transmit override receive']
