# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:20:34 2016

@author: olbrich
"""
import warnings
from .constants import CSI_BWS, CSI_HT20_RATES, CSI_HT40_RATES, CSI_PHY_ERRS


def map_csi_pkt_bw(chanbw):
    chanbw_codes = [0, 1]

    if chanbw in chanbw_codes:
        return CSI_BWS[chanbw]  # MHz
    else:
        warnings.warn('Invalid CSI chanBW code.', RuntimeWarning, stacklevel=2)
        return None


def map_csi_pkt_rate(rate, chanbw):
    rate_codes = range(128, 152, 1)
    chanbw_codes = [0, 1]
    rates = [CSI_HT20_RATES, CSI_HT40_RATES]

    if rate in rate_codes:
        if chanbw in chanbw_codes:
            return rates[chanbw][rate_codes.index(rate)]  # Mbps
        else:
            warnings.warn('Invalid CSI chanBW code.', RuntimeWarning, stacklevel=2)
            return None
    else:
        warnings.warn('Invalid CSI rate code.', RuntimeWarning, stacklevel=2)
        return None


def map_csi_pkt_phyerr(phyerr):
    codes = range(0, 8, 1)

    if phyerr in codes:
        return CSI_PHY_ERRS[codes.index(phyerr)]
    else:
        warnings.warn('Invalid CSI phyerr code.', RuntimeWarning, stacklevel=2)
        return None
