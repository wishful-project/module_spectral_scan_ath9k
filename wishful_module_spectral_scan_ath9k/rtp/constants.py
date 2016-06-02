# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:26:47 2016

@author: olbrich
"""
import numpy as np

IEEE80211_FRAME_TYPES = {
    0: 'MGMT',
    1: 'CTRL',
    2: 'DATA',
    3: 'RSVD',
}

IEEE80211_MGMT_FRAME_SUBTYPES = {
    0: 'Association Request',
    1: 'Association Response',
    2: 'Reassociation Request',
    3: 'Reassociation Response',
    4: 'Probe Request',
    5: 'Probe Response',
    8: 'Beacon',
    9: 'ATIM',
    10: 'Disassociation',
    11: 'Authentication',
    12: 'Deauthentication',
}

IEEE80211_CTRL_FRAME_SUBTYPES = {
    10: 'PS-Poll',
    11: 'RTS',
    12: 'CTS',
    13: 'ACK',
    14: 'CF End',
    15: 'CF End + CF ACK',
}

IEEE80211_DATA_FRAME_SUBTYPES = {
    0: 'Data',
    1: 'Data+CF-Ack',
    2: 'Data+CF-Poll',
    3: 'Data+CF-Ack+CF-Poll',
    4: 'Null function',
    5: 'CF-Ack',
    6: 'CF-Poll',
    7: 'CF-Ack+CF-Poll',
    8: 'QoS Data',
    9: 'QoS Null',
    10: 'QoS Data+CF-Ack',
    11: 'QoS Data+CF-Poll',
    12: 'QoS Data+CF-Ack+CF-Poll',
    13: 'QoS CF-Poll',
    14: 'QoS CF-ACK+CF-Poll',
}

IEEE80211_FRAME_SUBTYPES = [IEEE80211_MGMT_FRAME_SUBTYPES,
                            IEEE80211_CTRL_FRAME_SUBTYPES,
                            IEEE80211_DATA_FRAME_SUBTYPES]

# Table of data rates, indexed by MCS index, bandwidth (0 for 20, 1 for 40),
# amd guard interval (0 for long, 1 for short).
MAX_MCS_INDEX = 15 # 76

IEEE80211_FLOAT_HTRATES = [
     [ [6.5, 7.2], [13.5, 15.0] ],          # MCS 0
     [ [13.0, 14.4], [27.0, 30.0] ],        # MCS 1
     [ [19.5, 21.7], [40.5, 45.0] ],        # MCS 2
     [ [26.0, 28.9], [54.0, 60.0] ],        # MCS 3
     [ [39.0, 43.3], [81.0, 90.0] ],        # MCS 4
     [ [52.0, 57.8], [108.0, 120.0] ],      # MCS 5
     [ [58.5, 65.0], [121.5, 135.0] ],      # MCS 6
     [ [65.0, 72.2], [135.0, 150.0] ],      # MCS 7
     [ [13.0, 14.4], [27.0, 30.0] ],        # MCS 8
     [ [26.0, 28.9], [54.0, 60.0] ],        # MCS 9
     [ [39.0, 43.3], [81.0, 90.0] ],        # MCS 10
     [ [52.0, 57.8], [108.0, 120.0] ],      # MCS 11
     [ [78.0, 86.7], [162.0, 180.0] ],      # MCS 12
     [ [104.0, 115.6], [216.0, 240.0] ],    # MCS 13
     [ [117.0, 130.0], [243.0, 270.0] ],    # MCS 14
     [ [130.0, 144.4], [270.0, 300.0] ],    # MCS 15
    ]

# radiotap look up table:
# bit number, field name, required alignment, field data types
# see: http://lxr.free-electrons.com/source/include/net/ieee80211_radiotap.h?v=4.1
# see: http://fossies.org/linux/wireshark/epan/dissectors/packet-ieee80211-radiotap.c
RADIOTAP_LUT = np.array([
    [0, 'TSFT', 8, [np.dtype(np.uint64)]],
    [1, 'Flags', 1, [np.dtype(np.uint8)]],
    [2, 'Rate', 1, [np.dtype(np.uint8)]],
    [3, 'Channel', 2, [np.dtype(np.uint16), np.dtype(np.uint16)]],
    [4, 'FHSS', 2, [np.dtype(np.uint8), np.dtype(np.uint8)]],
    [5, 'dBm Antenna Signal', 1, [np.dtype(np.int8)]],
    [6, 'dBm Antenna Noise', 1, [np.dtype(np.int8)]],
    [7, 'Lock Quality', 2, [np.dtype(np.uint16)]],
    [8, 'TX Attenuation', 2, [np.dtype(np.uint16)]],
    [9, 'dB TX Attenuation', 2, [np.dtype(np.uint16)]],
    [10, 'dBm TX Power', 1, [np.dtype(np.int8)]],
    [11, 'Antenna', 1, [np.dtype(np.uint8)]],
    [12, 'dB Antenna Signal', 1, [np.dtype(np.uint8)]],
    [13, 'dB Antenna Noise', 1, [np.dtype(np.uint8)]],
    [14, 'RX flags', 2, [np.dtype(np.uint16)]],
    [15, 'TX flags', 2, [np.dtype(np.uint16)]],
    [16, 'b16', 0, []],
    [17, 'Data Retries', 1, [np.dtype(np.uint8)]],
    [18, 'b18', 0, []],
    [19, 'MCS', 1, [np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8)]],
    [20, 'A-MPDU Status', 4, [np.dtype(np.uint32), np.dtype(np.uint16), np.dtype(np.uint8), np.dtype(np.uint8)]],
    [21, 'VHT information', 2, [np.dtype(np.uint16), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint16)]],
    [22, 'b22', 0, []],
    [23, 'b23', 0, []],
    [24, 'b24', 0, []],
    [25, 'b25', 0, []],
    [26, 'b26', 0, []],
    [27, 'b27', 0, []],
    [28, 'b28', 0, []],
    [29, 'Radiotap NS next', 0, []],
    [30, 'Vendor NS next', 2, [np.dtype(np.uint8), np.dtype(np.uint8), np.dtype(np.uint16)]],
    [31, 'Ext', 0, []],
], dtype=object)
