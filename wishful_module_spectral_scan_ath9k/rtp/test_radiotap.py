#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
#import os
from scapy.all import *
import numpy as np
import math
from astropy.table import Table
import collections
import queue



# define scan params
IFACE = 'mon0'
MYADDR = '04:f0:21:1e:70:7f' # srv1
DEBUG = False



###################################################################################################
# definitions
###################################################################################################

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


def decode_header(fm, offset):
    dtype = np.dtype([
        ("ver", np.uint8),
        ("pad", np.uint8),
        ("len", np.uint16), # length of the whole header in bytes, including it_version, it_pad, it_len, and data fields
    ])

    ret = np.frombuffer(fm[offset:offset+dtype.itemsize], dtype=np.dtype(dtype))
    offset += dtype.itemsize
    return ret, offset


def decode_presence_mask(fm, offset):
    # decode radiotap header presence mask (32 bits)
    # see: http://www.radiotap.org/
    # see: http://static.askapache.com/wireshark-1.1.3/epan/dissectors/packet-radiotap.c
    # see: https://github.com/boundary/wireshark/blob/master/epan/dissectors/packet-ieee80211-radiotap.c
    # see: https://libtins.github.io/docs/latest/d7/d0e/classTins_1_1RadioTap.html
    mask_len = 4 # bytes
    presence_mask_ext = True
    ret = []

    while presence_mask_ext:
        buf = bytearray(b'')
        mask_raw = np.frombuffer(fm[offset:offset+mask_len], dtype=np.uint32)

        for i in range(0, 32, 1):
            buf += np.bool_(int(mask_raw) & (1 << i) != 0).tobytes()

        ret_tmp = np.frombuffer(buf, dtype=bool)
        ret.append(ret_tmp)
        offset += mask_len
        presence_mask_ext = ret_tmp[31]

    return ret, offset


def decode_dummy(fm, offset, bitnum):
    req_alignment = RADIOTAP_LUT[bitnum, 2] # bytes
    in_dtypes = RADIOTAP_LUT[bitnum, 3]
    out_dtypes = []
    out_dtype_cnt = 0
    buf = bytearray(b'')
    for in_dtype in in_dtypes:
        pad_bytes = int((math.ceil(offset / req_alignment) - (offset / req_alignment)) * req_alignment)
        offset += pad_bytes
        in_buf_len = in_dtype.itemsize
        buf += fm[offset : offset + in_buf_len]
        out_dtypes.append(('d%d' % out_dtype_cnt, in_dtype))
        offset += in_buf_len
        out_dtype_cnt += 1

    out_dtype = np.dtype(
        out_dtypes
    )

    if buf:
        ret = np.frombuffer(buf, dtype=out_dtype)
    else:
        ret = None
    return ret, offset


def get_field_data(fm, offset, bitnum):
    req_alignment = RADIOTAP_LUT[bitnum, 2] # bytes
    in_dtypes = RADIOTAP_LUT[bitnum, 3]
    in_buf = []
    for in_dtype in in_dtypes:
        pad_bytes = int((math.ceil(offset / req_alignment) - (offset / req_alignment)) * req_alignment)
        offset += pad_bytes
        in_buf_len = in_dtype.itemsize
        in_buf.append(np.frombuffer(fm[offset : offset + in_buf_len], dtype=in_dtype))
        offset += in_buf_len

    return in_buf, offset


def decode_tsft(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("mactime [usec]", np.uint64),
    ])

    buf += in_buf[0].tobytes()
    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_flags(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("CFP", np.bool),
        ("Short Preamble", np.bool),
        ("WEP", np.bool),
        ("Fragmentation", np.bool),
        ("FCS at end", np.bool),
        ("Data Pad", np.bool),
        ("Bad FCS", np.bool),
        ("Short GI", np.bool),
    ])

    buf += np.bool_(int(in_buf[0]) & 0x01).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x02).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x04).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x08).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x10).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x20).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x40).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x80).tobytes()

    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_rate(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("tx/rx rate [Mbps]", np.uint8),
    ])

    buf += np.uint8(in_buf[0] / 2).tobytes()

    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_channel(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("Frequency [MHz]", np.uint16),
        ("Turbo", np.bool_),
        ("CCK", np.bool_),
        ("OFDM", np.bool_),
        ("2 GHz spectrum", np.bool_),
        ("5 GHz spectrum", np.bool_),
        ("Passive", np.bool_),
        ("Dynamic CCK-OFDM", np.bool_),
        ("GFSK (FHSS PHY)", np.bool_),
    ])

    buf += in_buf[0].tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0010).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0020).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0040).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0080).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0100).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0200).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0400).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x0800).tobytes()

    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_dbm_antenna_signal(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("SSI Signal [dBm]", np.int8),
    ])

    buf += in_buf[0].tobytes()
    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_antenna_index(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("Antenna Index", np.uint8),
    ])

    buf += in_buf[0].tobytes()
    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_rxflags(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("Bad PLCP", np.bool),
    ])

    buf += np.bool_(int(in_buf[0]) & 0x0002).tobytes()

    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_txflags(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("excessive retries", np.bool),
        ("CTS", np.bool),
        ("RTS/CTS handshake", np.bool),
        ("no ACK", np.bool),
        ("pre-configured SEQ", np.bool),
    ])

    buf += np.bool_(int(in_buf[0]) & 0x0001).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x0002).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x0004).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x0008).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x0010).tobytes()

    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_retries(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("retries", np.uint8),
    ])

    buf += in_buf[0].tobytes()
    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def decode_mcs(fm, offset, bitnum):
    in_buf, offset = get_field_data(fm, offset, bitnum)
    buf = bytearray(b'')
    out_dtype = np.dtype([
        ("Bandwidth known", np.bool_),
        ("MCS known", np.bool_),
        ("Guard Interval known", np.bool_),
        ("HT Format known", np.bool_),
        ("FEC type known", np.bool_),
        ("STBC known", np.bool_),
        ("Ness known", np.bool_),
        ("Ness data known", np.bool_),
        ("Bandwidth", np.uint8),
        ("Guard Interval", np.uint8),
        ("HT format", np.uint8),
        ("FEC type", np.uint8),
        ("STBC streams ", np.uint8),
        ("Ness - bit 0", np.uint8),
        ("MCS index", np.uint8),
    ])

    buf += np.bool_(int(in_buf[0]) & 0x01).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x02).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x04).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x08).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x10).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x20).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x40).tobytes()
    buf += np.bool_(int(in_buf[0]) & 0x80).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x03).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x04).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x08).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x10).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x60).tobytes()
    buf += np.bool_(int(in_buf[1]) & 0x80).tobytes()
    buf += in_buf[2].tobytes()

    ret = np.frombuffer(buf, dtype=out_dtype)
    return ret, offset


def radiotap_decode_field(index, fm, offset):

    RADIOTAP_DECODE = {
        0 :  decode_tsft,
        1 :  decode_flags,
        2 :  decode_rate,
        3 :  decode_channel,
        4 :  decode_dummy,
        5 :  decode_dbm_antenna_signal,
        6 :  decode_dummy,
        7 :  decode_dummy,
        8 :  decode_dummy,
        9 :  decode_dummy,
        10 : decode_dummy,
        11 : decode_antenna_index,
        12 : decode_dummy,
        13 : decode_dummy,
        14 : decode_rxflags,
        15 : decode_txflags,
        16 : decode_dummy,
        17 : decode_retries,
        18 : decode_dummy,
        19 : decode_mcs,
        20 : decode_dummy,
        21 : decode_dummy,
        22 : decode_dummy,
        23 : decode_dummy,
        24 : decode_dummy,
        25 : decode_dummy,
        26 : decode_dummy,
        27 : decode_dummy,
        28 : decode_dummy,
        29 : decode_dummy,
        30 : decode_dummy,
        31 : decode_dummy,
    }

    return RADIOTAP_DECODE[index](fm, offset, index)


def sniffer(myaddr, ret_q, debug):
    def __sniffer(fm):

        # check if we got a radiotap frame
        if fm.haslayer(RadioTap):
            if debug:
                print("Got Radiotap header.")
                print("--------------------------------------------------------------------------------")
                np.set_printoptions(formatter={'float': '{: 0.1f}'.format, 'bool': '{: 0.0f}'.format}, linewidth=160)

            # get raw data of the whole frame
            fm_raw = bytes(fm)

            # init data container fur current radiotap frame
            rtp_frame_data = collections.OrderedDict()

            # decode radiotap header from packet raw data
            offset = 0
            rtp_hdr, offset = decode_header(fm_raw, offset)
            rtp_frame_data['hdr'] = rtp_hdr

            # decode radiotap header presence mask (32 bits)
            rtp_presence_mask, offset = decode_presence_mask(fm_raw, offset)
            rtp_frame_data['pmask'] = rtp_presence_mask

            if debug:
                print("Radiotap Header--> ver: %d -- pad: %d -- len: %d" % (rtp_hdr['ver'], rtp_hdr['pad'], rtp_hdr['len']))
                print("Radiotap Presence Mask:")
                for pmask in rtp_presence_mask:
                    print(RADIOTAP_LUT[pmask, 1])
                print("\n")

            # decode present radiotap fields
            rtp_frame_data['dat'] = [None] * len(rtp_presence_mask)

            for pmask_ix, pmask in enumerate(rtp_presence_mask):
                present_fix = RADIOTAP_LUT[pmask, 0]
                rtp_frame_data['dat'][pmask_ix] = collections.OrderedDict()
                rtp_frame_data['dat'][pmask_ix]['fnames'] = []
                rtp_frame_data['dat'][pmask_ix]['fdata'] = []
                for fix in present_fix:
                    field_name = RADIOTAP_LUT[fix, 1]
                    field_data, offset = radiotap_decode_field(fix, fm_raw, offset)
                    rtp_frame_data['dat'][pmask_ix]['fnames'].append(field_name)
                    rtp_frame_data['dat'][pmask_ix]['fdata'].append(field_data)
                    if debug:
                        print("Radiotap Field--> %s:" % field_name)
                        print(Table(field_data))
                        print("\n")

            if debug:
                print("--------------------------------------------------------------------------------\n")


            # calculate the 802.11 radio frame length and datarate
            radio_frame_len = int(len(fm_raw) - rtp_hdr['len']) * 8 # bits

            rtp_ix = 0 # only use the base header, not the extension headers
            haveRate = rtp_frame_data['pmask'][rtp_ix][2] # legacy field, only valid for non-HT transmission
            haveMCS = rtp_frame_data['pmask'][rtp_ix][19]

            if haveRate:
                field_ix = rtp_frame_data['dat'][rtp_ix]['fnames'].index('Rate')
                datarate = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['tx/rx rate [Mbps]'][0] * 1e6 # bits/sec
            elif haveMCS:
                # for datarate calculation we need:
                # 1) MCS index
                # 2) guard interval length (0: long GI, 1: short GI)
                # 3) channel bandwidth (0: 20, 1: 40, 2: 20L, 3: 20U )
                field_ix = rtp_frame_data['dat'][rtp_ix]['fnames'].index('MCS')
                mcs_known = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['MCS known'][0]
                bw_known = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['Bandwidth known'][0]
                gi_known = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['Guard Interval known'][0]

                # If we have the MCS index, channel width, and
                # guard interval length, and the MCS index is
                # valid, we can compute the rate.  If the resulting
                # rate is non-zero, report it.  (If it's zero,
                # it's an MCS/channel width/GI combination that
                # 802.11n doesn't support.)
                can_calculate_rate = True
                mcs_index = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['MCS index'][0]
                bw_index = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['Bandwidth'][0]
                gi_index = rtp_frame_data['dat'][rtp_ix]['fdata'][field_ix]['Guard Interval'][0]

                if not (mcs_known & mcs_index):
                    can_calculate_rate = False
                if not (mcs_known & bw_index & bw_known):
                    can_calculate_rate = False
                if not (mcs_known & gi_index & gi_known):
                    can_calculate_rate = False

                if (can_calculate_rate & mcs_index <= MAX_MCS_INDEX):
                    datarate = IEEE80211_FLOAT_HTRATES[mcs_index][bw_index][gi_index] * 1e6 # bits/sec
                else:
                    datarate = 0.0

            # calculate the estimated airtime
            # TODO: what about Frame Control Field duration flag?
            airtime = (radio_frame_len / datarate) * 1e6 # usec

            if debug:
                print('Radio frame datarate: %.2f Mbps -- airtime: %.2f usec' % (datarate/1e6, airtime) )

        # check if we got a 802.11 radio frame
        if fm.haslayer(Dot11):

            if debug:
                print("Got Dot11 header.")
                print("--------------------------------------------------------------------------------")

                # decode 802.11 frame type
                frame_type_str = IEEE80211_FRAME_TYPES[fm.type]
                frame_subtype_str = IEEE80211_FRAME_SUBTYPES[fm.type][fm.subtype]

#            # scapy Frame Control Field
#            DS = fm.FCfield & 0x3
#            toDS = DS & 0x1 != 0
#            fromDS = DS & 0x2 != 0

            # MAC addresses (non-WDS, i.e- To-DS=1, From-DS=0)
            rx_addr = fm.addr1
            tx_addr = fm.addr2

#            if (toDS == False) and (fromDS == False):
#                dst_addr = fm.addr1
#                src_addr = fm.addr2
#                bss_addr = fm.addr3
#            elif (toDS == False) and (fromDS == True):
#                dst_addr = fm.addr1
#                src_addr = fm.addr3
#                bss_addr = fm.addr2
#            elif (toDS == True) and (fromDS == False):
#                dst_addr = fm.addr3
#                src_addr = fm.addr2
#                bss_addr = fm.addr1
#            else:
#                dst_addr = None
#                src_addr = None
#                bss_addr = None

            if (myaddr in [rx_addr, tx_addr]) or (rx_addr == 'ff:ff:ff:ff:ff:ff'):
                print("Radio frame from or to me -- RX: %s -- TX: %s" %(rx_addr, tx_addr))
                ret_q.put([0, airtime])
            else:
                print("Radio frame not for me -- RX: %s -- TX: %s" %(rx_addr, tx_addr))
                ret_q.put([1, airtime])

            if debug:
                #print("Dot11 %s/%s frame--> From: %s -- To: %s -- BSSID: %s" % (frame_type_str, frame_subtype_str, src_addr, dst_addr, bss_addr))
                print("Dot11 %s/%s frame--> RX addr: %s -- TX addr: %s" % (frame_type_str, frame_subtype_str, rx_addr, tx_addr))
                print("--------------------------------------------------------------------------------\n")

    return __sniffer


###############################################################################
# main
###############################################################################
q_airtime = queue.Queue(maxsize=0)

#packets = sniff(offline='fio_tcptest.pcapng', prn=sniffer(MYADDR, q_airtime, DEBUG), count=100)
#packets = sniff(offline='fio_tcptest.pcapng', prn=sniffer(MYADDR, q_airtime, DEBUG))
sniff(iface=IFACE, prn=sniffer(MYADDR, q_airtime, DEBUG), count=200)

# calculate total airtime
qsize_airtime = q_airtime.qsize()
airtime_intern = 0
airtime_extern = 0
num_frames = 0

for i in range(0, qsize_airtime, 1):
    dat = q_airtime.get()
    num_frames += 1
    if (dat[0] == 0):
        airtime_intern += dat[1]
    elif (dat[0] == 1):
        airtime_extern += dat[1]

print('Received number frames: %d' % num_frames)
print('Total airtime intern: %.2f usec' % airtime_intern)
print('Total airtime extern: %.2f usec' % airtime_extern)
