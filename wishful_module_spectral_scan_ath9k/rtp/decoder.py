# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:32:46 2016

@author: olbrich
"""
import math
import collections
import numpy as np
from astropy.table import Table
from .constants import RADIOTAP_LUT


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


def decode_field(index, fm, offset):

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


def get_radiotap_info(frame, offset, debug):

    # init container for decoded radiotap header
    rtp_info = collections.OrderedDict()

    # decode radiotap header
    hdr, offset = decode_header(frame, offset)
    rtp_info['hdr'] = hdr

    # decode radiotap presence mask(s) (32 bits each)
    presence_mask, offset = decode_presence_mask(frame, offset)
    rtp_info['pmask'] = presence_mask

    if debug:
        print("Radiotap Header--> ver: %d -- pad: %d -- len: %d" % (hdr['ver'], hdr['pad'], hdr['len']))
        print("Radiotap Presence Mask:")
        for pmask in presence_mask:
            print(RADIOTAP_LUT[pmask, 1])
        print("\n")

    # decode present radiotap fields
    rtp_info['dat'] = [None] * len(presence_mask)

    for pmask_ix, pmask in enumerate(presence_mask):
        present_fix = RADIOTAP_LUT[pmask, 0]
        rtp_info['dat'][pmask_ix] = collections.OrderedDict()
        rtp_info['dat'][pmask_ix]['fnames'] = []
        rtp_info['dat'][pmask_ix]['fdata'] = []
        for fix in present_fix:
            field_name = RADIOTAP_LUT[fix, 1]
            field_data, offset = decode_field(fix, frame, offset)
            rtp_info['dat'][pmask_ix]['fnames'].append(field_name)
            rtp_info['dat'][pmask_ix]['fdata'].append(field_data)
            if debug:
                print("Radiotap Field--> %s:" % field_name)
                print(Table(field_data))
                print("\n")

    return rtp_info
