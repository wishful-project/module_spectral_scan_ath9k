# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
import os
import numpy as np
from .decoder import get_csi_matrix
from .constants import DTYPE_CSI_HDR


def scan(csi_dev='/dev/CSI_dev', debug=False):

    # init return
    csi_pkt = None

    # start receiving/decoding CSI packets
    if debug:
        print("Start reading CSI data from CSI device...")

    # open CSI device and read CSI header
    # here we need buffered I/O with Python 3, otherwise get always segfaults
    fd = open(csi_dev, 'rb')
    fd.seek(0, os.SEEK_SET)
    buf = fd.read(24)

    # read and decode CSI device buffer if we have data in it
    if buf:

        # decode packet header
        hdr = np.frombuffer(buf, dtype=DTYPE_CSI_HDR)

        if debug:
            print("Receiving CSI header: %s" % hdr)

        # open CSI device and read data from device directly
        # here we need direct I/O with Python 3, otherwise NumPy complaints
        fdd = open(csi_dev, 'rb', buffering=0)
        csi_len = hdr[0][1]
        fdd.seek(25, os.SEEK_SET)
        csi = np.fromfile(fdd, dtype=np.uint8, count=csi_len)

        # calculate CSI matrix
        nr = hdr[0][8]
        nc = hdr[0][9]
        num_tones = hdr[0][7]
        csi_matrix = get_csi_matrix(csi, nr, nc, num_tones)

        if debug:
            print("Receiving CSI matrix:")
            np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=120)
            print(csi_matrix)

        # read payload data
        pld_len = hdr[0][14]
        fdd.seek(25 + csi_len, os.SEEK_SET)
        pld = np.fromfile(fdd, dtype=np.uint8, count=pld_len)

        # combine data into common structure
        dtype_csi_pkt = np.dtype([
            ("header", DTYPE_CSI_HDR),
            ("csi_matrix", np.complex, (nr, nc, num_tones)),
            ("payload", np.uint8, (pld_len,)),
        ])

        csi_pkt = np.array([
            (hdr, csi_matrix, pld),
        ], dtype=dtype_csi_pkt)

        # close CSI device
        fdd.close()

    else:
        if debug:
            print("CSI device buffer empty...")

    # close CSI device
    fd.close()

    # finish
    if debug:
        print("Finished reading CSI data from CSI device...")

    return csi_pkt
