# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:20:59 2015

@author: olbrich
"""
import queue
import numpy as np
from .helper import get_debugfs_dir
from .decoder import get_psd_vector
from .constants import SAMPLE_HDR_DTYPE, SAMPLE_NUM_BINS, DTYPE_PSD_TLV


def scan(iface='wlan0', q=queue.Queue(), debug=False):

    # process input params
    debugfs_dir = get_debugfs_dir(iface)
    scan_fn = debugfs_dir + 'spectral_scan0'

    # init return
    psd_pkt = None

    # start receiving/decoding PSD packets
    if debug:
        print("Start reading PSD data from scan device...")

    # open PSD device and read all contents
    fd = open(scan_fn, 'rb')
    buf = fd.read()
    fd.close()

    # read and decode PSD device buffer if we have data in it
    if buf:

        seek = 0
        while seek < len(buf):

            # read TLV
            tlv_buf = buf[seek:seek+3]
            tlv = np.frombuffer(tlv_buf, dtype=DTYPE_PSD_TLV)
            tlv_len = DTYPE_PSD_TLV.itemsize

            # read sample header
            dtype_psd_hdr = SAMPLE_HDR_DTYPE[int(tlv['type'])]
            hdr_len = dtype_psd_hdr.itemsize
            hdr_buf = buf[seek:seek+hdr_len]
            hdr = np.frombuffer(hdr_buf, dtype=dtype_psd_hdr)

            if debug:
                print("Reading PSD header: %s" % hdr)

            # read and decode PSD
            psd_num_bins = SAMPLE_NUM_BINS[int(hdr['type'])]
            psd_buf = buf[seek+hdr_len:seek+hdr_len+psd_num_bins]
            psd = np.frombuffer(psd_buf, dtype=np.uint8, count=psd_num_bins)
            psd_vector = get_psd_vector(hdr, psd)

            if debug:
                print("Reading PSD vector:")
                np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=120)
                print(psd_vector)

            # combine data into common structure
            dtype_psd_pkt = np.dtype([
                ("header", dtype_psd_hdr),
                ("psd_vector", np.float64, (psd_num_bins)),
            ])

            psd_pkt = np.array([
                (hdr, psd_vector),
            ], dtype=dtype_psd_pkt)

            #yield psd_pkt
            q.put(psd_pkt)

            # update seek
            sample_len = tlv_len + int(hdr['length'])   # total sample length [Byte]
            seek += sample_len                          # start of next sample

    else:
        if debug:
            print("PSD device buffer empty...")

    # finish
    if debug:
        print("Finished reading PSD data from scan device...")
