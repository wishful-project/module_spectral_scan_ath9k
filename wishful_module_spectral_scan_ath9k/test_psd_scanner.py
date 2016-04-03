#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
#import os
import time
import queue
import numpy as np
from psd import scanner as psd_scanner
from psd import helper as psd_helper
from psd import plotter as psd_plotter


# define scan params
iface = 'wlan0'
runt = 30
ival = 0.05
debug = False
spectral_mode = 'background'    # 'background', 'manual'
spectral_count = 8              # default=8
spectral_period = 255           # default=255, max=255
spectral_fft_period = 15        # default=15, max=15
spectral_short_repeat = 1       # default=1


# configure scan device
psd_helper.scan_dev_configure(
    iface=iface,
    mode=spectral_mode,
    count=spectral_count,
    period=spectral_period,
    fft_period=spectral_fft_period,
    short_repeat=spectral_short_repeat
)

# set timer
start_time = time.time()
end_time = start_time + runt

# create plotter object
plt = psd_plotter.Plotter()

# create scanner queue
psdq = queue.Queue()

# start scan
print("Start scanning PSD for %d sec in %d sec intervals on interface %s..." % (runt, ival, iface))
if (spectral_mode == 'background'):
    psd_helper.scan_dev_start(iface)

ival_cnt = 0
while time.time() < end_time:

    if (spectral_mode == 'manual'):
        psd_helper.scan_dev_start(iface)

    # start scanner in a separate thread???
    psd_scanner.scan(iface, psdq, debug)

    # collect all samples from last scan interval
    qsize = psdq.qsize()
    print("ival_cnt: %d, qsize: %d" % (ival_cnt, qsize))
    ret = np.full((56, qsize), (np.nan), dtype=np.float64)

    #while not myQueue.empty():
        #psd_pkt = myQueue.get()
        #print("Receiving PSD header: %s" % psd['header'])

    for i in range(0, qsize, 1):
        psd_pkt = psdq.get()
        ret[:,i] = psd_pkt['psd_vector']

    # calculate statistics for last scan interval
    avg = 10*np.log10(np.mean(10**np.divide(ret, 10), axis=1))
    env = np.max(ret, axis=1)

    if debug:
        print("Average power spectrum of last %d samples (%d sec):" % (qsize, ival))
        np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=120)
        print(avg)

        print("Envelope power spectrum of last %d samples (%d sec):" % (qsize, ival))
        np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=120)
        print(env)

    # update plotter
    plt.updateplot(avg, env)

    # sleep for ival seconds
    ival_cnt += 1
    time.sleep(ival)

# stop scan
psd_helper.scan_dev_stop(iface)
