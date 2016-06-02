#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
import time
import queue
from scapy.all import sniff
from threading import Thread
from rtp.sniffer import get_airtime


# define scan params
IFACE = 'mon0'
MYADDR = '04:f0:21:1e:70:7f' # srv1
DEBUG = False


###############################################################################
# functions
###############################################################################
def airtime_sniffer(myaddr, q, debug):
  #sniff(iface=IFACE, prn=get_airtime(myaddr, q, debug), count=200)
  #sniff(iface=IFACE, prn=get_airtime(myaddr, q, debug))
  sniff(offline='fio_tcptest.pcapng', prn=get_airtime(myaddr, q, debug), count=10000)


###############################################################################
# main
###############################################################################

# create sniffer queue and start it in a separate thread
q = queue.Queue(maxsize=0)
sniffer = Thread(target = airtime_sniffer, args = (MYADDR, q, DEBUG,))
sniffer.daemon = True
sniffer.start()

while True:

    # calculate total airtime
    qsize= q.qsize()

    airtime_intern = 0
    airtime_extern = 0
    num_frames = 0

    for i in range(0, qsize, 1):
        dat = q.get()
        num_frames += 1
        if (dat[0] == 0):
            airtime_intern += dat[1]
        elif (dat[0] == 1):
            airtime_extern += dat[1]

    print('Received number frames: %d' % num_frames)
    print('Total airtime intern: %.2f usec' % airtime_intern)
    print('Total airtime extern: %.2f usec' % airtime_extern)
    print('')

    time.sleep(1)
