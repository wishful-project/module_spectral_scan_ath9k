#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
import os
import time
from csi import receiver as csi_receiver


# define receiver params
csi_dev = '/dev/CSI_dev'
runt = 60
ival = 1.0
debug = True

# check CSI device
if not os.path.exists(csi_dev):
    raise ValueError('Could not find CSI device: %s.' % csi_dev)

# set timer
start_time = time.time()
end_time = start_time + runt

print("Start receiving CSI for %d sec in %d sec intervals..." %(runt, ival))
while time.time() < end_time:
    csi = csi_receiver.scan(debug=debug)
    time.sleep(ival)

#csi_0 = csi[0].view(np.recarray)
#csi_0.header
#csi_0.csi_matrix

#csi[0]['csi_matrix']
#csi[3]['csi_matrix']
#csi[0:2]
