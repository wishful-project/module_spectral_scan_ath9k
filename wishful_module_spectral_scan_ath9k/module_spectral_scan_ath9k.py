import os
import time
import logging
#import random
#import time
import wishful_upis as upis
import wishful_framework as wishful_module
from .csi import receiver as csi_receiver

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2015, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


@wishful_module.build_module
class SpectralScanAth9kModule(wishful_module.AgentModule):
    def __init__(self):
        super(SpectralScanAth9kModule, self).__init__()
        self.log = logging.getLogger('wifi_module.main')
        self.channel = 1
        self.power = 1

    @wishful_module.bind_function(upis.radio.set_channel)
    def set_channel(self, channel):
        self.log.debug("Simple Module sets channel: {} on interface: {}".format(channel, self.interface))
        self.channel = channel
        return ["SET_CHANNEL_OK", channel, 0]

    @wishful_module.bind_function(upis.radio.get_channel)
    def get_channel(self):
        self.log.debug("Simple Module gets channel of interface: {}".format(self.interface))
        return self.channel

    @wishful_module.bind_function(upis.radio.set_power)
    def set_power(self, power):
        self.log.debug("Simple Module sets power: {} on interface: {}".format(power, self.interface))
        self.power = power
        return {"SET_POWER_OK_value" : power}

    @wishful_module.bind_function(upis.radio.get_power)
    def get_power(self):
        self.log.debug("Simple Module gets power on interface: {}".format(self.interface))
        return self.power

    @wishful_module.bind_function(upis.radio.receive_csi)
    def receive_csi(self, runt=60):
        self.log.debug("Simple Module receives CSI on interface: {}".format(self.interface))

        # define CSI device
        csi_dev = '/dev/CSI_dev'

        # check CSI device
        if not os.path.exists(csi_dev):
            raise ValueError('Could not find CSI device: %s.' % csi_dev)

        # receive CSI
        csi = csi_receiver.scan(csi_dev=csi_dev, debug=True)
        #csi = "I am CSI."

#        # set timer
#        start_time = time.time()
#        end_time = start_time + runt
#
#        while time.time() < end_time:
#            ret = random.randint(0,9)
#            #yield random.randint(0,9)
#            time.sleep(1.0)

        return csi
