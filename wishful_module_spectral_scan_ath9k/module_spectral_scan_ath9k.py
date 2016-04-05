import os
import time
import logging
import random
import wishful_upis as upis
import wishful_framework as wishful_module
import queue
import numpy as np
import threading
import warnings
from .csi import receiver as csi_receiver
from .psd import scanner as psd_scanner
from .psd import helper as psd_helper
#from .psd import plotter as psd_plotter


__author__ = "M. Olbrich"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"


@wishful_module.build_module
class SpectralScanAth9kModule(wishful_module.AgentModule):
    def __init__(self):
        super(SpectralScanAth9kModule, self).__init__()

        self.log = logging.getLogger('wifi_module.main')
        self.bgd_thread = threading.Thread()
        self.bgd_run = False
        self.bgd_recvq = queue.Queue(maxsize=0)
        self.bgd_sendq = queue.Queue(maxsize=0)
        self.iface = 'wlan0'
        self.scan_mode = None
        self.ival = None


    @wishful_module.bind_function(upis.radio.receive_csi)
    def receive_csi(self):
        self.log.debug("Simple Module receives CSI on interface: {}".format(self.interface))

        # define CSI device
        csi_dev = '/dev/CSI_dev'

        # check CSI device
        if not os.path.exists(csi_dev):
            raise ValueError('Could not find CSI device: %s.' % csi_dev)

        # receive CSI
        csi = csi_receiver.scan(csi_dev=csi_dev, debug=True)

        return csi


    def psd_bgd_fun_mock(self):
        print("psd_bgd_fun_mock(): Entering.")

        while self.bgd_run:
            print("psd_bgd_fun_mock(): Looping with ival=%d" % self.ival)

            # collect all samples of last scan interval from recvq
            qsize = random.randint(8, 1024)
            recvq_pkts = np.full((56, qsize), (np.nan), dtype=np.float64)
            print("psd_bgd_fun_mock(): Getting %d pkts from recvq." % qsize)

            for i in range(0, qsize, 1):
                recvq_pkts[:,i] = -120*np.random.rand(56)

            # calculate statistics for last scan interval
            recvq_pkts_avg = 10*np.log10(np.mean(10**np.divide(recvq_pkts, 10), axis=1))
            #env = np.max(ret, axis=1)

            # upload statistics for last scan interval
            self.bgd_sendq.put(recvq_pkts_avg)
            print("psd_bgd_fun_mock(): Adding pkt to sendq. New sendq queue size: %d." % self.bgd_sendq.qsize())

            time.sleep(self.ival)


    def psd_bgd_fun(self):
        print("psd_bgd_fun(): Entering.")
        debug = False

        while self.bgd_run:
            print("psd_bgd_fun(): Looping with ival=%d." % self.ival)

            if (self.scan_mode == 'manual'):
                psd_helper.scan_dev_start(self.iface)

            psd_scanner.scan(self.iface, self.bgd_recvq, debug)

            # determine size of recvq
            qsize = self.bgd_recvq.qsize()

            # calculate statistics for last scan interval and upload
            if (qsize > 0):

                # collect all samples of last scan interval from recvq
                recvq_pkts = np.full((56, qsize), (np.nan), dtype=np.float64)
                print("psd_bgd_fun(): Getting %d pkts from recvq." % qsize)

                for i in range(0, qsize, 1):
                    psd_pkt = self.bgd_recvq.get()
                    recvq_pkts[:,i] = psd_pkt['psd_vector']

                # calculate aggregate metric(s)
                recvq_pkts_avg = 10*np.log10(np.mean(10**np.divide(recvq_pkts, 10), axis=1))
                #recvq_pkts_env = np.max(recvq_pkts, axis=1)

                # upload aggregate metric(s) for last scan interval
                self.bgd_sendq.put(recvq_pkts_avg)
                print("psd_bgd_fun(): Adding pkt to sendq. New sendq queue size: %d." % self.bgd_sendq.qsize())

            time.sleep(self.ival)


    def psd_bgd_start(self, count=8, period=255, fft_period=15, short_repeat=1):
        print("psd_bgd_start(): Entering.")

        # disable scan device and drain its queue
        psd_helper.scan_dev_stop(self.iface)
        psd_helper.scan_dev_drain(self.iface)

        # configure scan device
        psd_helper.scan_dev_configure(
            iface=self.iface,
            mode=self.scan_mode,
            count=count,
            period=period,
            fft_period=fft_period,
            short_repeat=short_repeat
        )

        if (self.scan_mode == 'background'):
            psd_helper.scan_dev_start(self.iface)

        # start background daemon
        self.bgd_run = True
        self.bgd_thread = threading.Thread(target=self.psd_bgd_fun)
        self.bgd_thread.daemon = True
        self.bgd_thread.start()


    def psd_bgd_stop(self):
        print("psd_bgd_stop(): Entering.")

        if not self.bgd_thread.is_alive():
            warnings.warn('Scanner daemon already stopped.', RuntimeWarning, stacklevel=2)
            return

        # stop background daemon
        self.bgd_run = False
        self.bgd_thread.join()

        # disable scan device and drain its queue
        psd_helper.scan_dev_stop(self.iface)
        psd_helper.scan_dev_drain(self.iface)


    @wishful_module.bind_function(upis.radio.scand_start)
    def scand_start(self, iface='wlan0', mode='background', ival=1):
        print("scand_start(): Entering.")

        if self.bgd_thread.is_alive():
            warnings.warn('Scanner daemon already running.', RuntimeWarning, stacklevel=2)
            return

        # drain receive queue
        while not self.bgd_recvq.empty():
            self.bgd_recvq.get()

        # drain send queue
        while not self.bgd_sendq.empty():
            self.bgd_sendq.get()

        # set scanning params
        self.iface = iface
        self.scan_mode = mode
        self.ival = ival

        # start backgound daemon
        if (self.scan_mode == 'background') or (self.scan_mode == 'manual'):
            self.psd_bgd_start()


    @wishful_module.bind_function(upis.radio.scand_stop)
    def scand_stop(self):
        print("scand_stop(): Entering.")

        # stop backgound daemon
        if (self.scan_mode == 'background') or (self.scan_mode == 'manual'):
            self.psd_bgd_stop()

        # drain receive queue
        while not self.bgd_recvq.empty():
            self.bgd_recvq.get()

        # drain send queue
        while not self.bgd_sendq.empty():
            self.bgd_sendq.get()


    @wishful_module.bind_function(upis.radio.scand_reconf)
    def scand_reconf(self, iface='wlan0', mode='background', ival=1):
        print("scand_reconf(): Entering.")

        # stop backgound daemon
        if self.bgd_thread.is_alive():
            self.psd_bgd_stop()

        # drain receive queue
        while not self.bgd_recvq.empty():
            self.bgd_recvq.get()

        # drain send queue
        while not self.bgd_sendq.empty():
            self.bgd_sendq.get()

        # update scanning params
        self.iface = iface
        self.scan_mode = mode
        self.ival = ival

        # start backgound daemon again
        self.psd_bgd_start()


    @wishful_module.bind_function(upis.radio.scand_read)
    def scand_read(self):
        print("scand_read(): Entering.")

        qsize = self.bgd_sendq.qsize()
        print("scand_read(): Current send queue size: %d." % qsize)

        if (qsize > 0):
            ret = np.full((56, qsize), (np.nan), dtype=np.float64)
            for i in range(0, qsize, 1):
                psd_pkt = self.bgd_sendq.get()
                ret[:,i] = psd_pkt
        else:
            ret = np.full((0), (np.nan), dtype=np.float64)

        return ret
