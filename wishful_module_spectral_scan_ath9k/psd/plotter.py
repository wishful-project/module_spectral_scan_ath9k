# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:20:59 2015

@author: olbrich
"""
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from .constants import SPECTRAL_HT20_NUM_BINS


class Plotter():

    def __init__(self):

        # create application window
        self._win = pg.GraphicsWindow( title="ATH9K Spectral Scan")
        self._app = QtGui.QApplication.instance()

        if self._app is None:
            #pg.setConfigOption( 'background', 'w')
            #pg.setConfigOption( 'foreground', 'k')
            self._app = pg.mkQApp()

        if self._win is None:
            self._win = pg.GraphicsWindow( title="ATH9K Spectral Scan")
        self._win.clear()

        # create dummy data
        self.f = list(range(0,SPECTRAL_HT20_NUM_BINS,1))
        self.avg = -120*np.ones(SPECTRAL_HT20_NUM_BINS)
        #self.env = -120*np.ones(SPECTRAL_HT20_NUM_BINS)

        # create plot layout
        self._plt = self._win.addPlot(row=1, col=1)
        self._plt.showGrid(x = True, y = True, alpha = 0.3)
        self._plt.setTitle(title = "RX Spectrum")
        self._plt.setLabel('left', 'Receive Power [dBm]')
        self._plt.setLabel('bottom', 'Frequency [MHz]')
        self._plt.enableAutoRange(x = False, y = False)
        self._plt.setXRange(0- 0.25, 55 + 0.25)
        self._plt.setYRange(-120, -40)
        self._plt.clear()
        self._plt.plot(self.f, self.avg, pen=(0, 0, 255))
        #self._plt.plot(self.f, self.env, pen=(255, 0, 0))
        self._app.processEvents()


    def updateplot(self, avg):

        # update data
        self.avg = avg
        #self.env = env

        # update plot
        self._plt.clear()
        self._plt.plot(self.f, self.avg, pen=(0, 0, 255))
        #self._plt.plot(self.f, self.env, pen=(255, 0, 0))
        self._app.processEvents()
