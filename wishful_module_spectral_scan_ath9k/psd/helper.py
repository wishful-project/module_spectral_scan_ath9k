# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:20:59 2015

@author: olbrich
"""
import os


def file_write(fn, msg):
    f = open(fn, 'w')
    f.write(msg)
    f.close()
    return None


def get_device_driver(iface):
    """ get driver name for this interface """
    dn = '/sys/class/net/' + iface + '/device/driver/module'
    if ( os.path.isdir(dn) ):
        driver = os.path.split(os.readlink(dn))[1]
        return driver
    return None


def get_device_phyid(iface):
    ''' get phy id for this interface '''
    fn = '/sys/class/net/' + iface + '/phy80211/name'
    if ( os.path.isfile(fn) ):
        f = open(fn, 'r')
        phyid = f.read().strip()
        f.close()
        return phyid
    return None


def get_debugfs_dir(iface):
    ''' get debugfs directory for this interface '''
    phy_id = get_device_phyid(iface)
    driver = get_device_driver(iface)
    debugfs_dir = '/sys/kernel/debug/ieee80211/' + phy_id + '/' + driver + '/'
    if ( os.path.isdir(debugfs_dir) ):
        return debugfs_dir
    else:
        raise ValueError('Could not find debugfs directory for interface %s.' % iface)
    return None


def scan_dev_start(iface):
    ''' enable ath9k built-in spectral analysis on this device '''
    debugfs_dir = get_debugfs_dir(iface)
    ctrl_fn = debugfs_dir + 'spectral_scan_ctl'
    cmd = 'trigger'
    file_write(ctrl_fn, cmd)
    return None


def scan_dev_stop(iface):
    ''' disable ath9k built-in spectral analysis on this device '''
    debugfs_dir = get_debugfs_dir(iface)
    ctrl_fn = debugfs_dir + 'spectral_scan_ctl'
    cmd = 'disable'
    file_write(ctrl_fn, cmd)
    return None


def scan_dev_drain(iface):
    ''' drain ath9k built-in spectral sample queue on this device '''
    debugfs_dir = get_debugfs_dir(iface)
    scan_fn = debugfs_dir + 'spectral_scan0'
    fd = open(scan_fn, 'rb')
    fd.read()
    fd.close()
    return None


def scan_dev_configure(iface='wlan0', mode='manual', count=8, period=255, fft_period=15, short_repeat=1):
    ''' configure ath9k built-in spectral analysis on this device '''
    debugfs_dir = get_debugfs_dir(iface)

    # spectral_scan_ctl (scan mode)
    ctrl_fn = debugfs_dir + 'spectral_scan_ctl'
    cmd = str(mode)
    file_write(ctrl_fn, cmd)

    # spectral_count
    ctrl_fn = debugfs_dir + 'spectral_count'
    cmd = str(count)
    file_write(ctrl_fn, cmd)

    # spectral_period
    ctrl_fn = debugfs_dir + 'spectral_period'
    cmd = str(period)
    file_write(ctrl_fn, cmd)

    # spectral_fft_period
    ctrl_fn = debugfs_dir + 'spectral_fft_period'
    cmd = str(fft_period)
    file_write(ctrl_fn, cmd)

    # spectral_short_repeat
    ctrl_fn = debugfs_dir + 'spectral_short_repeat'
    cmd = str(short_repeat)
    file_write(ctrl_fn, cmd)
