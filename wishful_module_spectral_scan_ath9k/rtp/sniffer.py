# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:30:15 2016

@author: olbrich
"""
import numpy as np
from scapy.all import RadioTap, Dot11
from .decoder import get_radiotap_info
from .constants import IEEE80211_FLOAT_HTRATES, MAX_MCS_INDEX, IEEE80211_FRAME_TYPES, IEEE80211_FRAME_SUBTYPES


def get_airtime(myaddr, ret_q, debug):
    def __get_airtime(fm):

        # check if we got a radiotap frame
        if fm.haslayer(RadioTap):
            if debug:
                print("Got Radiotap header.")
                print("--------------------------------------------------------------------------------")
                np.set_printoptions(formatter={'float': '{: 0.1f}'.format, 'bool': '{: 0.0f}'.format}, linewidth=160)

            # get raw data of the whole frame
            fm_raw = bytes(fm)

            # decode radiotap information
            offset = 0
            rtp_info = get_radiotap_info(fm_raw, offset, debug)

            if debug:
                print("--------------------------------------------------------------------------------\n")

            # calculate the 802.11 radio frame length and datarate
            radio_frame_len = int(len(fm_raw) - rtp_info['hdr']['len']) * 8 # bits
            rtp_ix = 0 # only use the base header, not the extension headers
            haveRate = rtp_info['pmask'][rtp_ix][2] # legacy field, only valid for non-HT transmission
            haveMCS = rtp_info['pmask'][rtp_ix][19]

            if haveRate:
                field_ix = rtp_info['dat'][rtp_ix]['fnames'].index('Rate')
                datarate = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['tx/rx rate [Mbps]'][0] * 1e6 # bits/sec
            elif haveMCS:
                # for datarate calculation we need:
                # 1) MCS index
                # 2) guard interval length (0: long GI, 1: short GI)
                # 3) channel bandwidth (0: 20, 1: 40, 2: 20L, 3: 20U )
                field_ix = rtp_info['dat'][rtp_ix]['fnames'].index('MCS')
                mcs_known = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['MCS known'][0]
                bw_known = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['Bandwidth known'][0]
                gi_known = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['Guard Interval known'][0]

                # If we have the MCS index, channel width, and
                # guard interval length, and the MCS index is
                # valid, we can compute the rate.  If the resulting
                # rate is non-zero, report it.  (If it's zero,
                # it's an MCS/channel width/GI combination that
                # 802.11n doesn't support.)
                can_calculate_rate = True
                mcs_index = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['MCS index'][0]
                bw_index = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['Bandwidth'][0]
                gi_index = rtp_info['dat'][rtp_ix]['fdata'][field_ix]['Guard Interval'][0]

                if not (mcs_known & mcs_index):
                    can_calculate_rate = False
                if not (mcs_known & bw_index & bw_known):
                    can_calculate_rate = False
                if not (mcs_known & gi_index & gi_known):
                    can_calculate_rate = False

                if (can_calculate_rate & mcs_index <= MAX_MCS_INDEX):
                    datarate = IEEE80211_FLOAT_HTRATES[mcs_index][bw_index][gi_index] * 1e6 # bits/sec
                else:
                    datarate = 0.0

            # calculate the estimated airtime
            # TODO: what about Frame Control Field duration flag?
            airtime = (radio_frame_len / datarate) * 1e6 # usec

            if debug:
                print('Radio frame datarate: %.2f Mbps -- airtime: %.2f usec' % (datarate/1e6, airtime) )

        # check if we got a 802.11 radio frame
        if fm.haslayer(Dot11):

            if debug:
                print("Got Dot11 header.")
                print("--------------------------------------------------------------------------------")

                # decode 802.11 frame type
                frame_type_str = IEEE80211_FRAME_TYPES[fm.type]
                frame_subtype_str = IEEE80211_FRAME_SUBTYPES[fm.type][fm.subtype]

#            # scapy Frame Control Field
#            DS = fm.FCfield & 0x3
#            toDS = DS & 0x1 != 0
#            fromDS = DS & 0x2 != 0

            # MAC addresses (non-WDS, i.e- To-DS=1, From-DS=0)
            rx_addr = fm.addr1
            tx_addr = fm.addr2

#            if (toDS == False) and (fromDS == False):
#                dst_addr = fm.addr1
#                src_addr = fm.addr2
#                bss_addr = fm.addr3
#            elif (toDS == False) and (fromDS == True):
#                dst_addr = fm.addr1
#                src_addr = fm.addr3
#                bss_addr = fm.addr2
#            elif (toDS == True) and (fromDS == False):
#                dst_addr = fm.addr3
#                src_addr = fm.addr2
#                bss_addr = fm.addr1
#            else:
#                dst_addr = None
#                src_addr = None
#                bss_addr = None

            if (myaddr in [rx_addr, tx_addr]) or (rx_addr == 'ff:ff:ff:ff:ff:ff'):
                #print("Radio frame from or to me -- RX: %s -- TX: %s" %(rx_addr, tx_addr))
                ret_q.put([0, airtime])
            else:
                #print("Radio frame not for me -- RX: %s -- TX: %s" %(rx_addr, tx_addr))
                ret_q.put([1, airtime])

            if debug:
                #print("Dot11 %s/%s frame--> From: %s -- To: %s -- BSSID: %s" % (frame_type_str, frame_subtype_str, src_addr, dst_addr, bss_addr))
                print("Dot11 %s/%s frame--> RX addr: %s -- TX addr: %s" % (frame_type_str, frame_subtype_str, rx_addr, tx_addr))
                print("--------------------------------------------------------------------------------\n")

        return None
    return __get_airtime
