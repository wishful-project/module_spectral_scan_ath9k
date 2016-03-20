# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:19:19 2016

@author: olbrich
"""
import warnings
import numpy as np


def signbit_convert(data, maxbit):
    if (data & (1 << (maxbit - 1))) != 0:
        data -= (1 << maxbit)
    return data


def get_csi_matrix(buf, nr, nc, num_tones):

    # check input
    if (len(buf)*8/(2*10)) < (int(nr)*int(nc)*int(num_tones)):
        warnings.warn('Invalid CSI buffer length detected.', RuntimeWarning, stacklevel=2)
        return np.array([], dtype=complex)
    elif buf.dtype != np.dtype(np.uint8):
        warnings.warn('Invalid CSI buffer data type detected.', RuntimeWarning, stacklevel=2)
        return np.array([], dtype=complex)
    elif (nr == 0) or (nc == 0) or (num_tones == 0):
        warnings.warn('Invalid CSI matrix dimensions detected.', RuntimeWarning, stacklevel=2)
        return np.array([], dtype=complex)
    else:
        buf = buf.tolist()
        csi_matrix = np.full((nr, nc, num_tones), (np.nan + np.nan*1j), dtype=complex)

    # decode CSI matrix H from buffer
    # real and imag parts of H are stored with 10 bit resolution, i.e. e.g.:
    # 1710*8/(2*10) = 684 = 3*2*114
    # see: csi_fun.c
    BITS_PER_BYTE = 8
    BITS_PER_SYMBOL = 10
    bits_left = 16  # process 16 bits at a time
    bitmask = (1 << BITS_PER_SYMBOL) - 1
    idx = 0
    local_h = buf
    h_data = local_h[idx]
    idx += 1
    h_data += (local_h[idx] << BITS_PER_BYTE)
    current_data = h_data & ((1 << 16) - 1)  # get 16 LSBs first

    for nt_idx in range(0, num_tones):
        for nc_idx in range(0, nc):
            for nr_idx in range(0, nr):

                # get the next 16 bits
                if (bits_left - BITS_PER_SYMBOL) < 0:
                    idx += 1
                    h_data = local_h[idx]
                    idx += 1
                    # h_data += (local_h[idx] << BITS_PER_BYTE)
                    h_data += ((local_h[idx] if local_h[idx:] else 0) << BITS_PER_BYTE)
                    current_data += h_data << bits_left
                    bits_left += 16

                imag = current_data & bitmask
                imag = float(signbit_convert(imag, BITS_PER_SYMBOL))
                bits_left -= BITS_PER_SYMBOL
                current_data >>= BITS_PER_SYMBOL

                # get the next 16 bits
                if (bits_left - BITS_PER_SYMBOL) < 0:
                    idx += 1
                    h_data = local_h[idx]
                    idx += 1
                    # h_data += (local_h[idx] << BITS_PER_BYTE)
                    h_data += ((local_h[idx] if local_h[idx:] else 0) << BITS_PER_BYTE)
                    current_data += h_data << bits_left
                    bits_left += 16

                real = current_data & bitmask
                real = float(signbit_convert(real, BITS_PER_SYMBOL))
                bits_left -= BITS_PER_SYMBOL
                current_data >>= BITS_PER_SYMBOL

                csi_matrix[nr_idx, nc_idx, nt_idx] = (real + imag*1j)

    return csi_matrix
