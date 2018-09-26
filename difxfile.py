#!/usr/bin/env python

import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt
import sys
import os

swin_hdr = np.dtype([   ('sync', 'i4'), \
                        ('ver',  'i4'), \
                        ('no_bl','i4'), \
                        ('mjd',  'i4'), \
                        ('sec',  'f8'), \
                        ('config_idx',   'i4'), \
                        ('src_idx',      'i4'), \
                        ('freq_idx',     'i4'), \
                        ('polar',        'a2'), \
                        ('pulsarbin',    'i4'), \
                        ('weight',       'f8'), \
                        ('uvw',  'f8', 3)])

class DiFXScan(object):
    
    def __init__(self):
        self.mjd    =   -1
        self.fullsec    =   -1
        self.dur    =   -1

class DiFX(object):

    def __init__(self):

        self.path   =   ''
        self.fmt    =   ''
        self.stn    =   []
        self.bl_dict=   {}
        self.nchan  =   -1
        self.freq_list  =   []
        self.nfreq  =   -1
        self.ap     =   -1.0
        self.bw     =   -1.0
        self.scan   =   {}

    def load_scan(self, scan_no):

        scan    =   self.scan[scan_no]
        swin_rec = np.dtype([   ('h', swin_hdr), \
                                ('vis', 'c8', self.nchan)])
        foldername  =   self.fmt % (scan_no)
        filename    =   "%s/%s/DIFX_%5d_%06d.s0000.b0000" \
                        % (self.path, foldername, scan.mjd, scan.sec)
        recs    =   np.fromfile(filename, dtype = swin_rec)
        return recs

    def rec2arr(self, recs, scan_no, bl_no, freq_list):
        
        scan    =   self.scan[scan_no]
        nfreq   =   len(freq_list) 
        fd  =   {}
        for i, fid in enumerate(freq_list):
            fd[fid]    =   i
        
        nap =   int(np.ceil(self.scan[scan_no].dur / self.ap))
        buf =   np.zeros((nap, nfreq, self.nchan), dtype = np.complex64)
        head=   np.zeros((nap, nfreq), dtype = swin_hdr)
            
        arr2rec   =   {}

        nap =   0
        for i, rec in enumerate(recs):

#            print rec['h']

            if rec['h']['no_bl']    !=  bl_no:
                continue
            fid =   rec['h']['freq_idx']
            if not fd.has_key(fid):
                continue
            apid    =   int(((rec['h']['mjd'] - scan.mjd) * 86400.0 \
                    + (rec['h']['sec'] - scan.sec)) / self.ap)
            buf[apid, fd[fid], :]   =   rec['vis'][:]
            head[apid, fd[fid]] =   rec['h']
            
            arr2rec[apid * nfreq + fid] =   i
            nap +=  1
            if nap % 10000 == 0:
                print '%d APs. have been loaded.' % (nap)
#            if nap == 1050000:
#                break

        return head, buf, arr2rec

