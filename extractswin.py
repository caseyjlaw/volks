#!/usr/bin/env python

import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import difxfile
#import cvnply

scan_no     =   -1
ph0_dict    =   {1: 0, 69: 0.581375, 71: 0.414097, 73: 0.246818}

def calc_psr_phase(t):

    c1  =   -6.89262836039208383e-03
    F0  =   1.399538059693
#    ph0 =   0.581375 # scan 0069
#    ph0 =   0.414097 # scan 0071
#    ph0 =   0.246818 # scan 0073
    ph0 =   ph0_dict[scan_no]
    return (t - 10.000512) * (F0 + c1 / 60.) + ph0

# f1_in, f2_in in Hz, time diff in s
def calc_tshift(f1_in, f2_in, dm):
    
    f1  =   f1_in / 1E6
    f2  =   f2_in / 1E6

    return 4.15E6 * dm * (1. / (f1 * f1) - 1. / (f2 * f2)) * 1E-3

class DataDescriptor():

    def __init__(self):
        self.time   =   0.0
        self.tstart =   0.0
        self.nap    =   -1
        self.nfreq  =   -1
        self.nvis   =   -1
        self.ap_time=   -1.0
        self.bw     =   -1.0
        self.freq_list  =   []
        self.sb_list    =   []
        self.pcal       =   []
        self.df_mb      =   0.0
        self.ids_mb     =   []
        self.freq_ref   =   0.0
        self.dm         =   0.0


def dedisp(dd, buf):

    tb  =   np.zeros((dd.nfreq, dd.nvis), dtype = int)
    
    file   =   open('dm.shift', 'w')
    df  =   dd.bw / dd.nvis
    for fid in range(dd.nfreq):
        sign    =   1.0
        if dd.sb_list[fid]  ==  'L':
            sign    =   -1.0
        for vid in range(dd.nvis):
            f       =   dd.freq_list[fid] + sign * df * vid
            tshift  =   calc_tshift(f, dd.freq_max, dd.dm)
            assert(tshift >= 0.0)
            ishift  =   (tshift / dd.ap_time + 0.5)
            tb[fid, vid] =   ishift
            file.write('%d\t%d\t%d\n' % (fid, vid, ishift))
#            print 'fid %d, vid %d, shift %d' % (fid, vid, ishift)
    next    =   np.max(tb)
    file.close()
#    print 'exit after dm.shift dump'
#    sys.exit(0)
    
    buf1    =   np.zeros((dd.nap, dd.nfreq, dd.nvis), dtype = np.complex64)
    
    for fid in range(dd.nfreq):
        for vid in range(dd.nvis):
            k   =   tb[fid, vid]
            buf1[0:dd.nap - k, fid, vid]    =   buf[k:dd.nap, fid, vid]

    return buf1
   
# 0: ap, 1: freq, 2: vis
def extract_swin(dd, head, buf, id0, id1):
    if dd.nap == 0:
        return
    assert(np.shape(buf) == (dd.nap, dd.nfreq, dd.nvis))

    s   =   ''
    
    for apid in range(id0, id1):
        for fid in range(dd.nfreq):
            s   +=  head[apid, fid].tostring()
            s   +=  buf[apid, fid, :].tostring()

    return s

def reorder(s, nvis):

    dtype_rec =   np.dtype([  ('h', difxfile.swin_hdr), \
                        ('vis', 'c8', nvis)])
    recs    =   np.fromstring(s, dtype = dtype_rec)  
    nrec    =   len(recs)
    print 'total recs: ', nrec

    mjds    =   []
    secs    =   []
    freqs   =   []
    bls     =   []
    for rec in recs:
        mjds.append(rec['h']['mjd'])
        secs.append(rec['h']['sec'])
        bls.append(rec['h']['no_bl'])
        freqs.append(rec['h']['freq_idx'])

    mjd0    =   np.min(mjds)
    sec0    =   np.min(secs)

    fullsecs    =   (np.array(mjds) - mjd0) * 86400. + (np.array(secs) - sec0)

#    ids =   np.argsort(fullsecs)
    ids =   np.lexsort((bls, freqs, fullsecs))
#    ids =   np.lexsort((freqs, bls, fullsecs))

#    print ids
#    print np.array(fullsecs)[ids]

    recs    =   recs[ids]

    for i in range(nrec):
        print '%.8f\t\t%d\t\t%d' % (recs[i]['h']['sec'], \
                    recs[i]['h']['freq_idx'], recs[i]['h']['no_bl'])

    return recs.tostring()

def main():

    if len(sys.argv) < 2:
        print 'genfitdump.py scan_no sp_file'
        sys.exit(0)

    difx    =   difxfile.DiFX()
    difx.path   =   '/home/liulei/Pulsar/psrf02_1.024ms'
#    difx.path   =   '/home/liulei'
    difx.fmt    =   'B0329_%03d.difx'
    difx.stn    =   ['Km', 'Sh', 'Ur']
    difx.bl_dict[1] =   1 * 256 + 2     # Sh-Km
    difx.bl_dict[2] =   2 * 256 + 3     # Sh-Ur
    difx.bl_dict[4] =   1 * 256 + 3     # Km-Ur
    difx.nchan      =   32
    difx.freq_list  =   np.arange(10, 16)
    difx.nfreq      =   len(difx.freq_list)
    difx.ap     =   1.024E-3
    difx.bw     =   16E6

    scan        =   difxfile.DiFXScan()
    scan.mjd    =   57068
    scan.sec    =   38871.0
    scan.dur    =   180.0
    difx.scan[69]   =   scan

    scan        =   difxfile.DiFXScan()
    scan.mjd    =   57068
    scan.sec    =   39161.0
    scan.dur    =   180.0
    difx.scan[71]   =   scan

    scan        =   difxfile.DiFXScan()
    scan.mjd    =   57068
    scan.sec    =   39451.0
    scan.dur    =   180.0
    difx.scan[73]   =   scan

    dd      =   DataDescriptor()

    global  scan_no

    dd.freq_ref =   2192.0E6
    dd.df_mb    =   16.0E6
    
#    dd.pcal     =   np.zeros(6)
#    dd.pcal     =   pcal_dict[blid]
#    dd.sbd      =   sbd_dict[blid]

    dd.ap_time  =   difx.ap # in s

    dd.nvis     =   difx.nchan
    dd.bw       =   difx.bw
    dd.tstart   =   0.0

    dd.chan_list    =   np.arange(10, 16)
    dd.freq_list    =   np.array([2192.0, 2208.0, 2224.0, 2240.0, 2256.0, 2272.0]) * 1E6
    dd.sb_list      =   ['U'] * 6

    dd.nfreq    =   len(dd.chan_list)

    dd.freq_max =   2288.0E6
    dd.dm       =   26.833030

    scan_no =   int(sys.argv[1])
    sp_file =   sys.argv[2]

    scan    =   difx.scan[scan_no]
    dd.nap  =   int(np.ceil(scan.dur / dd.ap_time))

    arr =   np.loadtxt(sp_file, comments = '&')
    print arr

    sps  =   arr[:, 0:3].astype(int)
    
    s   =   ''
    for blid in [1, 2, 4]:

#        break

        print 'bl %d...' % (blid)
        bl_no   =   difx.bl_dict[blid]
        recs    =   difx.load_scan(scan_no)
        head, buf, arr2rec = difx.rec2arr(recs, scan_no, bl_no, dd.chan_list)
        buf =   dedisp(dd, buf)

        for sp in sps:
            if sp[0] == blid:
                s   +=  extract_swin(dd, head, buf, sp[1], sp[1] + sp[2])

    filename    =   'DIFX_%5d_%06d.s0000.b0000' % (scan.mjd, scan.sec)

#    s   =   np.fromfile(filename, dtype = 'i1').tostring()

    s   =   reorder(s, dd.nvis)

    f   =   open(filename, 'wb')
    f.write(s)
    f.close()

if __name__ == '__main__':
    main()

