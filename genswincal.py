#!/usr/bin/env python

import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
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

def buf_dump(dd, buf, fname):

    df  =   dd.bw / dd.nvis / 1E6
 
    fp  =   open(fname+'.txt', 'w')
    for i in range(len(dd.freq_list)):
        freq    =   dd.freq_list[i] / 1E6
        id_mb   =   dd.ids_mb[i]
        arr     =   buf[id_mb, 0:dd.nvis]
        deg =   np.angle(arr)
        amp =   np.absolute(arr)
        x   =   freq + np.arange(dd.nvis) * df
        for k in range(1, dd.nvis):
            fp.write('%.5f\t%.3e\t%.3e\n' % (x[k], amp[k], deg[k]))
    fp.close()



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

def fit_sbd(dd, s):

    nsb     =   1
    while nsb < dd.nvis:
        nsb <<= 1
    nsb <<= 4
    sb_res  =   1. / (dd.bw / dd.nvis) / nsb
    sb_range    =   sb_res * np.arange(-nsb / 2, nsb / 2, dtype = float)

    print 'sb range: %.3e ~ %.3e' % (sb_range[0], sb_range[-1])

    sbd =   [] 
    for fid in range(dd.nfreq):
        id_mb   =   dd.ids_mb[fid]
        r       =   np.fft.fft(s[id_mb, 0:dd.nvis], n = nsb)
        r       =   np.fft.fftshift(r)
        m       =   np.absolute(r)
        isb     =   np.argmax(m)
        sbd.append(sb_range[isb])

    return sbd
        
def rot_sbd_inplace(dd, s, sbd, shift = False):
    
    res =   dd.bw / dd.nvis
    for fid in range(dd.nfreq):
        id_mb   =   dd.ids_mb[fid]
        dph0    =   0.
        if shift:
            dph0    =   -2. * np.pi * dd.bw * 0.5 * sbd[fid]
        for ivis in range(dd.nvis):
            dph             =   dph0 + 2. * np.pi * res * ivis * sbd[fid]
            s[id_mb, ivis]  *=  np.exp(-1j * dph)

def rot_mb_inplace(dd, s, mb, sb):
    
    spec_res    =   dd.bw / dd.nvis
    for fid in range(dd.nfreq):
        id_mb   =   dd.ids_mb[fid]
        dph_mb  =   (dd.freq_list[fid] - dd.freq_ref) * mb * 2. * np.pi
        for ivis in range(dd.nvis):
            dph =   dph_mb + 2. * np.pi * spec_res * ivis * sb
            s[id_mb, ivis]  *=  np.exp(-1j * dph)

def plot_mb_sb_3D(m, mb_range, sb_range):
    
    matplotlib.rc('font', size = 12)
 
    fig     =   plt.figure()
    ax      =   fig.gca(projection = '3d')

#    print 'shape of mb_range: ', np.shape(mb_range)
#    print 'shape of sb_range: ', np.shape(sb_range)
    
    X, Y    =   np.meshgrid(sb_range * 1E6, mb_range * 1E9)
#    print 'shape of X: ', np.shape(X)
#    print 'shape of Y: ', np.shape(Y)
#    print 'shape of m: ', np.shape(m)

    surf    =   ax.plot_surface(X, Y, m)
#    fig.colorbar(surf, shrink = 0.5, aspect = 5)
#    ax.tick_params(axis = 'both', which = 'major', pad = 15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base = 0.5))
    plt.xlabel('SBD [us]')
    plt.ylabel('MBD [ns]')
#    plt.show()
    plt.savefig('mbd_sbd_3d.png')


def plot_mb_sb(m, mb_range, sb_range):
    
    fig     =   plt.figure()
    ax      =   fig.gca(projection = '3d')

#    print 'shape of mb_range: ', np.shape(mb_range)
#    print 'shape of sb_range: ', np.shape(sb_range)
    
    X, Y    =   np.meshgrid(sb_range * 1E9, mb_range * 1E9)
#    print 'shape of X: ', np.shape(X)
#    print 'shape of Y: ', np.shape(Y)
#    print 'shape of m: ', np.shape(m)

    surf    =   ax.plot_surface(X, Y, m)
#    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.xlabel('sb [ns]')
    plt.ylabel('mb [ns]')
    plt.show()

def plot_mb(dd, buf, fname):

    plt.clf()
    fig =   plt.figure()
    ax  =   fig.add_subplot(111)
    nmb, nsb    =   np.shape(buf)
    
    fig =   plt.figure()
    arr =   np.sum(buf, axis = 1)
    arr =   arr[dd.ids_mb]
    deg =   np.angle(arr)
    amp =   np.absolute(arr)
    amp =   10 * np.log10(amp)

    x   =   dd.freq_list / 1E6

    plt.plot(x, deg, 'rs', ms = 5, mew = 0) 
#    plt.xlim(2200., 2350.)
    plt.xlim(2190, 2280)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('S Band [MHz]')
    plt.ylabel('Phase [Rad]', color = 'r')
    plt.ylim(-np.pi, np.pi)
#    plt.show()

    ax.twinx()
    plt.plot(x, amp, 'b-', lw = 1)
    plt.xlim(2190, 2280)
    plt.ylabel('Amplitude [dB]', color = 'b')

    plt.savefig('multiband_%s.png' % (fname))

def plot_sb_all_chan(dd, buf, fname):

    fmin    =   np.min(dd.freq_list) / 1E6
    fmax    =   (np.max(dd.freq_list) + dd.bw) / 1E6

    df  =   dd.bw / dd.nvis / 1E6
    
    plt.clf()

    fig =   plt.figure()
    ax  =   fig.add_subplot(111)
 
    matplotlib.rc('font', size = 15)
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.12)

    npt =   32

    for i in range(len(dd.freq_list)):
        freq    =   dd.freq_list[i] / 1E6
        id_mb   =   dd.ids_mb[i]
        arr     =   buf[id_mb, 0:dd.nvis]
        arr     =   np.sum(arr.reshape(npt, -1), axis = 1)
        deg =   np.angle(arr)

        x   =   freq + np.arange(dd.nvis) * df
        x   =   np.sum(x.reshape(npt, -1), axis = 1) / (dd.nvis / npt)
 
        plt.plot(x, deg, 'rs', ms = 5, mew = 0) 

    plt.xlim(fmin, fmax)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('S band [MHz]')
    plt.ylabel('Phase [Rad]', color = 'r')

    ax.twinx()
    for i in range(len(dd.freq_list)):
        freq    =   dd.freq_list[i] / 1E6
        id_mb   =   dd.ids_mb[i]
        arr     =   buf[id_mb, 0:dd.nvis]
        arr     =   np.sum(arr.reshape(npt, -1), axis = 1)
        amp =   np.absolute(arr)
        amp =   10 * np.log10(amp)

        x   =   freq + np.arange(dd.nvis) * df
        x   =   np.sum(x.reshape(npt, -1), axis = 1) / (dd.nvis / npt)
 
        plt.plot(x, amp, 'b-', ms = 5, mew = 0) 

    plt.xlim(fmin, fmax)
#    plt.ylim(-np.pi, np.pi)
#    plt.xlabel('S band [MHz]')
    plt.ylabel('Amplitude [dB]', color = 'b')

    plt.savefig('sb_all_%s.png' % (fname))

def plot_sb(dd, buf, fname):

    plt.clf()
    fig =   plt.figure()

    matplotlib.rc('font', size = 15)
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.12)

    arr =   np.sum(buf, axis = 0)
    deg =   np.angle(arr)
    df  =   dd.bw / dd.nvis / 1E6
    x   =   np.arange(dd.nvis) * df
    plt.plot(x, deg[0:dd.nvis], 'rs', ms = 5, mew = 0) 
    plt.xlim(0, dd.bw / 1E6)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('Bandwidth [MHz]')
    plt.ylabel('Phase [Rad]')
    plt.savefig('singleband_%s.png' % (fname))

def argmax2d(m):
    s   =   np.shape(m)[1]
    i   =   np.argmax(m)
    return (i / s, i % s)

# 0: ap, 1: freq, 2: vis
def fit_multiband(dd, buf):
    if dd.nap == 0:
        return
#    print 'buf shape: ', np.shape(buf)
    assert(np.shape(buf) == (dd.nap, dd.nfreq, dd.nvis))
# 0: freq, 1: ap, 2: vis
    buf =   np.moveaxis(buf, [0, 1, 2], [1, 0, 2])

#    dd.freq_ref =   2201.75E6
    
    freq_list   =   dd.freq_list - dd.freq_ref
    
#    df_mb   =   10E6
#    dd.df_mb=   df_mb
    df_mb   =   dd.df_mb
    ids_mb  =   (freq_list / df_mb + 0.5).astype(int)
    dd.ids_mb   =   ids_mb
    id_max  =   np.max(ids_mb)
    nmb     =   1
    while nmb < id_max:
        nmb <<= 1
    nmb <<= 2
#    nmb =   256
    print 'mb: id_max: %d, final nmb: %d' % (id_max, nmb)

    nsb     =   1
    while nsb < dd.nvis:
        nsb <<= 1
    nsb <<= 4
    print 'sb: nvis: %d, final nsb: %d' % (dd.nvis, nsb)

#    print 'shape of buf: ', np.shape(buf)
    
    mb_res  =   1. / df_mb / nmb
    sb_res  =   1. / (dd.bw / dd.nvis) / nsb
    mb_range    =   mb_res * np.arange(-nmb / 2, nmb / 2, dtype = float)
    sb_range    =   sb_res * np.arange(-nsb / 2, nsb / 2, dtype = float)

#    print 'mb_range: %e ~ %e ns' % (mb_range[0] * 1E9, mb_range[-1] * 1E9)
#    print 'sb_range: %e ~ %e ns' % (sb_range[0] * 1E9, sb_range[-1] * 1E9)
    
#    buf     =   buf[:, :, 0:dd.nvis]
    buf     =   buf[:, 7:16, 0:dd.nvis]
    buf     =   np.sum(buf, axis = 1)
#    buf     /=  np.absolute(buf)

    s =   np.zeros((nmb, nsb), dtype = np.complex64)

#    print 'shape of pcal: ', np.shape(dd.pcal)
    for fid in range(dd.nfreq):
        id_mb   =   ids_mb[fid]
#        s[id_mb, 0:dd.nvis]   =   buf[fid, 0:dd.nvis] * np.exp(-1j * dd.pcal[fid])
        s[id_mb, 1:dd.nvis]   =   buf[fid, 1:dd.nvis]


#    plot_mb(dd, s, 'original')
#    plot_sb(dd, s, 'original')
    plot_sb_all_chan(dd, s, 'original')
    buf_dump(dd, s, 'original')

    for fid in range(dd.nfreq):
        id_mb   =   ids_mb[fid]
        s[id_mb, 0:dd.nvis]   =   s[id_mb, 0:dd.nvis] * np.exp(-1j * dd.pcal[fid])
#        s[id_mb, 0:dd.nvis]   =   buf[fid, 0:dd.nvis]

    s0      =   s.copy()
    sum0    =   np.sum(s)

    r   =   np.fft.fft2(s)
    r   =   np.fft.fftshift(r)

    m   =   np.absolute(r)

    imb, isb = argmax2d(m)
    print 'max value (pcal only): %.3f, mb: %.3fns, sb: %.3fns' % \
        (m[imb, isb], mb_range[imb] * 1E9, sb_range[isb] * 1E9)

    mbd =   mb_range[imb]
    sbd =   sb_range[isb]

    rot_mb_inplace(dd, s, mbd, sbd)

    pcal_arr    =   np.sum(s, axis = 1)
    pcal_arr    =   pcal_arr[dd.ids_mb]
#    pcal_arr    *=  np.exp(1j * 2. * np.pi * mbd * (dd.freq_list - dd.freq_ref))
    deg         =   np.angle(pcal_arr)
    print 'pcal: ', deg.tolist()
#    print 'return after pcal output'
#    return

    sum =   np.sum(s)
#    print 'enhanced: ', np.absolute(sum) / np.absolute(sum0)

#    plot_mb(dd, s, 'fft2')
#    plot_sb(dd, s, 'fft2')
#    plot_sb_all_chan(dd, s, 'pcal')
#    buf_dump(dd, s, 'pcal')

    for fid in range(dd.nfreq):
        id_mb   =   ids_mb[fid]
        s[id_mb, 0:dd.nvis]   =   s[id_mb, 0:dd.nvis] * np.exp(-1j * deg[fid])
    plot_sb_all_chan(dd, s, 'pcal')
    buf_dump(dd, s, 'pcal')

######################### pcal and sbd ############################
#    sbd =   dd.sbd
    sbd     =   fit_sbd(dd, s) 
    print sbd
    rot_sbd_inplace(dd, s, sbd, shift = True)

    r   =   np.fft.fft2(s)
    r   =   np.fft.fftshift(r)

    imb, isb = argmax2d(m)

    m   =   np.absolute(r)

    imb, isb = argmax2d(m)
    print 'max value (pcal & sbd): %.3f, mb: %.3fns, sb: %.3fns' % \
        (m[imb, isb], mb_range[imb] * 1E9, sb_range[isb] * 1E9)

    mbd =   mb_range[imb]
    sbd =   sb_range[isb]

    rot_mb_inplace(dd, s, mbd, sbd)

#    print 'enhanced: ', np.absolute(sum) / np.absolute(sum0)
    
#    plot_mb(dd, s, 'sbd')
#    plot_sb(dd, s, 'sbd')
    plot_sb_all_chan(dd, s, 'chansb')
    buf_dump(dd, s, 'chansb')

    plot_mb_sb_3D(m, mb_range, sb_range)

    return r

def main_multiband(blid):

    difx    =   difxfile.DiFX()
    difx.path   =   '/home/liulei/Pulsar/psrf02_1.024ms'
    difx.fmt    =   'B0329_%03d.difx'
    difx.stn    =   ['Km', 'Sh', 'Ur']
    difx.bl_dict[1] =   1 * 256 + 2     # Sh-Km
    difx.bl_dict[2] =   2 * 256 + 3     # Sh-Ur
    difx.bl_dict[4] =   1 * 256 + 3     # Km-Ur
    difx.nchan      =   32
    difx.freq_list  =   np.arange(10, 16)
    difx.nfreq      =   len(difx.freq_list)
    difx.ap     =   1.28
    difx.bw     =   16E6

    scan        =   difxfile.DiFXScan()
    scan.mjd    =   57068
    scan.sec    =   71740.0
    scan.dur    =   360.0
    difx.scan[293]   =   scan

    bl_no   =   difx.bl_dict[blid]

    dd      =   DataDescriptor()

    pcal_dict       =   {}
    sbd_dict    =   {}

# sbd, pcal taken from 3C273/calcsbmb.py

    pcal_dict[1]    =   [0.0] * 6
    pcal_dict[2]    =   [0.0] * 6
    pcal_dict[4]    =   [0.0] * 6

    sbd_dict[1]     =   [0.0] * 6
    sbd_dict[2]     =   [0.0] * 6
    sbd_dict[4]     =   [0.0] * 6

#    pcal_dict[1]     =   [-2.009403944015503, 2.8115944862365723, -0.918035089969635, -2.3163623809814453, -2.25700044631958, -2.443556785583496]
#    pcal_dict[4]     =   [1.6856310367584229, 2.506923198699951, 2.2702338695526123, -3.1027116775512695, 2.209333658218384, 1.653191328048706]
#    pcal_dict[2]     =   [-1.9500532150268555, -2.3401710987091064, -1.5803169012069702, -1.9621654748916626, 0.5817174315452576, -2.4779889583587646]

#    sbd_dict[1]     =   [0.0, -3.9062499999999998e-09, 2.734375e-08, 0.0, -3.9062499999999998e-09, -3.9062499999999998e-09]
#    sbd_dict[4]     =   [-7.8124999999999996e-09, -3.9062499999999998e-09, 0.0, 3.9062499999999998e-09, 0.0, -3.9062499999999998e-09]
#    sbd_dict[2]     =   [0.0, 7.8124999999999996e-09, -1.9531249999999998e-08, 1.171875e-08, 1.171875e-08, 7.8124999999999996e-09]

    global  scan_no

#    data_path   =   '.'
#    job_name    =   data_path + '/' + 'psrf02_cvn1.2.job'
#    chan_list=   np.arange(0, 6)
#    blid    =   1   # Sh-Km
#    blid    =   2   # Sh-Ur
#    blid    =   4   # Km-Ur
#    scan_no_list    =   [69, 71, 73]
    dd.freq_ref =   2192.0E6
    dd.df_mb    =   16.0E6
    
    dd.pcal     =   np.zeros(6)
    dd.pcal     =   pcal_dict[blid]
    dd.sbd      =   sbd_dict[blid]

#    job         =   cvnply.readCJob(job_name)
#    dd.ap_time  =   job.custom[0].time_avg
#    dd.nvis     =   job.custom[0].fftsize / 2
#    dd.bw       =   job.fm[0].bandwidth * 1E6
#    dd.tstart   =   10.0 # J0332+5434: 10 - 170
    dd.ap_time  =   difx.ap # in s
    dd.nvis     =   difx.nchan
    dd.bw       =   difx.bw
    dd.tstart   =   0.0

#    for fid in chan_list:
#        dd.freq_list.append(job.fm[0].freq[fid].freq * 1E6)

    dd.chan_list    =   np.arange(10, 16)
    dd.freq_list    =   np.array([2192.0, 2208.0, 2224.0, 2240.0, 2256.0, 2272.0]) * 1E6
    dd.sb_list      =   ['U'] * 6

    dd.nfreq    =   len(dd.chan_list)

    dd.freq_max =   2288.0E6
    dd.dm       =   26.833030


#    nsum_list   =   [4, 8, 12, 16, 20, 24, 28]
#    nsum_list   =   [4, 8, 16, 24, 32]
    nsum_list   =   np.array([4, 8, 16, 24, 32])

#    nstn    =   job.num_station
#    nbl     =   (nstn * (nstn + 1)) / 2
#    nchan   =   len(job.fm[0].freq)

#    for scan_no in scan_no_list:
    if True:
        scan    =   difx.scan[scan_no]
        recs    =   difx.load_scan(scan_no)
        buf, arr2rec = difx.rec2arr(recs, scan_no, bl_no, dd.chan_list)
        dur     =   scan.dur 
#        dat_name=   '%s/%s_%s.dat' % (data_path, scan.start, scan.src_name)
#        dur     =   job.diff_time(scan.start, scan.stop)
#        buf     =   np.fromfile(dat_name, dtype = np.complex64)
#        buf     =   buf.reshape((-1, nchan, nbl, dd.nvis))
        dd.nap  =   len(buf)
#        buf     =   buf[:, chan_list, blid, :]
        assert(dd.nap != 0)
#        dd.time =   job.diff_time(job.start, scan.start)
        dd.dur  =   dur
        print 'Scan %d, time %.1f, nap: %d' % (scan_no, dd.time, dd.nap)
        fit_multiband(dd, buf) 

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'genfitdump.py scan_no blid'
        sys.exit(0)

    scan_no =   int(sys.argv[1])
    blid    =   int(sys.argv[2])

#    global scan_no

    main_multiband(blid)
