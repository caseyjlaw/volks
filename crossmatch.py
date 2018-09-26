#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import sys

count_min   =   2

dtype_cand  =   np.dtype([  ('t',       'f8'), \
                            ('hr_min',  'f8'), \
                            ('hr_max',  'f8'), \
                            ('mbdavg',  'f8'), \
                            ('mbdstd',  'f8'), \
                            ('pavg',    'f8'), \
                            ('pstd',    'f8'), \
                            ('psrph',   'f8'), \
                            ('nsum',    'i4')])

scan_no     =   -1
ph0_dict    =   {69: 0.581375, 71: 0.414097, 73: 0.246818}

def calc_psr_phase(t):

    c1  =   -6.89262836039208383e-03
    F0  =   1.399538059693
#    ph0 =   0.581375 # scan 69
#    ph0 =   0.414097 # scan 71
#    ph0 =   0.246818 # scan 73
    ph0 =   ph0_dict[scan_no]
    return (t - 10.000512) * (F0 + c1 / 60.) + ph0

def is_overlap(cand, cm):
    if np.abs(cand.t - cm.t) <= cand.hr + cm.hr:
        return True
    return False

def is_psr(t, hr):

    F0  =   1.399538059693
    dph =   F0 * hr
    
    ph  =   calc_psr_phase(t)
    ph  -=  np.floor(ph)
    if np.abs(ph - 0.978) <= 0.005 + dph:
        return True
    return False

class Candidate(object):
    
    def __init__(self, a, blid):
        self.t  =   a['t']
        self.hr =   a['hr_min']
#        self.hr =   a['hr_max']
        self.mbd =   a['mbdavg']
        self.p   =   a['pavg']
        self.pstd   =   a['pstd']
        self.psrph  =   a['psrph']
        self.nsum   =   a['nsum']
        self.blid   =   blid
        
class CrossMatch(object):

    def __init__(self, cand):
        
        self.t      =   cand.t
        self.count  =   1
        self.hr     =   cand.hr
        self.d      =   {}
        self.d[cand.blid]=   cand
        self.psrph  =   [cand.psrph]
        self.bls    =   [cand.blid]

    def insert_cand(self, cand):
        
        if self.d.has_key(cand.blid):
            print 'Warning! bl %d, t %.6f already inserted (t %.6f)!' % \
                    (cand.blid, cand.t, self.t)
            return
        
        self.count  +=  1
        t_min   =   np.min([self.t - self.hr, cand.t - cand.hr])
        t_max   =   np.max([self.t + self.hr, cand.t + cand.hr])
        self.t  =   (t_min + t_max) * 0.5
        self.hr =   (t_max - t_min) * 0.5
        self.d[cand.blid]   =   cand
        self.psrph.append(cand.psrph)
        self.bls.append(cand.blid)

def load_cand_file(blid):

    prefix  =   '.'
    fname   =   '%s/bl%03d.nsum' % (prefix, blid)
    print 'load %s...' % (fname)
    arr     =   np.loadtxt(fname, dtype = dtype_cand) 
    cands   =   []
    for a in arr:
        cands.append(Candidate(a, blid)) 
    return cands

def match_cand(cands, cms):
    
    npsr    =   0
    for cand in cands:
#        if is_psr(cand.t, cand.hr):
#            npsr    +=  1
        inserted    =   False
        for cm in cms:
            if is_overlap(cand, cm):
                cm.insert_cand(cand) 
                inserted    =   True
                break
        if not inserted:
            cms.append(CrossMatch(cand))
#    print 'npsr from match_cand(): %d' % (npsr)

def output_cms(cms):

    blname  =   {1: 'Sh-Km', 2: 'Sh-Ur', 4: 'Km-Ur'}
    
    nbl4        =   0
    nbl4_psr    =   0
    ncm     =   np.zeros(3, dtype = float)
    npsr    =   np.zeros(3, dtype = float)
    for cm in cms:
        if 4 in cm.bls:
            nbl4    +=  1
            if is_psr(cm.t, cm.hr):
                nbl4_psr    +=  1
        if cm.count > 1:
            ph  =   calc_psr_phase(cm.t)
            print 'count %d, t: %.6f, range: %.6f, phase %.6f' % \
                (cm.count, cm.t, cm.hr, ph)
            for blid, cand in cm.d.items():
                print 'blid: %d, SNR: %.3f, mbd: %.3f' % \
                    (blid, cand.p, cand.mbd)
            print ''
            ncm[cm.count - 1]     +=  1
            if is_psr(cm.t, cm.hr):
                npsr[cm.count - 1]    +=  1

#        if cm.count == 3:
        if cm.count >= count_min:
            f   =   open('scan%04d_%.6f.sp' % (scan_no, cm.t), 'w')
#            f.write('#blid\tapid\twidth\ttime\twidth\tmbd\tsnr\n')
            for blid, cand in cm.d.items():
                i   =   int((cand.t - cand.hr - 0.0) / 1.024E-3 + 0.1)
                w   =   int(cand.hr / 1.024E-3 + 0.1) * 2
                f.write('%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.3f\t%.3f\n' \
                        % (blid, i, w, cand.t, calc_psr_phase(cand.t), \
                            cand.hr * 2, cand.mbd, cand.p))
                f.write('& %s & %.6f & %.3f & %.3f & %.3f\n' \
                        % (blname[blid], cand.t, calc_psr_phase(cand.t), \
                            cand.hr * 2 * 1000., cand.p))

            f.close()
#    print 'ncm: %d, npsr: %d' % (ncm, npsr)
    print 'ncm: ', ncm
    print 'npsr: ', npsr
    print 'nbl4: %d, nbl4_psr: %d' % (nbl4, nbl4_psr)

def gen_patch(cm):

    wt  =   2
    ph  =   -1
    pl  =   1E5
    for blid, cand in cm.d.items():
        if ph < cand.p:
            ph  =   cand.p
        if pl > cand.p:
            pl  =   cand.p
    h   =   (ph - pl) * 0.5 + 1.0
    p0  =   (ph + pl) * 0.5

    lw  =   0.5
    if cm.count > 2:
        lw  =   2

    return patches.Rectangle((cand.t - wt, p0 - h), wt * 2, h * 2, \
            fill = False, edgecolor = 'k', lw = lw)
    
def plot_cms(cms):

    fontsize    =   15
    matplotlib.rc('font', size = fontsize)
    
    plt.clf()
    fig =   plt.figure()
    fig.set_figwidth(6)
    fig.set_figheight(4.5)
    fig.subplots_adjust(left = 0.12, right = 0.95, top = 0.95, bottom = 0.15, \
            hspace = 0)
    ax = fig.add_subplot(111)

    mk  =   {1:'v', 2:'o', 4:'s'}
    c   =   {1:'b', 2:'c', 4:'m'}
    lbl =   {1:'Sh - Km', 2:'Sh - Ur', 4:'Km- Ur'}

    size    =  20 
    for cm in cms:
        if cm.count <= 1:
            continue
        for blid, cand in cm.d.items():
            plt.scatter(cand.t, cand.p, marker = mk[blid], c = c[blid], \
                    s = size, edgecolor = 'None')
            p   =   gen_patch(cm)
            ax.add_patch(p)

    plt.xlabel('Time [s]')
    plt.ylabel('SNR')

    plt.xlim(10, 170)
    plt.ylim(0, 60)

    ms  =   5
    
    ll  =   []
    for blid in [1, 2, 4]:
       ll.append(mlines.Line2D([], [], color = c[blid], marker = mk[blid], \
                markersize = ms, markeredgecolor = 'None', label = lbl[blid], \
                ls = 'None'))
    ll.append(patches.Patch(fill = False, lw = 0.5, label = '2 baselines'))
    ll.append(patches.Patch(fill = False, lw = 2.0, label = '3 baselines'))
    lg  =   plt.legend(handles = ll, prop = {'size': 12})
    lg.get_frame().set_linewidth(0)

    plt.savefig('scan%04d.eps' % (scan_no))

def main():

    if len(sys.argv) < 2:
        print '../crossmatch.py scan_no'
        sys.exit(0)

    global scan_no
    scan_no =   int(sys.argv[1])
    
    bls =   [1, 2, 4]
#    bls =   [4]

    cms =   []
    for blid in bls:
        cands   =   load_cand_file(blid) 
        match_cand(cands, cms) 

    output_cms(cms)
    plot_cms(cms)

if __name__ == '__main__':
    main()
