#!/usr/bin/env python

import sys
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ambig   =   1.0E9 / 16.0E6 # in ns
hambig  =   ambig * 0.5

scan_no     =   -1
ph0_dict    =   {1: 0, 69: 0.581375, 71: 0.414097, 73: 0.246818}

def calc_psr_phase(t):

#    return 0.0

    c1  =   -6.89262836039208383e-03
    F0  =   1.399538059693
#    ph0 =   0.581375 # scan 69
#    ph0 =   0.414097 # scan 0071
#    ph0 =   0.246818 # scan 0073
    ph0 =   ph0_dict[scan_no]
    return (t - 10.000512) * (F0 + c1 / 60.) + ph0

def is_overlap(e1, e2):
#    if np.abs(e1.time - e2.time) <= np.max([e1.hrange, e2.hrange]):
    if np.abs(e1.time - e2.time) <= e1.hrange + e2.hrange:
        return True
    return False

def merge_mbd(t1, t2):
    if np.abs(t1 - t2) < ambig * 0.5:
        return (t1 + t2) * 0.5
    if t2 < t1:
        t2  +=  ambig
    else:
        t2  -=  ambig
    return (t1 + t2) * 0.5

def keep_mbd_ambig(mbd, a, hr):
    
    if mbd < a - hambig:
        mbd +=  ambig
    elif mbd > a + hambig:
        mbd -=  ambig
    if np.abs(mbd - a) < hr:
        return True
    return False


def keep_mbd(mbd, a, s):
    
    r   =   s * 2.0
    a0  =   a - r
    a1  =   a + r
    if mbd < a0:
        mbd +=  ambig
    elif mbd > a1:
        mbd -=  ambig
    if a0 <= mbd and mbd <= a1:
        return True
    return False

def calc_avg_std_mbd(mbd_l):
    
    a   =   np.average(mbd_l)
    r   =   np.std(mbd_l) * 2.0
    a0  =   a - r
    a1  =   a + r

    for i in range(len(mbd_l)):
        if mbd_l[i] < a0:
            mbd_l[i]    +=  ambig
        elif mbd_l[i] > a1:
            mbd_l[i]    -=  ambig
    return np.average(mbd_l), np.std(mbd_l)

def select_by_mbd(mbdarr, nsumlst):
    
    mbdavg  =   np.average(mbdarr)
    mbdstd  =   np.std(mbdarr)
    mbdrange    =   mbdstd * 2.0
    mbd0    =   mbdavg - mbdrange
    mbd1    =   mbdavg + mbdrange

    newarr  =   []
    nsumlst_sel =   []
    for i in range(len(mbdarr)):
        mbd     =   mbdarr[i]
        nsum    =   nsumlst[i]
        if mbd < mbd0:
            mbd +=  ambig
        elif mbd > mbd1:
            mbd -=  ambig
        if mbd >= mbd0 and mbd <= mbd1:
            newarr.append(mbd)  
            nsumlst_sel.append(nsum)
    return np.average(newarr) , np.std(newarr), nsumlst_sel

class Event(object):
    def __init__(self):
        self.time   =   0.0
        self.hrange =   0.0
        self.nsum   =   0
        self.p      =   0.0
        self.sbd    =   0.0
        self.mbd    =   0.0
        self.phase  =   0.0

    def __str__(self):
        return 't %f hrange %f nsum %d mbd %.3f pwr %.3f ph %.3f' % \
            (self.time, self.hrange, self.nsum, self.mbd, self.p, calc_psr_phase(self.time))

class Candidate(object):
    def __init__(self, e):
        self.time   =   e.time
        self.hrange =   e.hrange
        self.d      =   OrderedDict()
        self.d[e.nsum]  =   [e]


    def sum_to_file_max_power(self, f):

# loop each nsum to find out pmax and collect mbd
#        p_d     =   OrderedDict()

#        mbd_l   =   []
        pmax    =   -1.0
        for nsum, e_l in self.d.items():
            p_l =   []
            for e in e_l:
                if pmax < e.p:
                    pmax    =   e.p
                    e_max   =   e
#                mbd_l.append(e.mbd)

#        mbd_avg, mbd_std    =   calc_avg_std_mbd(mbd_l)
       
        f.write("%.6f\t%.6f\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n" % \
                (e_max.time, e_max.hrange, self.hrange, e_max.mbd, 0.0, \
                 e_max.p, 0.0, calc_psr_phase(e_max.time), len(self.d)))
 
    def sum_to_file_max_width(self, f):

# loop each nsum to find out pmax and collect mbd
#        p_d     =   OrderedDict()
        mbd_l   =   []
        pmax    =   -1.0
        nsum_pmax=   -1
        for nsum, e_l in self.d.items():
            p_l =   []
            for e in e_l:
                p_l.append(e.p)
                mbd_l.append(e.mbd)
            p_avg   =   np.average(p_l)
            if pmax < p_avg:
                pmax        =   p_avg
                nsum_pmax    =   nsum
#            p_d[nsum]   =   np.average(p_l)

        mbd_avg, mbd_std    =   calc_avg_std_mbd(mbd_l)
       
        l0  =   []
        l1  =   []
        p_l =   []
        for e in self.d[nsum_pmax]:
            l0.append(e.time - e.hrange)
            l1.append(e.time + e.hrange)
            p_l.append(e.p)
        t0  =   np.min(l0)
        t1  =   np.max(l1)
        t   =   (t0 + t1) * 0.5
        hr  =   (t1 - t0) * 0.5
    
        f.write("%.6f\t%.6f\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n" % \
                (t, hr, self.hrange, mbd_avg, mbd_std, pmax, \
                 np.std(p_l), calc_psr_phase(t), len(self.d)))
 
    def sum_to_file_old(self, f):

        nsumlst =   []
        mbdarr  =   []
        for nsum, e in self.d.items():

            nsumlst.append(nsum)
            mbdarr.append(e.mbd)

        mbdavg, mbdstd, nsumlst_sel   =   select_by_mbd(mbdarr, nsumlst)

        nsum_min    =   1024
        ph          =   0.0
        t           =   0.0
        hr_min      =   0.0

#        pharr   =   []
#        tarr    =   []
        parr    =   []
        for nsum, e in self.d.items():

            if not (nsum in nsumlst_sel):
                continue

            if nsum_min > nsum:
                nsum_min    =   nsum
                ph          =   e.phase
                t           =   e.time 
                hr_min      =   e.hrange

#            pharr.append(e.phase)
#            tarr.append(e.time) 
            parr.append(e.p)         
            mbdarr.append(e.mbd) 
         
        f.write("%.6f\t%.6f\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n" % \
                (t, hr_min, self.hrange, mbdavg, mbdstd, np.average(parr), \
                 np.std(parr), ph, len(nsumlst_sel)))
        
    def __str__(self):
        return 'time: %f, hrange: %f, nitems: %d' % \
            (self.time, self.hrange, len(self.d))

    def insert_and_keep(self, ei):
        if self.d.has_key(ei.nsum):
            self.d[ei.nsum].append(ei)
        else:
            self.d[ei.nsum] =   [ei]
        hr0  =   np.min([self.time - self.hrange, ei.time + ei.hrange])
        hr1  =   np.max([self.time + self.hrange, ei.time + ei.hrange])
        self.time   =   (hr0 + hr1) * 0.5
        self.hrange =   (hr1 - hr0) * 0.5

    def insert(self, ei):
        
        for nsum, e in self.d.items():
            if nsum != ei.nsum:
                if not is_overlap(e, ei):
                    return
        if self.d.has_key(ei.nsum):
            e   =   self.d[ei.nsum]
#            if  np.abs(e.mbd - ei.mbd) > 5.0: # if mbd diff > 5 ns, exclude
#                return
            e.time      =   (e.time + ei.time) * 0.5
            e.hrange    *=  0.5
            ei.time     =   e.time
            ei.hrange   =   e.hrange
            e.p         =   (e.p + ei.p) * 0.5
            e.phase     =   (e.phase + ei.phase) * 0.5
#            print 'nsum %d, time %f merged (%.3f|%.3f)' % (e.nsum, e.time, e.mbd, ei.mbd)
            e.mbd       =   merge_mbd(e.mbd, ei.mbd)
        else:
            self.d[ei.nsum]  =   ei

        if self.hrange < ei.hrange:
            self.time   =   ei.time
            self.hrange =   ei.hrange

class Match(object):
    def __init__(self):
        self.prefix     =   ''
        self.nsum_list  =   []
        self.aptime     =   0.0
        self.sigma      =   0.0
        self.factor     =   0.0
        self.blid       =   -1

        self.cl         =   []

        self.ne_min     =   0

    def load_fitdump_both(self, blid, nsum):

#        print 'nsum %d, noffset %d...' % (nsum, noffset)

        noffset =   0
        fname   =   '%s/bl%03d_sum%03d_offset%03d.fitdump' % \
                        (self.prefix, blid, nsum, noffset)
        a0 =   np.loadtxt(fname, dtype = 'float')


        noffset =   nsum / 2
        fname   =   '%s/bl%03d_sum%03d_offset%03d.fitdump' % \
                        (self.prefix, blid, nsum, noffset)
        a1 =   np.loadtxt(fname, dtype = 'float')

        a   =   np.concatenate((a0, a1), axis = 0)

        hrange  =   self.aptime * nsum * 0.5 * self.factor

        idx =   np.argsort(a[:, 1])
        a   =   a[idx]
        t   =   a[:, 1]
    
        ps  =   a[:, 4]
        ntot    =   len(a)
        n0  =   np.int(ntot * 10. / 300)
        n1  =   np.int(ntot * 290. / 300.)
        ps1 =   ps[n0:n1]
        ave =   np.average(ps1)
        std =   np.std(ps1)

#        plt.clf()
#        plt.scatter(t, ps / std, marker = '.', s = 0.5, edgecolor = 'none')
#        plt.xlim(10, 170)
#        plt.ylim(0, 15)
#        plt.savefig('fd_bl%d_sum%d_offset%d.png' % (blid, nsum, noffset))

        print 'nsum: %d, ave: %f, std: %f' % (nsum, ave, std)
#        print 'return after ave, std output...'
#        sys.exit(0)

#        ids =   np.where(ps > 328382.790122)[0]
        ps  =   (ps - ave) / std
        ids =   np.where(ps > self.sigma)[0]
     
        es  =   []
        for id in ids:
            e       =   Event()
            e.time  =   a[id, 1]
            e.hrange=   hrange
            e.nsum  =   nsum
            e.p     =   ps[id]
            e.mbd   =   a[id, 6]
            e.phase =   a[id, 2]
            es.append(e)
        return es
    
    def load_fitdump(self, blid, nsum, noffset):

#        print 'nsum %d, noffset %d...' % (nsum, noffset)

        fname   =   '%s/bl%03d_sum%03d_offset%03d.fitdump' % \
                        (self.prefix, blid, nsum, noffset)
        a =   np.loadtxt(fname, dtype = 'float')

        hrange  =   self.aptime * nsum * 0.5 * self.factor
    
        ps  =   a[:, 4]
        ave =   np.average(ps)
        std =   np.std(ps)

        print 'nsum: %d, ave: %f, std: %f' % (nsum, ave, std)
#        print 'return after ave, std output...'
#        sys.exit(0)

#        ids =   np.where(ps > 328382.790122)[0]
        ps  =   (ps - ave) / std
        ids =   np.where(ps > self.sigma)[0]
     
        es  =   []
        for id in ids:
            e       =   Event()
            e.time  =   a[id, 1]
            e.hrange=   hrange
            e.nsum  =   nsum
            e.p     =   ps[id]
            e.mbd   =   a[id, 6]
            e.phase =   a[id, 2]
            es.append(e)
        return es

    def insert_to_cl(self, e):
#        if len(self.cl) == 0:
#            self.cl.append(Candidate(e))
#            return
        has_overlap =   False
        for c in self.cl:
            if is_overlap(c, e):
                has_overlap =   True
                c.insert_and_keep(e)
                break
        if not has_overlap:
            self.cl.append(Candidate(e))

    def trim_cl(self):

        hambig  =   ambig * 0.5
#        hr  =   ambig / 4.0
        hr  =   2.0

        for i in range(len(self.cl)):
            
            c   =   self.cl[i]
            ne  =   len(c.d)
            if ne < 3:
                continue
# first loop to collect mbd and to calc avg, std
            mbd_l   =   []
            p_l     =   []
            for nsum, e_l in c.d.items():
                for e in e_l:
                    mbd_l.append(e.mbd)
                    p_l.append(e.p)
            a   =   np.average(mbd_l, weights = p_l)
            s   =   np.std(mbd_l)

# resolve ambig:
            for k in range(len(p_l)):
                if mbd_l[k] < a - hambig:
                    mbd_l[k]    +=  ambig
                if mbd_l[k] > a + hambig:
                    mbd_l[k]    -=  ambig

# calculate average again:
            a   =   np.average(mbd_l, weights = p_l)
            
# second loop to exclude mbd with large deviation
            dnew    =   OrderedDict()
            for nsum, e_l in c.d.items():
                l   =   []
                for e in e_l:
                    if keep_mbd_ambig(e.mbd, a, hr):
                        l.append(e)
                if len(l) > 0:
                    self.cl[i].d[nsum]   =   l
                else:
                    self.cl[i].d.pop(nsum)
#            break

    def print_cl(self):

        fname   =   'bl%03d.log' % (self.blid)
        f       =   open(fname, 'w')
        for i in range(len(self.cl)):
            c   =   self.cl[i]
            ne  =   len(c.d)
#            if ne == len(self.nsum_list):
            if ne >= self.ne_min:
#                print '%d:' % (i)
                f.write('#%d:\n' % (i))
                for nsum, e_l in c.d.items():
#                    print nsum, e.time, e.hrange, e.p, e.phase
                    for e in e_l:
                        f.write(e.__str__() + '\n')
#                        print e
#                print ''
                f.write('###\n\n')
        f.close()

    def sum_cl(self):
        
        fname   =   'bl%03d.nsum' % (self.blid)
        f   =   open(fname, 'w')
        ncl =   len(self.cl)
        for i in range(ncl):
            c   =   self.cl[i]
            ne  =   len(c.d)
            if ne >= self.ne_min:
                c.sum_to_file_max_power(f)  
        f.close()

def main(blid):

    if len(sys.argv) < 2:
        print '../winmatch.py scan_no'
        sys.exit(0)

    global scan_no
    scan_no =   int(sys.argv[1])

    m   =   Match()

    m.blid  =   blid

    m.nsum_list =   [4, 8, 16, 24, 32]
#    m.nsum_list =   [32]
    m.nsum_list =   np.sort(m.nsum_list)[::-1]
    m.ne_min    =   3
#    m.ne_min    =   int(np.ceil(len(m.nsum_list) * 0.5))

    m.prefix  =   '.'
    m.aptime  =   1.024E-3  # in s
    m.sigma   =   3.0
    m.factor  =   1.0

    for nsum in m.nsum_list:

#        es0 =   m.load_fitdump(m.blid, nsum, 0)
#        es1 =   m.load_fitdump(m.blid, nsum, nsum / 2)
#        es  =   es0 + es1
        es  =   m.load_fitdump_both(m.blid, nsum)

        t_arr   =   []
        for e in es:
            t_arr.append(e.time) 
        ids =   np.argsort(t_arr)
        for id in ids:
            m.insert_to_cl(es[id])

#        es  =   m.load_fitdump(m.blid, nsum, 0)
#        for e in es:
#            m.insert_to_cl(e)
#        es  =   m.load_fitdump(m.blid, nsum, nsum / 2)
#        for e in es:
#            m.insert_to_cl(e)

    m.trim_cl()
    m.print_cl()
    m.sum_cl()

if __name__ == '__main__':
    bls =   [1, 2, 4] 
    for blid in bls:
        main(blid)
