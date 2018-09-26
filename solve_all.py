#!/usr/bin/env python

import numpy as np
#import matplotlib.pyplot as plt
from astropy.io import fits
import ctypes
import partial

nu  =   2048
nv  =   2048
cellsize    =   0.002 # arcsec
binwid      =   2
nmask   =   2
NGCF    =   301
tg2cg   =   (NGCF - 1.0) / (nmask + 0.5)
c_light =   299792458.0  # speed of light

libcalc =   ctypes.CDLL('calc9.1/libcalc_cwrapper.so')

def call_by_struct(din):

    dout    =   partial.COut()
    libcalc.call_calc_by_struct(ctypes.byref(din), ctypes.byref(dout))
    return np.array([dout.pd_pra, dout.pd_pdec])

def prep_gcf():

    hwhm    =   0.7
    cghwhm  =   tg2cg * hwhm
    recvar  =   np.log(2.0) / cghwhm ** 2
    convfn  =   np.zeros(NGCF, dtype = float)
    for i in range(NGCF):
        convfn[i]   =   np.exp(-recvar * i**2)

    return convfn

def prep_uv(d):

    nd  =   len(d)
#    nchan   =   1
    nchan   =   32
    nif     =   6
    
    f0  =   2192.0E6
    bw  =   16.0E6
    dbw =   bw / nchan

    farr    =   np.zeros((nif, nchan), dtype = float)
    wl      =   np.zeros((nif, nchan), dtype = float)
    for fid in range(nif):
        for cid in range(nchan):
            farr[fid, cid]  =   f0 + bw * fid + dbw * cid
            wl[fid, cid]    =   c_light / farr[fid, cid]

    vis  =   np.reshape(d['DATA'], (-1, 6, nchan, 3))
    vis  =   vis[:, :, :, 0] + 1j * vis[:, :, :, 1]

    uu  =   d['UU---SIN'] * c_light
    vv  =   d['VV---SIN'] * c_light
    
    vals    =   []
    uv  =   []
    for i in range(nd):
        for fid in range(nif):
            for cid in range(nchan):
                if vis[i, fid, cid] != 0.0:
                    vals.append(vis[i, fid, cid])
                    uv.append([uu[i] / wl[fid, cid], vv[i] / wl[fid, cid]])
                    
    vals    =   np.array(vals)
    uv      =   np.array(uv)

    print 'total vis retrived: %d' % (len(uv))
    return vals, uv

def image(d, fid):

# This operation will lead to wronge UU---SIN value. A bug? Try avoid 
# caoncatenate operation on dtypes
#    d   =   np.concatenate((d[258], d[259], d[515]), axis = 0)
    
    urange  =   1. / (cellsize / 3600. / 180. * np.pi)
    vrange  =   urange
    umin    =   -urange * 0.5
    vmin    =   -vrange * 0.5
    du      =   urange / nu
    dv      =   vrange / nv
    
    dubc    =   du * binwid
    dvbc    =   dv * binwid
    
    bc  =   np.zeros((nv / binwid + 1, nu / binwid + 1), dtype = int)

    uvimg       =   np.zeros((nv, nu), dtype = 'c8')
    cval, uv    =   prep_uv(d)

    cval    =   np.concatenate((cval, np.conj(cval)), axis = 0)
    uv      =   np.concatenate((uv, -uv), axis = 0)

#    plt.plot(uv[:, 0], uv[:, 1], 'r.')
#    plt.xlim(umin, -umin)
#    plt.ylim(vmin, -vmin)
#    plt.show()

    for i in range(len(uv)):
        iu  =   int((uv[i, 0] - umin) / dubc + 0.5) 
        iv  =   int((uv[i, 1] - vmin) / dvbc + 0.5)
        bc[iv, iu]  +=  1

    convfn  =   prep_gcf()
 
    bm_wsum =   0.0
    bm_muu  =   0.0
    bm_mvv  =   0.0
    bm_muv  =   0.0

    wsum    =   0.0
    for i in range(len(uv)):
        ufrc    =   (uv[i, 0] - umin) / du
        vfrc    =   (uv[i, 1] - vmin) / dv
        upix  =   int(ufrc + 0.5) 
        vpix  =   int(vfrc + 0.5)

        iubc  =   int((uv[i, 0] - umin) / dubc + 0.5) 
        ivbc  =   int((uv[i, 1] - vmin) / dvbc + 0.5)
#        w   =   1. / (bc[ivbc, iubc]) 
        w   =   1.0

        uu  =   uv[i, 0]
        vv  =   uv[i, 1]
        bm_wsum +=  w
        runwt   =   w / (bm_wsum)
        bm_muu  +=  runwt * (uu * uu - bm_muu)
        bm_mvv  +=  runwt * (vv * vv - bm_mvv)
        bm_muv  +=  runwt * (uu * vv - bm_muv)

        for iv in range(vpix - nmask, vpix + nmask + 1):
            fv  =   w * convfn[int(tg2cg * np.abs(iv - vfrc) + 0.5)]
            for iu in range(upix - nmask, upix + nmask + 1):
                fuv  =   fv * convfn[int(tg2cg * np.abs(iu - ufrc) + 0.5)] 
                wsum +=  fuv
                uvimg[iv, iu] +=  fuv * cval[i]
    
    uvimg   /=  wsum
    uvimg   =   np.fft.fftshift(uvimg)
    img     =   np.fft.fft2(uvimg)
    img     =   np.fft.fftshift(img)
    img     =   np.real(img)

    idpeak  =   np.argmax(img)
    iypeak  =   idpeak / nu
    ixpeak  =   idpeak % nu

    rapeak  =   (ixpeak - nu / 2) * cellsize
    decpeak =   (iypeak - nv / 2) * cellsize

    print 'Ra offset: %.3f mas' % (rapeak * 1E3)
    print 'Dec offset: %.3f mas' % (decpeak * 1E3)

    fudge   =   0.7
    ftmp    =   np.sqrt((bm_muu - bm_mvv) ** 2 + 4.0 * bm_muv ** 2)
    e_bpa   =   -0.5 * np.arctan2(2.0 * bm_muv, bm_muu - bm_mvv)
    e_bmin  =   fudge / (np.sqrt(2.0 * (bm_muu + bm_mvv) + 2.0 * ftmp))
    e_bmaj  =   fudge / (np.sqrt(2.0 * (bm_muu + bm_mvv) - 2.0 * ftmp))

    rad2mas =   180. / np.pi * 3600. * 1000. 
    print 'bmin = %.4f mas, bmax = %.4f mas, bpa = %.4f degrees' % \
            (e_bmin * rad2mas, e_bmaj * rad2mas, e_bpa / np.pi * 180.)

    avgi    =   np.average(img)
    stdi    =   np.std(img)
    maxi    =   np.max(img)
    snr     =   (maxi - avgi) / stdi
    print 'average: %.3e, std: %.3e, max: %.3e, snr: %.1f' % (avgi, stdi, maxi, snr)

    np.save('img%04d.npy' % fid, img)

    return [rapeak * 1E3, decpeak * 1E3, e_bmin * rad2mas, e_bmaj * rad2mas, e_bpa, snr]

#    plt.imshow(img, vmin = np.min(img), vmax = np.max(img))
#    plt.show()

def plot_vis(vis):
    a   =   np.angle(vis, deg = True)
    x   =   np.arange(len(vis))
    plt.plot(x, a, 'rs')
    plt.xlim(0, x[-1])
    plt.ylim(-180, 180)
    plt.show()

def fit(d):

    if len(d) == 0:
        return 0.0, 1E5

    nif     =   6
    nchan   =   32
   
    a   =   d['DATA'].copy()
    a   =   np.reshape(a, (-1, nif, nchan, 3))
    a   =   np.sum(a, axis = 0)
    a   =   np.reshape(a, (-1, 3))
    vis =   a[:, 0] + 1j * a[:, 1]

    bw  =   96E6
    nchan_tot   =   nchan * nif
    assert nchan_tot == len(vis)
    dbw =   bw / nchan_tot
    
    nfft    =   nchan_tot * 16
    spec    =   np.fft.fft(vis, n = nfft)
    spec    =   np.fft.fftshift(spec) 
    
    mag     =   np.abs(spec)
    idx     =   np.argmax(mag)
    trange  =   1. / dbw
    dt      =   trange / nfft
    t       =   (np.arange(nfft) - nfft / 2) * dt
    tau0    =   t[idx]
    print 'Coarse delay (FFT): %.3f ns' % (tau0 * 1E9) 

    fx  =   np.arange(nchan * nif) * dbw
    vis =   vis * np.exp(-1j * 2 * np.pi * fx * tau0)

    ph0 =   np.angle(np.sum(vis))
    vis =   vis * np.exp(-1j * ph0)

    y   =   np.angle(vis)
    w =   np.abs(vis)
    p, v    =   np.polyfit(fx, y, 1, w = w, full = False, cov = True)
    print v
    tau1    =   p[0] / (2. * np.pi)
    vis =   vis * np.exp(-1j * 2 * np.pi * fx * tau1)

    y   =   np.angle(vis)

#    sigma2   =   np.sum((y - np.polyval(p, fx)) ** 2 * w) / np.sum(w)
    sigma2   =   v[0, 0] / (2. * np.pi) ** 2
    print 'Find delay (polyfit): %.3f (%.3f) ns' % \
            (tau1 * 1E9, np.sqrt(sigma2) * 1E9)

    return tau0 + tau1, sigma2
#    plot_vis(vis)

def load_fits():

    hdu =   fits.open('PSR_NOSUM.FITS')
    d   =   hdu[0].data
    return d

def prep_partial(sp, din, f, sid):

    if len(sp) == 0:
        return np.array([0.0, 0.0])

    set_datetime(din, dt)
    set_site(din, f, sid)

    return call_by_struct(din)
    
def solve(sp, f, id):

    din =   partial.CIn()
    partial.set_eop(din, 'eop.txt')
    set_src(din, f)

    bls =   [258, 259, 515]
    nstn    =   len(bls)
    nbl =   ((nstn - 1) * nstn) / 2

    y       =   np.zeros(nbl, dtype = float)
    sig2    =   np.zeros(nbl, dtype = float)

    nbl_fit =   0
    for i in range(nbl):
        ids =   np.where(sp['BASELINE'] == bls[i])[0]
        y[i], sig2[i]   =   fit(sp[ids])
        if sig2[i] < 1E5:
#            nbl_fit +=  1 
            nbl_fit +=  1 << i
    print y

    dts     =   sp['DATE'].copy()
    dts     -=  (2400000 + 0.5)
    dt      =   np.average(dts)
    din.date    =   int(dt)
    din.time    =   dt - din.date

    ps  =   []
    for i in range(nstn):
        set_site(din, f, i)
        ps.append(call_by_struct(din))

    A   =   []
    for i in range(0, nbl):
        for j in range(i + 1, nbl):
            A.append(ps[i] - ps[j]) 
    A   =   np.array(A)

    np.save('fit_y_%04d.npy' % (id), y)
    np.save('fit_sig2_%04d.npy' % (id), sig2)
    np.save('fit_A_%04d.npy' % (id), A)

    y       =   y[:, np.newaxis]
    Msig    =   np.diag(sig2)
    W       =   np.linalg.inv(Msig)
    At      =   np.transpose(A)
    m0  =   np.dot(np.dot(At, W), A)
    m0  =   np.linalg.inv(m0)
    x   =   np.dot(np.dot(np.dot(m0, At), W), y)
    print x / np.pi * 180. * 3600.
    m1  =   np.linalg.inv(Msig)
    m1  =   np.dot(np.dot(At, m1), A)
    Merr    =   np.linalg.inv(m1)
    print Merr
    sig_ra  =   np.sqrt(Merr[0, 0]) / np.pi * 180. * 3600 * np.cos(din.dec)
    sig_dec =   np.sqrt(Merr[1, 1]) / np.pi * 180. * 3600
    ra  =   x[0, 0] / np.pi * 180. * 3600 * np.cos(din.dec)
    dec =   x[1, 0] / np.pi * 180. * 3600

    print 'Ra offset: %.3f (%.3f) mas, Dec offset: %.3f (%.3f) mas' % \
            (ra * 1E3, sig_ra * 1E3, dec * 1E3, sig_dec * 1E3)

    return [nbl_fit, -ra * 1E3, sig_ra * 1E3, -dec * 1E3, sig_dec * 1E3, din.date + din.time]

def set_datetime(din, dt):
    din.date    =   int(dt)
    din.time    =   dt - din.date

def set_site(din, f, sid):
    
    d   =   f[3].data[sid]
    din.x   =   d['STABXYZ'][0]
    din.y   =   d['STABXYZ'][1]
    din.z   =   d['STABXYZ'][2]
    din.axis_off    =   d['STAXOF']
    din.stnname     =   d['ANNAME']

def set_eop(din, f):
    partial.set_eop(din, 'psrf02.vex')

def set_src(din, f):
    h   =   f[0].header
    din.srcname     =   h['OBJECT']
    assert h['CTYPE6']  ==  'RA'
    din.ra          =   h['CRVAL6'] / 180. * np.pi
    assert h['CTYPE7']  ==  'DEC'
    din.dec         =   h['CRVAL7'] / 180. * np.pi

def main():

    bls =   [258, 259, 515]

    f   =   fits.open('PSR_NOSUM.FITS')

    d   =   f[0].data

    mjd     =   57068
    mjd     +=  2400000 + 0.5
    
    t   =   d['DATE'].copy()
    t   -=  mjd
    t   *=  86400.

    ids =   np.argsort(t)
    t   =   t[ids]
    d   =   d[ids]

    sps     =   []
    tmax    =   -1.0
#    for rec in d:
    for i in range(len(d)):
        if np.abs(tmax - t[i]) > 0.05:
            tmax    =   t[i]
            sps.append([])
        sps[-1].append(i)

    res_sps =   []
    for i in range(len(sps)):
        sp  =   sps[i]
        print 'sp %d:' % (i)
        bl0  =   d[sp]['BASELINE']
        for bl in bls:
            print 'bl %d: %d APs' % (bl, len(np.where(bl0 == bl)[0]))
        print ''

        res_sp  =   []
        res_sp  +=  solve(d[sp], f, i)
        res_sp  +=  image(d[sp], i)
        res_sps.append(res_sp)

    res =   np.array(res_sps) 
    np.save('loc_sp.npy', res)
   
if __name__ == '__main__':
    main()
