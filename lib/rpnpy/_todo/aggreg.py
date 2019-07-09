#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author:  <@canada.ca>
"""
Perform aggregartion on a field.
"""
import os
import numpy as np
import rpnpy.librmn.all as rmn
import pylab
import scipy.ndimage

def aggregate0(d, fac):
    """
    Perform aggregation

    Args:
       d (ndarray) : data to aggregate
       fac (float) : aggregation factor
    Returns:
       ndarray, aggregated data
    """
    ni, nj = d.shape
    shape = (int(ni/fac), int(nj/fac))
    # WARNING: rpnpy get its fields from librmn fst functions
    #          these fields are most of the times Fortran real*4 array
    #          while the default numpy array is C double
    #TODO: use np.asarray?
    d2  = np.zeros(shape, dtype=np.float32, order='F')
    for i_lo in range(shape[0]):
        for j_lo in range(shape[1]):
            i_hi_beg = i_lo*int(fac)
            j_hi_beg = j_lo*int(fac)
            i_hi_end = min((i_lo + 1)*int(fac) - 1, ni)
            j_hi_end = min((j_lo + 1)*int(fac) - 1, nj)
            d2[i_lo,j_lo] =  pylab.mean(
                d[i_hi_beg:i_hi_end+1, j_hi_beg:j_hi_end+1])
    return d2

def aggregate1(d, fac):
    ## http://stackoverflow.com/questions/25173979/aggregate-numpy-array-by-summing
    ## http://stackoverflow.com/questions/27274604/python-aggregate-groupby-2d-matrix
    ni, nj = d.shape
    if (ni%fac, nj%fac) != (0,0):
        raise ValueError('This function only supports dimensions that are an integer multiple of the aggregation factor. shape=({}, {}), fac={}'.format(ni,nj,fac))
    ii = np.arange(0, ni, fac)
    jj = np.arange(0, nj, fac)
    step1 = np.add.reduceat(d, ii, axis=0)/float(fac)
    step2 = np.add.reduceat(step1, jj, axis=1)/float(fac)
    return step2

def aggregate2(d, fac):
    ## http://stackoverflow.com/questions/25173979/aggregate-numpy-array-by-summing
    ## http://stackoverflow.com/questions/27274604/python-aggregate-groupby-2d-matrix
    ni, nj = d.shape
    ii = np.arange(0, ni, fac)
    jj = np.arange(0, nj, fac)
    w = np.ones_like(d)
    w = np.add.reduceat(w, ii, axis=0)
    w = np.add.reduceat(w, jj, axis=1)
    tmp = np.add.reduceat(d, ii, axis=0)
    o = np.add.reduceat(tmp, jj, axis=1) / w
    return o

def aggregate3(d, weights, xyfac=None):
    #TODO: accept weights, xyfac as 1d... make 2d
    order = 'F' if d.flags['F_CONTIGUOUS'] else 'C'
    ## if weights is None:
    ##     weights = np.full((fac, fac), 1., dtype=np.float32, order=order)
    ni, nj = d.shape
    xfac, yfac = weights.shape if xyfac is None else xyfac
    #TODO: what if weights.shape < xyfac
    nij = (int(round(float(ni)/float(xfac))), int(round(float(nj)/float(yfac))))
    o = np.zeros(nij, dtype=d.dtype, order=order)
    w = np.zeros(nij, dtype=np.float32, order=order)
    #TODO: deal with borders
    for dj in range(yfac):
        for di in range(xfac):
            o[:,:] += d[di::xfac,dj::yfac] * weights[di,dj]
            w[:,:] += weights[di,dj]
    o /= w
    return o

def interp_near(d, zoom):
    return scipy.ndimage.zoom(d, zoom, order=0) #order can be 1 for linear

ni, nj = 5000, 5500
## ni, nj = 44, 22
##ni, nj = 15, 10
fac = 5

## ni, nj = 3, 2
## fac = 1 # 5
## zoom = 3 # Not working for zoom > 2...  zoomed image is not right! Try with more recent version of scipy on u14 (same thing!)

d=np.reshape(np.arange(ni*nj),(ni,nj), order='F')

d0 = aggregate0(d,fac)
## print d0

d=np.reshape(np.arange(ni*nj),(ni,nj), order='F')
d1 = aggregate1(d,fac)
## print d1

d=np.reshape(np.arange(ni*nj),(ni,nj), order='F')
d2 = aggregate2(d,fac)

## d=np.reshape(np.arange(ni*nj),(ni,nj), order='F')
## dz = interp_near(d, zoom)
## d2 = aggregate1(dz,fac*zoom)

d=np.reshape(np.arange(ni*nj),(ni,nj), order='F')
weights = np.ones((fac, fac), dtype=np.float32, order='F')
d3 =  aggregate3(d, weights)

weights = np.ones((fac+2, fac+2), dtype=np.float32, order='F')
d3b =  aggregate3(d, weights, (fac, fac))

## dd = np.abs(d0 - d1)
## print 'all eq:', np.all(d0 == d1), (float(np.where(dd != 0)[0].size)/float(d0.size)*100., '%')
## print 'meanval0:', np.mean(d0.flat), '; maxval =', np.max(d0)
## print 'meanval0:', np.mean(d1.flat), '; maxval =', np.max(d1)
## print 'meandiff:', np.mean(dd.flat), '; maxdiff=', np.max(dd)

dd = np.abs(d1 - d2)
print 'all eq:', np.all(d1 == d2), (float(np.where(dd != 0)[0].size)/float(d1.size)*100., '%')
print 'meanval1:', np.mean(d1.flat), '; maxval =', np.max(d1)
print 'meanval2:', np.mean(d2.flat), '; maxval =', np.max(d2)
print 'meandiff:', np.mean(dd.flat), '; maxdiff=', np.max(dd)

dd = np.abs(d1 - d3)
print 'all eq:', np.all(d1 == d3), (float(np.where(dd != 0)[0].size)/float(d1.size)*100., '%')
print 'meanval1:', np.mean(d1.flat), '; maxval =', np.max(d1)
print 'meanval3:', np.mean(d3.flat), '; maxval =', np.max(d3)
print 'meandiff:', np.mean(dd.flat), '; maxdiff=', np.max(dd)

print d
## print dz
## print d[1,0], d[0,1]
## print scipy.ndimage.zoom(d, 2, order=0)
print d0
print d1
print d2
print d3

## n = np.where(dd != 0)
## print n[0].shape, d0.shape, d0.size
## print np.mean(dd.flat), np.max(dd)
## for k in range(10):
##     ia, ja = n[0][k], n[1][k]
##     i0, j0 = ia*fac, ja*fac
##     print (i0, j0, d[i0,j0]), ':', (d0[ia, ja], '-', d1[ia, ja], '=', (ia, ja, dd[ia, ja]))

## python -m cProfile -s 'time' aggreg.py | grep aggregate
##         1    5.440    5.440   17.454   17.454 aggreg.py:12(aggregate0)
##         1    0.023    0.023    0.175    0.175 aggreg.py:39(aggregate1)
## 
##         1    5.207    5.207   16.805   16.805 aggreg.py:13(aggregate0)
##         1    0.250    0.250    0.253    0.253 aggreg.py:65(aggregate3)
##         1    0.124    0.124    0.355    0.355 aggreg.py:52(aggregate2)
##         1    0.062    0.062    0.178    0.178 aggreg.py:40(aggregate1)

