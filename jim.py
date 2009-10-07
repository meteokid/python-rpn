"""Module jim contains the classes used to work with icosahedral grid (Janusz Icosahedral Model)

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpn_version
from rpn_helpers import *
import jimc
import scrip

import math
import numpy

def jim_flatten_shape(field,nkfirst=False,nhalo=2):
    """Return the shape of Flattened data as packed by jim_flatten(field,keepnk)

    shape = jim_flatten_shape(jim_3d_field)
    shape = jim_flatten_shape(jim_3d_field,nkfirst,nhalo)

    @param jim_3d_field (numpy.ndarray(nij,nij,nk,ngrids))
    @param if nkfirst: jim_flat_field.shape == (nk,...),
           else: jim_flat_field.shape == (...,nk)
           (default=False)
    @param nhalo field's halo size (default=2) (int)
    @return shape (tuple of int)
    """
    #TODO: check that shape is what is expected
    nGrids = field.shape[3]
    nijh = field.shape[0]
    nk   = field.shape[2]
    if nkfirst:
        nijh = field.shape[1]
        nk   = field.shape[0]
    ij0  = nhalo
    ijn  = nijh - nhalo - 1
    nij  = (ijn-ij0+1)
    sizexy  = nij*nij
    npoles = 2
    if nGrids==1:
        npoles = 0
    if nkfirst:
        return (nk,sizexy*nGrids+npoles)
    else:
        return (sizexy*nGrids+npoles,nk)


def jim_flatten(field,nkfirst=False,nhalo=2):
    """Semi Flatten/pack data organized as a stack of 10 icosahedral grids with halo into a 2D array (i_j_grid,nk) without halo points

    jim_flat_field = jim_flatten(jim_3d_field)
    jim_flat_field = jim_flatten(jim_3d_field,nkfirst,nhalo)

    @param jim_3d_field (numpy.ndarray(nij,nij,nk,ngrids))
    @param if nkfirst jim_flat_field.shape == (nk,...),
           else jim_flat_field.shape == (...,nk)
           (default=False)
    @param nhalo jim_3d_field's halo size (default=2) (int)
    @return jim_flat_field (numpy.ndarray(nijg,nk)) or (numpy.ndarray(nk,nijg))

    Return a new numpy.ndarray (not a reference to the original one)
    """
    #TODO: check that shape is what is expected
    nGrids = field.shape[3]
    nijh = field.shape[0]
    nk   = field.shape[2]
    if nkfirst:
        nijh = field.shape[1]
        nk   = field.shape[0]
    ij0  = nhalo
    ijn  = nijh - nhalo - 1
    nij  = (ijn-ij0+1)
    npoles = 2
    if nGrids==1:
        npoles = 0
    NP_G = 0
    NP_I = ij0
    NP_J = ijn+1
    SP_G = 1
    SP_I = ijn+1
    SP_J = ij0
    sizexy  = nij*nij
    sizexyP = nij*nij+npoles
    f2 = numpy.resize(numpy.array([field[0,0,0,0]],order='FORTRAN'),jim_flatten_shape(field,nkfirst))
    np_val = None
    sp_val = None
    if nkfirst:
        np_val = field[:,NP_I,NP_J,NP_G]
        sp_val = field[:,SP_I,SP_J,SP_G]
    else:
        np_val = field[NP_I,NP_J,:,NP_G]
        sp_val = field[SP_I,SP_J,:,SP_G]
    ijg0 = 0
    ijgn = 0
    for g in range(nGrids):
        ijg0 = npoles+g*sizexy
        ijgn = ijg0+sizexy
        for k in range(nk):
            if nkfirst:
                if g==NP_G and npoles>0:
                    f2[k,0] = np_val[k]
                if g==SP_G and npoles>1:
                    f2[k,1] = sp_val[k]
                a = field[k,ij0:ijn+1,ij0:ijn+1,g]
                f2[k,ijg0:ijgn] = field[k,ij0:ijn+1,ij0:ijn+1,g].reshape(-1)
            else:
                if g==NP_G and npoles>0:
                    f2[0,k] = np_val[k]
                if g==SP_G and npoles>1:
                    f2[1,k] = sp_val[k]
                f2[ijg0:ijgn,k] = field[ij0:ijn+1,ij0:ijn+1,k,g].reshape(-1)
    return f2


def jim_unflatten(field,nkfirst=None,nhalo=2,nGrids=10):
    """Unpack JIM data to a stack of 10 icosahedral grids with halo
    Inverse operation of jim_flatten()

    jim_3d_field = jim_unflatten(jim_flat_field)
    jim_3d_field = jim_unflatten(jim_flat_field,nkfirst,nhalo,nGrids)

    @param jim_flat_field (numpy.ndarray)
    @param nkfirst jim_flat_field's packed order (bool)
    @param nhalo  jim_3d_field's halo size (default=2) (int)
    @param nGrids jim_flat_field's nb of packed grids (default=10) (int)
    @return jim_3d_field (numpy.ndarray)

    Return a new numpy.ndarray (not a reference to the original one)
    """
    #try to guess if nkfirst
    nijglist = (12,42,162,642,2562,10242,40962,163842,655362,2621442,10485762,41943042)
    if nkfirst is None:
        #TODO: to update, this only "works" for nGrids==10
        nkfirst=True
        if (field.shape[0] in nijglist) or field.shape[0]>nijglist[-1]:
            nkfirst=False
    if nkfirst:
        nijg = field.shape[1]
        nk   = field.shape[0]
    else:
        nijg = field.shape[0]
        nk   = field.shape[1]
    npoles = 2
    if nGrids==1:
        npoles = 0
    nij  = int(math.sqrt((nijg-npoles)/nGrids))
    nijh = nij + 2*nhalo
    ij0  = nhalo
    ijn  = nijh - nhalo - 1
    NP_G = 0
    NP_I = ij0
    NP_J = ijn+1
    SP_G = 1
    SP_I = ijn+1
    SP_J = ij0
    sizexy  = nij*nij
    sizexyP = nij*nij+npoles
    f2 = None
    if nkfirst:
        f2 = numpy.resize(numpy.array([field[0,0]],order='FORTRAN'),(nk,nijh,nijh,nGrids))
    else:
        f2 = numpy.resize(numpy.array([field[0,0]],order='FORTRAN'),(nijh,nijh,nk,nGrids))
    f2[...] = 0.
    ijg0 = 0
    ijgn = 0
    for g in range(nGrids):
        ijg0 = npoles+g*sizexy
        ijgn = ijg0+sizexy
        for k in range(nk):
            if nkfirst:
                if g==NP_G and npoles>0:
                    f2[k,NP_I,NP_J,NP_G] = field[k,0]
                if g==SP_G and npoles>1:
                    f2[k,SP_I,SP_J,SP_G] = field[k,1]
                #f3 = f2[ij0:ijn+1,ij0:ijn+1,k,g].reshape(-1)
                #f3[...] = field[k,ijg0:ijgn]
                f2[k,ij0:ijn+1,ij0:ijn+1,g] = field[k,ijg0:ijgn].reshape(nij,nij)
            else:
                if g==NP_G and npoles>0:
                    f2[NP_I,NP_J,k,NP_G] = field[0,k]
                if g==SP_G and npoles>1:
                    f2[SP_I,SP_J,k,SP_G] = field[1,k]
                #f3 = f2[ij0:ijn+1,ij0:ijn+1,k,g].reshape(-1)
                #f3[...] = field[ijg0:ijgn,k]
                f2[ij0:ijn+1,ij0:ijn+1,k,g] = field[ijg0:ijgn,k].reshape(nij,nij)
    return f2


def jim_grid_corners_la_lo(ndiv,igrid=0):
    """Compute RPNGridI (JIM) grid points centers and corners lat/lon

    (la,lo,cla,clo) = jim_grid_corners_la_lo(ndiv)
    (la,lo,cla,clo) = jim_grid_corners_la_lo(ndiv,igrid)

    @param ndiv  JIM grid number of division (int)
    @param igrid JIM grid number (default=0 : all 10 grids) (int)
    @return la,lo   Grid point centers lat,lon (numpy.ndarray(nijh,nijh))
    @return cla,clo Grid point corners lat,lon (numpy.ndarray(nc,nijh,nijh))
    """
    (la,lo,cla,clo) = jimc_grid_corners_la_lo(ndiv,igrid)
    s = list(la.shape)
    s.insert(2,1)
    la = la.reshape(s)
    lo = lo.reshape(s)
    cla = numpy.rollaxis(cla,2,0)
    clo = numpy.rollaxis(clo,2,0)
    return (la,lo,cla,clo)


class RPNGridI(RPNGridHelper):
    """RPNGrid Helper class for JIM-type grid (Icosahedron based)

    ip1-4: ndiv, itile, nhalo, npxy
    ndiv : number of division (factor 2) from base icosahedron
    itile = 0 global grid (ni=2+10*nijh^2,nj=1)
    itile = 1-10 ico-tile (ni=nj=nijh, min nhalo=1 to have poles)
    nhalo: number of points in the halo
    npxy : subdivision of each itile (ignored for itile==0)

    Multi-grid (10) icosahedral type
    The sum of the 10 grids is a global domain
    dx=dy and ni=nj are defined by ndiv=nb_grid_division (factor 2 from Icosahedron)
    The 2 poles sits in the halo, in grid 0 (North pole) and 9 (South) respectively

    Each of these grid can be divided into npx=npy "MPI-tiles"

    We define:
    ndiv: number of factor 2 grid division from the basic Icosahedron
    igrid: ico-grid number [1-10]
    itile: MPI tile (can be equivalent to igrid or a sub set of it)
    nij  : size of igrid along x or y (ni=nj)
    nhalo: size of the halo on each side (halox=haloy)
    nijh : nij + 2*nhalo
    i0,j0: position of itile SW corner in igrid [note that SW corner pt is not pt(1,1) of grid if there is a halo]
    npxy : MPI grid division along x and y (npx=npy)

    3 cases:
    1) Global grid (all igrid)
        TT grtyp=I, ig1=ndiv, ig2=0, ig3=nhalo
            ni=nijh*nijh*10+2, nj=1, nk=nk (or split nk)
    2) One igrid
        TT grtyp=I, ig1=ndiv, ig2=igrid [1-10], ig3=nhalo,
            ni=nj=nijh, nk=nk (or split nk)
    2B) One tile (MPI-tile #-grid)
        TT grtyp=#, ig1/2=grd_id, ig3=i0,ig4=j0
            ni=nj=nijh (of itile, not igrid), nk=nk (or split nk)
            nij and nhaloij can be computed (if we know npxy, from grd_id)
        >> grtyp=I, ig1=ndiv, ig2=igrid [1-10], ig3=nhalo, ig4=npxy
            ip1/2=grd_id,
            ni=nj=nk=1 if to recompute grid pts centers lon
            (otherwise ni=nijh [of igrid, not itile],nj=nijh)
        ^^ same has ">>" for grid pts centers lat

    myJIMgrid = JIMgrid(grtyp='#',ndiv=6,nhaloij=2,npxy=4,grd_id=(7,9))
    myJIMgrid = JIMgrid(myRPNRec)
    myJIMgrid = JIMgrid(keys=myRPNRec)

    """
    def parseArgs(self,keys,args):
        return {}

    #@static
    def argsCheck(self,d):
        #TODO: may want to check more things (shape rel to ndiv...)
        if not (d['grtyp'] != 'I'):
            raise ValueError, 'RPNGridBase: invalid grtyp value'

    def getRealValues(self,d):
        kv = {
            'ndiv'  : d['ig14'][0],
            'igrid' : d['ig14'][1],
            'nhalo' : d['ig14'][2],
            'npxy'  : d['ig14'][3]
        }
        return kv

    def getEzInterpArgs(keyVals,isSrc):
        #TODO: at some point implement as Y-grid if isSrc==False
        return None

    def toScripGrid(self,keyVals,name=None):
        kv = self.getRealValues(keyVals)
        if name is None:
            name = "grdI-%i-%i-%i-%i" % keyVals['ig14']
        (la,lo,cla,clo) = jim_grid_corners_la_lo(kv['ndiv'],kv['igrid'])
        la  = jim_flatten(la,nkfirst=False)
        lo  = jim_flatten(lo,nkfirst=False)
        cla = jim_flatten(cla,nkfirst=True)
        clo = jim_flatten(clo,nkfirst=True)
        la  *= (numpy.pi/180.)
        lo  *= (numpy.pi/180.)
        cla *= (numpy.pi/180.)
        clo *= (numpy.pi/180.)
        return scrip.ScripGrid(name,(la,lo,cla,clo))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
