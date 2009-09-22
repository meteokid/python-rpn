"""Module jimc contains the classes used to work with icosahedral grid (Janusz Icosahedral Model)

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
    @date: 2009-09
"""
import math
import numpy
import rpnstd
import jimc

__JIM_VERSION__ = '0.1-dev'
__JIM_LASTUPDATE__ = '2009-09'

def jim_flatten_shape(field,nkfirst=False):
    """Return the shape of Flattened data as packed by jim_flatten(field,keepnk)

    shape = jim_flatten_shape(jim_3d_field,nkfirst)

    @param jim_3d_field (numpy.ndarray(nij,nij,nk,ngrids))
    @param if nkfirst: jim_flat_field.shape == (nk,...),
           else: jim_flat_field.shape == (...,nk)
    @return shape (tuple of int)
    """
    #TODO: check that shape is what is expected
    HALO = 2
    IJ0  = HALO
    IJn  = field.shape[0] - HALO - 1
    NIJ  = (IJn-IJ0+1)
    NK   = field.shape[2]
    NGRIDS = field.shape[3]
    SIZEXY  = NIJ*NIJ
    if nkfirst:
        return (NK,SIZEXY*NGRIDS+2)
    else:
        return (SIZEXY*NGRIDS+2,NK)


def jim_flatten(field,nkfirst=False):
    """Semi Flatten/pack data organized as a stack of 10 icosahedral grids with halo into a 2D array (i_j_grid,nk) without halo points

    jim_flat_field = jim_flatten(jim_3d_field,nkfirst)

    @param jim_3d_field (numpy.ndarray(nij,nij,nk,ngrids))
    @param if nkfirst jim_flat_field.shape == (nk,...),
           else jim_flat_field.shape == (...,nk)
    @return jim_flat_field (numpy.ndarray(nijg,nk)) or (numpy.ndarray(nk,nijg))

    Return a new numpy.ndarray (not a reference to the original one)
    """
    #TODO: check that shape is what is expected
    HALO = 2
    IJ0  = HALO
    IJn  = field.shape[0] - HALO - 1
    NIJ  = (IJn-IJ0+1)
    NK   = field.shape[2]
    NGRIDS = field.shape[3]
    NP_G = 0
    NP_I = IJ0
    NP_J = IJn+1
    SP_G = 1
    SP_I = IJn+1
    SP_J = IJ0
    SIZEXY  = NIJ*NIJ
    SIZEXYP = NIJ*NIJ+2
    f2 = numpy.resize(numpy.array([field[0,0,0,0]],order='FORTRAN'),jim_flatten_shape(field,nkfirst))
    np_val = field[NP_I,NP_J,:,NP_G]
    sp_val = field[SP_I,SP_J,:,SP_G]
    ijg0 = 0
    ijgn = 0
    for g in range(NGRIDS):
        ijg0 = 2+g*SIZEXY
        ijgn = ijg0+SIZEXY
        for k in range(NK):
            if nkfirst:
                if g==NP_G:
                    f2[k,0] = np_val[k]
                if g==SP_G:
                    f2[k,1] = sp_val[k]
                f2[k,ijg0:ijgn] = field[IJ0:IJn+1,IJ0:IJn+1,k,g].reshape(1,-1)
            else:
                if g==NP_G:
                    f2[0,k] = np_val[k]
                if g==SP_G:
                    f2[1,k] = sp_val[k]
                f2[ijg0:ijgn,k] = field[IJ0:IJn+1,IJ0:IJn+1,k,g].reshape(-1)
    return f2


def jim_unflatten(field):
    """Unpack JIM data to a stack of 10 icosahedral grids with halo
    Inverse operation of jim_flatten()

    jim_3d_field = jim_unflatten(jim_flat_field)

    @param jim_flat_field (numpy.ndarray)
    @return jim_3d_field (numpy.ndarray)

    Return a new numpy.ndarray (not a reference to the original one)
    """
    #try to guess if nkfirst
    nijglist = (12,42,162,642,2562,10242,40962,163842,655362,2621442,10485762,41943042)
    nkfirst=True
    nijg = field.shape[1]
    nk   = field.shape[0]
    if (field.shape[0] in nijglist) or field.shape[0]>nijglist[-1]:
        nkfirst=False
        nijg = field.shape[0]
        nk   = field.shape[1]
    NGRIDS = 10
    HALO = 2
    nij  = int(math.sqrt((nijg-2)/NGRIDS))
    nijh = nij + 2*HALO
    IJ0  = HALO
    IJn  = nijh - HALO - 1
    NP_G = 0
    NP_I = IJ0
    NP_J = IJn+1
    SP_G = 1
    SP_I = IJn+1
    SP_J = IJ0
    SIZEXY  = nij*nij
    SIZEXYP = nij*nij+2
    f2 = numpy.resize(numpy.array([field[0,0]],order='FORTRAN'),(nijh,nijh,nk,NGRIDS))
    f2[...] = 0.
    ijg0 = 0
    ijgn = 0
    for g in range(NGRIDS):
        ijg0 = 2+g*SIZEXY
        ijgn = ijg0+SIZEXY
        for k in range(nk):
            if nkfirst:
                if g==NP_G:
                    f2[NP_I,NP_J,k,NP_G] = field[k,0]
                if g==SP_G:
                    f2[SP_I,SP_J,k,SP_G] = field[k,1]
                #f3 = f2[IJ0:IJn+1,IJ0:IJn+1,k,g].reshape(-1)
                #f3[...] = field[k,ijg0:ijgn]
                f2[IJ0:IJn+1,IJ0:IJn+1,k,g] = field[k,ijg0:ijgn].reshape(nij,nij)
            else:
                if g==NP_G:
                    f2[NP_I,NP_J,k,NP_G] = field[0,k]
                if g==SP_G:
                    f2[SP_I,SP_J,k,SP_G] = field[1,k]
                #f3 = f2[IJ0:IJn+1,IJ0:IJn+1,k,g].reshape(-1)
                #f3[...] = field[ijg0:ijgn,k]
                f2[IJ0:IJn+1,IJ0:IJn+1,k,g] = field[ijg0:ijgn,k].reshape(nij,nij)

    return f2


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    #(la,lo) = jimc.jimc_grid_la_lo(1)
    #la2 = jim_flatten(la,True)
    #print la2.shape
    #print la2.flatten()
    #la3 = jim_unflatten(la2)
    #print la3.shape
    #print la.shape
    #for g in range(10):
        ##print la3[:,:,0,g].transpose()
        #print la3[2:4,2:4,0,g] == la[2:4,2:4,0,g]
        #print la3[2:5,2:5,0,g].transpose()
