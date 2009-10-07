"""Module scrip contains the classes used to work with the SCRIP package

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpn_version
from scripc import *
import numpy

class ScripGrid:
    """Defines a grid suitable for SCRIP

    myScripGrid = ScripGrid('myName',(lat,lon,c_lat,c_lon))
    myScripGrid = ScripGrid('myName',shape=(nc,ni,nj))
    @param myName
    @param lat grid points center lat (numpy.ndarray(ni,nj)) (rad)
    @param lon grid points center lon (numpy.ndarray(ni,nj)) (rad)
    @param c_lat grid points corners lat (numpy.ndarray(nc,ni,nj)) (rad)
    @param c_lon grid points corners lon (numpy.ndarray(nc,ni,nj)) (rad)
    @param (nc,ni,nj) number of grid corners, grid dims
    @exception TypeError on wrong args type
    @exception TypeError on dimension mismatch
    """
    name  = 'scripGridNoName'
    lalo  = None
    shape = ()
    size  = 0

    def __init__(self,name,lalo=None,shape=None):
        self.name  = name
        if type(shape)==type(()):
            self.shape = shape
            self.size  = sum(shape)
        if lalo:
            if (type(lalo)==type(()) and len(lalo)==4
                and type(lalo[0])==numpy.ndarray
                and type(lalo[1])==numpy.ndarray
                and type(lalo[2])==numpy.ndarray
                and type(lalo[3])==numpy.ndarray):
                if (lalo[0].shape==lalo[1].shape and
                    lalo[2].shape==lalo[3].shape and
                    lalo[0].shape==lalo[2].shape[1:]):
                    self.lalo = lalo
                    self.shape = self.lalo[3].shape
                    self.size  = self.lalo[3].size
                else:
                    raise TypeError, "scripGrid(name,lalo): dimensions mismatch for lalo"
            else:
                raise TypeError, "scripGrid(name,lalo): wrong type for lalo, should be (center_lat, center_lon, corners_lat, corners_lon) of type (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)"


class Scrip:
    """SCRIP inerpolation package base class

    a = Scrip(inGrid,outGrid,addrWeights)
    a = Scrip(inGrid,outGrid)
    @param inGrid  source grid definition (ScripGrid)
    @param outGrid dest grid definition (ScripGrid)
    @param addrWeights Precomputed remapping params tuple of 3 numpy.ndarray as: (fromAddr,toAddr,weights), if not provided will be either read from disk or computed if not found
    @exception TypeError
    @exception ValueError
    """
    grids   = None
    weights = None
    #TODO: nbins,methode,typ_norm,typ_restric

    def __init__(self,inGrid,outGrid,weights=None):
        #TODO: add intepolation method options
        if isinstance(inGrid,scripGrid) and isinstance(outGrid,scripGrid):
            grids = (inGrid,outGrid)
        else:
            raise TypeError, "scripInterp(inGrid,outGrid,weights): wrong type for inGrid,outGrid; should be of type scripGrid"
        if weights:
            if (type(weights)==type(()) and
                len(weights)==3 and
                type(weights[0])==numpy.ndarray and
                type(weights[1])==numpy.ndarray and
                type(weights[2])==numpy.ndarray):
                if weights[0].shape == weights[1].shape ==  weights[2].shape:
                    self.weights = weights
                    #TODO: if grids have lalo, check dims mismatch (not sure we can do that w/o check toAddr and fromAddr)
                else:
                    raise TypeError, "scripInterp(inGrid,outGrid,weights): dimensions mismatch for weights"
            else:
                raise TypeError, "scripInterp(inGrid,outGrid,weights): wrong type for weights; should be (fromAddr,toAddr,weights) of type (numpy.ndarray, numpy.ndarray)"
        else:
            if file.exists('scripadrwts'+grids[0].name+grids[1].name):
                pass
                #TODO: try to read weights from file numpy.fromfile()
                #TODO: check file validity
            elif girds[0].lalo and grids[1].lalo:
                #TODO: convert latlon to radiants with proper range
                weights = scripc_addr_wts(
                    grids[0].lalo[0],grids[0].lalo[1],
                    grids[0].lalo[2],grids[0].lalo[3],
                    grids[1].lalo[0],grids[1].lalo[1],
                    grids[1].lalo[2],grids[1].lalo[3],
                    nbins,methode,typ_norm,typ_restric)
            else:
                raise ValueError, "scripInterp(inGrid,outGrid,weights): missing values, unable to compute weights since centers and corners lat/lon are not provided"
        #TODO: save weights on disk numpy.ndarray.tofile()

    def interp(field):
        """Perform interpolation"""
        #TODO: may want to check that field is on grids[0] (dims)
        return scripc_interp_o1(field,self.weights[0],self.weights[1],self.weights[2],grids[1].size)

    def inverse(self):
        """Reverse direction of intepolation if possible
        Either weights are already persent on disk
        Or center and corners lat lon were provided
        """
        #TODO: add interp options
        self.__init__(self.grids[1],self.grids[0])

    def __del__():
        """ """
        scripc_addr_wts_free(self.weights[0],self.weights[1],self.weights[2])

    def __repr__(self):
        pass #TODO


if __name__ == "__main__":
    import doctest
    doctest.testmod()
