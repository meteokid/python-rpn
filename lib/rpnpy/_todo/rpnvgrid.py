 
"""Module RPNVGrid contains the classes used to manipulate RPN style Veritcal Grid definition

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpn_version
#from rpn_helpers import *
from rpnbasedict import *

import numpy

class RPNVGgrid(RPNBaseDict):
    """RPN Vertical Grid container class.
    Hold Vertical Grid definiton and methods to interact with it

    Examples of use (also doctests):
    #TODO:
    """
    def _getDefaultKeyVal(self):
        return {
            'd'     : {'v':None,'t':numpy.ndarray},
            'name'  : {'v':None,'t':type('')},
            'type'  : {'v':None,'t':type('')},
            'etiket': {'v':None,'t':type('')},
            'vgrid' : {'v':None,'t':type(RPNVGgrid)},
            'hgrid' : {'v':None,'t':type(RPNHGgrid)},
            'date'  : {'v':None,'t':type('')}
            #'date'  : {'v':None,'t':type(RPNDate)}
            }

    def __init__(self,other=None):
        RPNBaseDict.__init__(self,other)


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
