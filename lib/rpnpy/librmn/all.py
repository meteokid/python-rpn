#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
 Short hand to load all rpnpy.librmn submodules in the same namespace

 See also:
     rpnpy.librmn
     rpnpy.librmn.proto
     rpnpy.librmn.const
     rpnpy.librmn.base
     rpnpy.librmn.fstd98
     rpnpy.librmn.interp
     rpnpy.librmn.grids

"""

from . import *
from .proto import *
from .const import *
from .base import *
from .fstd98 import *
from .interp import *
from .grids import *
