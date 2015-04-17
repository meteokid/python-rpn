#!/usr/bin/env python
"""
Short hand to load all librmn submodules in the same namespace

 See also:
 help(librmn)
 help(librmn.proto)
 help(librmn.const)
 help(librmn.base)
 help(librmn.fstd98)
 help(librmn.interp)

"""

from . import *
from .proto import *
from .const import *
from .base import *
from .fstd98 import *
from .interp import *
from .grids import *
