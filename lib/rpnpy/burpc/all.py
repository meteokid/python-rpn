#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Short hand to load all rpnpy.burpc submodules in the same namespace

See also:
     rpnpy.burpc
     rpnpy.burpc.proto
     rpnpy.burpc.const
     rpnpy.burpc.base
     rpnpy.burpc.brpobj

"""

from . import *
from .const  import *
from .proto  import *
from .base   import *
from .brpobj import *
