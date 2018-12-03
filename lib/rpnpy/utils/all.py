#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Short hand to load all rpnpy.utils submodules in the same namespace

 See also:
     rpnpy.utils
     rpnpy.utils.fstd3d
     rpnpy.utils.burpfile
     rpnpy.utils.tdpack_consts
     rpnpy.utils.tdpack
     rpnpy.utils.llacar
     rpnpy.utils.fstd_extras

"""

from . import *
from .fstd3d import *
from .burpfile import *
from .tdpack_consts import *
from .tdpack import *
from .llacar import *
from .fstd_extras import stamp2datetime, decode_ip1, all_params, maybeFST
