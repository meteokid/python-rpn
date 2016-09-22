#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a4/intel13sp1u2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module vgd is a ctypes import of vgrid's library (libdescrip.so)
 
The vgd.proto python module includes ctypes prototypes for many
vgrid's libdescrip C functions.

The functions in this module are actual C funtions and must thus be called
as such with appropriate argument typing and dereferencing.
It is highly advised in a python program to prefer the use of the python wrapper
found in rpnpy.vgd.base

See Also:
    rpnpy.vgd.base
    rpnpy.vgd.const

Notes:
    This Module is available from python-rpn version 2.0.b6

 === Functions C Prototypes ===

 c_vgd_construct():
    Returns a NOT fully initialized VGridDescriptor instance
    Proto:
       vgrid_descriptor* c_vgd_construct();
    Args:
       None
    Returns:
       POINTER(VGridDescriptor) : a pointer to a new VGridDescriptor object

 c_vgd_new_read(self, unit, ip1, ip2, kind, version):
    Construct a vgrid descriptor from !! in RPN standard file
    Proto:
       int Cvgd_new_read(vgrid_descriptor **self, int unit, int ip1, int ip2,
                         int kind, int version);
    Args:
       self (POINTER(POINTER(VGridDescriptor))):
               A VGridDescriptor obj to be filled with read vgrid values (I/O)
               This is obtained with c_vgd_construct
       unit     (int) : Openend RPN Std file unit number (I)
       ip1      (int) : Ip1 of the vgrid record to find, use -1 for any (I)
       ip2      (int) : Ip2 of the vgrid record to find, use -1 for any (I)
       kind     (int) : vgrid kind (I)
       version  (int) : vgrid version (I)
    Returns:
       int : Status VGD_OK or VGD_ERROR
    See Also:
       c_vgd_construct
       c_vgd_free

 c_vgd_new_gen(self, kind, version, hyb, rcoef1, rcoef2, ptop_8, pref_8,
               ptop_out_8, ip1, ip2, dhm, dht):
    Build a VGridDescriptor instance initialized with provided info 
    Proto:
       int Cvgd_new_gen(vgrid_descriptor **self, int kind, int version,
                        float *hyb, int size_hyb, float *rcoef1, float *rcoef2,
                        double *ptop_8, double *pref_8, double *ptop_out_8,
                        int ip1, int ip2, float *dhm, float *dht);
    Args:
       self (POINTER(POINTER(VGridDescriptor))):
               A VGridDescriptor obj to be filled provided vgrid values (I/O)
               This is obtained with c_vgd_construct
       kind     (int) : vgrid kind (I)
       version  (int) : vgrid version (I)
       hyb      (float array) : 
       size_hyb (int) : 
       rcoef1   (float ptr) :
       rcoef2   (float ptr) :
       ptop_8   (double ptr) :
       pref_8   (double ptr) :
       ptop_out_8 (double ptr) :
       ip1      (int) : 
       ip2      (int) : 
       dhm      (float ptr) :
       dht      (float ptr) :
    Returns:
       int : Status VGD_OK or VGD_ERROR
    See Also:
       c_vgd_construct
       c_vgd_free

 c_vgd_new_build_vert(self, kind, version, nk, ip1, ip2, ptop_8, pref_8,
                      rcoef1, rcoef2, a_m_8, b_m_8, a_t_8, b_t_8, ip1_m, ip1_t,
                      nl_m, nl_t):
    Build a vgrid descriptor from the building blocks e.g. list of A, B, ip1 rcoef etc
    Proto:
       int Cvgd_new_build_vert(vgrid_descriptor **self, int kind, int version,
                               int nk, int ip1, int ip2, double *ptop_8,
                               double *pref_8, float *rcoef1, float *rcoef2, 
                               double *a_m_8, double *b_m_8, double *a_t_8,
                               double *b_t_8, int *ip1_m, int *ip1_t,
                               int nl_m, int nl_t);
    Args:
       self (POINTER(POINTER(VGridDescriptor))):
               A VGridDescriptor obj to be filled with provided vgrid values (I/O)
               This is obtained with c_vgd_construct
       ...
    Returns:
       int : Status VGD_OK or VGD_ERROR
    See Also:
       c_vgd_construct
       c_vgd_free

 c_vgd_new_from_table(self, table, ni, nj, nk):
    Build a vgrid descriptor from the a VGridDescriptor stored as a table/array
    Proto:
       int Cvgd_new_from_table(vgrid_descriptor **self, double *table,
                               int ni, int nj, int nk);
    Args:
       self (POINTER(POINTER(VGridDescriptor))):
               A VGridDescriptor obj to be filled with provided vgrid values (I/O)
               This is obtained with c_vgd_construct
       table   (POINTER(c_double)) : 
       ni      (int) :
       nj      (int) :
       nk      (int) :
    Returns:
       int : Status VGD_OK or VGD_ERROR
    See Also:
       c_vgd_construct
       c_vgd_free
       c_vgd_get_double_3d
 
 c_vgd_write_desc(self, unit):
    Write vgrid descriptor in a previously opened RPN standard file
    Proto:
       int Cvgd_write_desc(vgrid_descriptor *self, int unit);
    Args:
       self (POINTER(VGridDescriptor)): A VGridDescriptor obj (I)
       unit     (int) : Openend RPN Std file unit number (I)
    Returns:
       int : Status VGD_OK or VGD_ERROR
    See Also:
       c_vgd_construct
       c_vgd_new_read
       c_vgd_new_gen
       c_vgd_new_build_vert
       c_vgd_new_from_table

 c_vgd_free(self):
    Free memory from previously created vgrid descriptor
    Proto:
       void Cvgd_free(vgrid_descriptor **self);
    Args:
       self (POINTER(POINTER(VGridDescriptor))): A VGridDescriptor obj (I/O)
    Returns:
       None
    See Also:
       c_vgd_construct
       c_vgd_new_read
       c_vgd_new_gen
       c_vgd_new_build_vert
       c_vgd_new_from_table
 
 c_vgd_vgdcmp(vgd1, vgd2):
    Test if two vgrid descriptors are equal,
    Returns 0 if they are the same like String function strcmp
    Proto:
       int Cvgd_vgdcmp(vgrid_descriptor *vgd1, vgrid_descriptor *vgd2);
    Args:
       vgd1 (VGridDescriptor ref) : (I) 
       vgd2 (VGridDescriptor ref) : (I) 
    Returns:
       int, 0 if the 2 VGridDescriptor are the same

 c_vgd_levels(self, ni, nj, nk, ip1_list, levels, sfc_field, in_log):
    Compute level positions (pressure) for the given ip1 list and surface field
    Proto:
       int Cvgd_levels(vgrid_descriptor *self, int ni, int nj, int nk,
                       int *ip1_list, float *levels, float *sfc_field,
                       int in_log);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR

 c_vgd_levels_8(self, ni, nj, nk, ip1_list, levels_8, sfc_field_8, in_log):
    Compute level positions (pressure) for the given ip1 list and surface field
    Proto:
       int Cvgd_levels_8(vgrid_descriptor *self, int ni, int nj, int nk,
                         int *ip1_list, double *levels_8, double *sfc_field_8,
                         int in_log);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR

 c_vgd_diag_withref(self, ni, nj, nk, ip1_list, levels, sfc_field, in_log, dpidpis):
    Compute level positions (pressure) for the given ip1 list and surface field
    Proto:
       int Cvgd_diag_withref(vgrid_descriptor *self, int ni, int nj, int nk,
                       int *ip1_list, float *levels, float *sfc_field,
                       int in_log, int dpidpis);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR

 c_vgd_diag_withref_8(self, ni, nj, nk, ip1_list, levels_8, sfc_field_8, in_log, dpidpis):
    Compute level positions (pressure) for the given ip1 list and surface field
    Proto:
       int Cvgd_diag_withref_8(vgrid_descriptor *self, int ni, int nj, int nk,
                         int *ip1_list, double *levels_8, double *sfc_field_8,
                         int in_log, int dpidpis);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_int(self, key, value, quiet):
    Get scalar integer attribute of vgrid descriptor
    Proto:
       int Cvgd_get_int(vgrid_descriptor *self, char *key,
                        int *value, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_getopt_int(key, value, quiet):
    Get scalar integer global VGD option 
    Proto:
       int Cvgd_getopt_int(char *key, int *value, int quiet);
    Args:
       key
       value
       quiet
    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_int_1d(self, key, value, nk, quiet):
    Get vector integer attribute of vgrid descriptor
    Proto:
       int Cvgd_get_int_1d(vgrid_descriptor *self, char *key,
                           int **value, int *nk, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_float(self, key, value, quiet):
    Get scalar float attribute of vgrid descriptor
    Proto:
       int Cvgd_get_float(vgrid_descriptor *self, char *key,
                          float *value, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_float_1d(self, key, value, nk, quiet):
    Get vector float attribute of vgrid descriptor
    Proto:
       int Cvgd_get_float_1d(vgrid_descriptor *self, char *key,
                             float **value, int *nk, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_double(self, key, value_get, quiet):
    Get scalar double attribute of vgrid descriptor
    Proto:
       int Cvgd_get_double(vgrid_descriptor *self, char *key,
                           double *value_get, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_double_1d(self, key, value, nk, quiet):
    Get vector double attribute of vgrid descriptor
    Proto:
       int Cvgd_get_double_1d(vgrid_descriptor *self, char *key,
                              double **value, int *nk, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_get_double_3d(self, key, value, ni, nj, nk, quiet):
    Get array double attribute of vgrid descriptor
    Proto:
       int Cvgd_get_double_3d(vgrid_descriptor *self, char *key, double **value,
                              int *ni, int *nj, int *nk, int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR

 c_vgd_get_char(self, key, out, quiet):
    Get character attribute of vgrid descriptor
    Proto:
       int Cvgd_get_char(vgrid_descriptor *self, char *key,
                         char out[], int quiet);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_put_char(self, key, value):
    Set scalar char attribute of vgrid descriptor
    Proto:
       int Cvgd_put_char(vgrid_descriptor **self, char *key, char *value);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_put_int(self, key, value):
    Set scalar int attribute of vgrid descriptor
    Proto:
       int Cvgd_put_int(vgrid_descriptor **self, char *key, int value);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_putopt_int(key, value):
    Set scalar integer global VGD option 
    Proto:
       int Cvgd_putopt_int(char *key, int value);
    Args:
       key
       value
       quiet
    Returns:
       int : Status VGD_OK or VGD_ERROR 

 c_vgd_put_double(self, key, value_put):
    Set scalar double attribute of vgrid descriptor   
    Proto:
       int Cvgd_put_double(vgrid_descriptor **self, char *key, double value_put);
    Args:
       self (VGridDescriptor ref) : (I/O)

    Returns:
       int : Status VGD_OK or VGD_ERROR 

"""
 ## c_vgd_print_desc(self, sout, convip):
 ##    Print informations on vgrid descriptor
 ##    Proto:
 ##       int Cvgd_print_desc(vgrid_descriptor *self, int sout, int convip);
 ##    Args:
 ##       self (VGridDescriptor ref) : (I/O)

 ##    Returns:
 ##       int, 

 ## c_vgd_print_vcode_description(vcode):
 ##    Print the description of a Vcode e.g. 5005
 ##    Proto:
 ##       int Cvgd_print_vcode_description(int vcode);
 ##    Args:

 ##    Returns:
 ##       int, 

 ## c_vgd_set_vcode_i(vgrid, kind, version):
 ##    ?
 ##    Proto:
 ##       int Cvgd_set_vcode_i(vgrid_descriptor *VGrid, int Kind, int Version);
 ##    Args:
 ##       vgrid (VGridDescriptor ref) : (I/O)

 ##    Returns:
 ##       int, 

 ## c_vgd_set_vcode(vgrid):
 ##    ?
 ##    Proto:
 ##       int Cvgd_set_vcode(vgrid_descriptor *VGrid);
 ##    Args:
 ##       vgrid (VGridDescriptor ref) : (I/O)

 ##    Returns:
 ##       int, 


import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc

from . import libvgd
from . import const as _cst


class VGridDescriptor(_ct.Structure):
    """
    Python Class equivalenet of the vgrid's C structure to hold the vgrid data

    This class has a private internal representation.
    You should only hold a pointer to it.
    Getting and setting values should be done throught
    vgd_get and vgd_put functions.

    To get an instance of a pointer to this class you may use the
    provided functions:

    myVGDptr = rpnpy.vgd.propo.c_vgd_construct()

    See Also:
       c_vgd_construct
    """
    _fields_ = [
        ("dummy", _ct.c_int),
        ]
    
    def __str__(self):
       return self.__class__.__name__ + str([x[0] + '=' + str(self.__getattribute__(x[0])) for x in self._fields_])
       ## s = self.__class__.__name__ + '('
       ## l = [y[0] for y in self._fields_]
       ## l.sort()
       ## for x in l:
       ##     s += x + '=' + str(self.__getattribute__(x)) + ', '
       ## s += ')'
       ## return s

    def __repr__(self):
       #return self.__class__.__name__ + str(self)
       return self.__class__.__name__ + repr([x[0] + '=' + repr(self.__getattribute__(x[0])) for x in self._fields_])

c_vgd_construct = _ct.POINTER(VGridDescriptor)

## int Cvgd_new_read(vgrid_descriptor **self, int unit, int ip1, int ip2,
##                   int kind, int version);
libvgd.Cvgd_new_read.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)), # vgrid_descriptor **self
    _ct.c_int, #int unit
    _ct.c_int, #int ip1
    _ct.c_int, #int ip2
    _ct.c_int, #int kind
    _ct.c_int) #int version
libvgd.Cvgd_new_read.restype = _ct.c_int
c_vgd_new_read = libvgd.Cvgd_new_read


## int Cvgd_new_gen(vgrid_descriptor **self, int kind, int version,
##                  float *hyb, int size_hyb, float *rcoef1, float *rcoef2,
##                  double *ptop_8, double *pref_8, double *ptop_out_8,
##                  int ip1, int ip2, float *dhm, float *dht);
libvgd.Cvgd_new_gen.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)), # vgrid_descriptor **self
    _ct.c_int, #int kind
    _ct.c_int, #int version
    _npc.ndpointer(dtype=_np.float32), #float *hyb
    _ct.c_int, #int size_hyb,
    _ct.POINTER(_ct.c_float), #float *rcoef1,
    _ct.POINTER(_ct.c_float), #float *rcoef2,
    _ct.POINTER(_ct.c_double), #double *ptop_8,
    _ct.POINTER(_ct.c_double), #double *pref_8,
    _ct.POINTER(_ct.c_double), #double *ptop_out_8
    _ct.c_int, #int ip1
    _ct.c_int, #int ip2
    _ct.POINTER(_ct.c_float), #float *dhm
    _ct.POINTER(_ct.c_float)) #float *dht
libvgd.Cvgd_new_gen.restype = _ct.c_int
c_vgd_new_gen = libvgd.Cvgd_new_gen


## int Cvgd_new_build_vert(vgrid_descriptor **self, int kind, int version,
##                         int nk, int ip1, int ip2, double *ptop_8,
##                         double *pref_8, float *rcoef1, float *rcoef2, 
##                         double *a_m_8, double *b_m_8, double *a_t_8,
##                         double *b_t_8, int *ip1_m, int *ip1_t,
##                         int nl_m, int nl_t);
libvgd.Cvgd_new_build_vert.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)), # vgrid_descriptor **self
    _ct.c_int, #int kind
    _ct.c_int, #int version
    _ct.c_int, #int nk,
    _ct.c_int, #int ip1
    _ct.c_int, #int ip2
    _ct.POINTER(_ct.c_double), #double *ptop_8,
    _ct.POINTER(_ct.c_double), #double *pref_8,
    _ct.POINTER(_ct.c_float), #float *rcoef1,
    _ct.POINTER(_ct.c_float), #float *rcoef2,
     _npc.ndpointer(dtype=_np.float64), #a_m_8
     _npc.ndpointer(dtype=_np.float64), #b_m_8
     _npc.ndpointer(dtype=_np.float64), #a_t_8
     _npc.ndpointer(dtype=_np.float64), #b_t_8
     _npc.ndpointer(dtype=_np.int32), #a_t_8
     _npc.ndpointer(dtype=_np.int32), #b_t_8
    _ct.c_int, #int nl_m
    _ct.c_int) #int nl_t

libvgd.Cvgd_new_build_vert.restype = _ct.c_int
c_vgd_new_build_vert = libvgd.Cvgd_new_build_vert

## int Cvgd_new_from_table(vgrid_descriptor **self, double *table,
##                         int ni, int nj, int nk);
libvgd.Cvgd_new_from_table.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)), # vgrid_descriptor **self
    _ct.POINTER(_ct.c_double), #double *table,
    _ct.c_int, #int ni
    _ct.c_int, #int nj
    _ct.c_int) #int nk
libvgd.Cvgd_new_from_table.restype = _ct.c_int
c_vgd_new_from_table = libvgd.Cvgd_new_from_table

## int Cvgd_write_desc(vgrid_descriptor *self, int unit);
libvgd.Cvgd_write_desc.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_int
    )
libvgd.Cvgd_write_desc.restype = _ct.c_int
c_vgd_write_desc = libvgd.Cvgd_write_desc

## void Cvgd_free(vgrid_descriptor **self);
libvgd.Cvgd_free.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)),
    )
c_vgd_free = libvgd.Cvgd_free

## int Cvgd_vgdcmp(vgrid_descriptor *vgd1, vgrid_descriptor *vgd2);
libvgd.Cvgd_vgdcmp.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.POINTER(VGridDescriptor))
libvgd.Cvgd_vgdcmp.restype = _ct.c_int
c_vgd_vgdcmp = libvgd.Cvgd_vgdcmp

## int Cvgd_levels(vgrid_descriptor *self, int ni, int nj, int nk,
##                 int *ip1_list, float *levels, float *sfc_field,
##                 int in_log);
libvgd.Cvgd_levels.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
libvgd.Cvgd_levels.restype = _ct.c_int
c_vgd_levels = libvgd.Cvgd_levels

## int Cvgd_levels_8(vgrid_descriptor *self, int ni, int nj, int nk,
##                   int *ip1_list, double *levels_8, double *sfc_field_8,
##                   int in_log);
libvgd.Cvgd_levels_8.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int),
    _npc.ndpointer(dtype=_np.float64),
    _npc.ndpointer(dtype=_np.float64),
    _ct.c_int
    )
libvgd.Cvgd_levels_8.restype = _ct.c_int
c_vgd_levels_8 = libvgd.Cvgd_levels_8


## int Cvgd_diag_withref(vgrid_descriptor *self, int ni, int nj, int nk,
##                       int *ip1_list, float *levels, float *sfc_field,
##                       int in_log, int dpidpis)
libvgd.Cvgd_diag_withref.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
libvgd.Cvgd_diag_withref.restype = _ct.c_int
c_vgd_diag_withref = libvgd.Cvgd_diag_withref

## int Cvgd_diag_withref_8(vgrid_descriptor *self, int ni, int nj, int nk,
##                         int *ip1_list, double *levels_8,
##                         double *sfc_field_8, int in_log, int dpidpis)
libvgd.Cvgd_diag_withref_8.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int),
    _npc.ndpointer(dtype=_np.float64),
    _npc.ndpointer(dtype=_np.float64),
    _ct.c_int
    )
libvgd.Cvgd_diag_withref_8.restype = _ct.c_int
c_vgd_diag_withref_8 = libvgd.Cvgd_diag_withref_8


libvgd.Cvgd_get_char.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.c_char_p,
    _ct.c_int)
libvgd.Cvgd_get_char.restype = _ct.c_int
c_vgd_get_char = libvgd.Cvgd_get_char


libvgd.Cvgd_get_int.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.c_int),
    _ct.c_int)
libvgd.Cvgd_get_int.restype = _ct.c_int
c_vgd_get_int = libvgd.Cvgd_get_int


libvgd.Cvgd_getopt_int.argtypes = (
    _ct.c_char_p,
    _ct.POINTER(_ct.c_int),
    _ct.c_int)
libvgd.Cvgd_getopt_int.restype = _ct.c_int
c_vgd_getopt_int = libvgd.Cvgd_getopt_int


libvgd.Cvgd_get_float.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.c_float),
    _ct.c_int)
libvgd.Cvgd_get_float.restype = _ct.c_int
c_vgd_get_float = libvgd.Cvgd_get_float


libvgd.Cvgd_get_double.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.c_double),
    _ct.c_int)
libvgd.Cvgd_get_double.restype = _ct.c_int
c_vgd_get_double = libvgd.Cvgd_get_double


libvgd.Cvgd_get_int_1d.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.POINTER(_ct.c_int)),
    _ct.POINTER(_ct.c_int),
    _ct.c_int)
libvgd.Cvgd_get_int_1d.restype = _ct.c_int
c_vgd_get_int_1d = libvgd.Cvgd_get_int_1d


libvgd.Cvgd_get_float_1d.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.POINTER(_ct.c_float)),
    _ct.POINTER(_ct.c_int),
    _ct.c_int)
libvgd.Cvgd_get_float_1d.restype = _ct.c_int
c_vgd_get_float_1d = libvgd.Cvgd_get_float_1d


libvgd.Cvgd_get_double_1d.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.POINTER(_ct.c_double)),
    _ct.POINTER(_ct.c_int),
    _ct.c_int)
libvgd.Cvgd_get_double_1d.restype = _ct.c_int
c_vgd_get_double_1d = libvgd.Cvgd_get_double_1d


libvgd.Cvgd_get_double_3d.argtypes = (
    _ct.POINTER(VGridDescriptor),
    _ct.c_char_p,
    _ct.POINTER(_ct.POINTER(_ct.c_double)),
    _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int),
    _ct.c_int)
libvgd.Cvgd_get_double_3d.restype = _ct.c_int
c_vgd_get_double_3d = libvgd.Cvgd_get_double_3d


libvgd.Cvgd_put_char.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)),
    _ct.c_char_p,
    _ct.c_char_p)
libvgd.Cvgd_put_char.restype = _ct.c_int
c_vgd_put_char = libvgd.Cvgd_put_char


libvgd.Cvgd_put_int.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)),
    _ct.c_char_p,
    _ct.c_int)
libvgd.Cvgd_put_int.restype = _ct.c_int
c_vgd_put_int = libvgd.Cvgd_put_int


libvgd.Cvgd_putopt_int.argtypes = (
    _ct.c_char_p,
    _ct.c_int)
libvgd.Cvgd_putopt_int.restype = _ct.c_int
c_vgd_putopt_int = libvgd.Cvgd_putopt_int


libvgd.Cvgd_put_double.argtypes = (
    _ct.POINTER(_ct.POINTER(VGridDescriptor)),
    _ct.c_char_p,
    _ct.c_double)
libvgd.Cvgd_put_double.restype = _ct.c_int
c_vgd_put_double = libvgd.Cvgd_put_double


## int Cvgd_print_desc(vgrid_descriptor *self, int sout, int convip);

## int Cvgd_print_vcode_description(int vcode);

## int Cvgd_set_vcode_i(vgrid_descriptor *VGrid, int Kind, int Version);

## int Cvgd_set_vcode(vgrid_descriptor *VGrid);

if __name__ == "__main__":
    pass #print vgd version 
    
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
