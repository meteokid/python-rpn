ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnpy/include/Makefile.local.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

ifeq (,$(wildcard $(rpnpy)/VERSION))
   $(error Not found: $(rpnpy)/VERSION)
endif
RPNPY_VERSION0  = $(shell cat $(rpnpy)/VERSION)
RPNPY_VERSION   = $(notdir $(RPNPY_VERSION0))
RPNPY_VERSION_X = $(dir $(RPNPY_VERSION0))

## Some Shortcut/Alias to Lib Names

ifeq (,$(RMN_VERSION_FULL))
   $(error RMN_VERSION_FULL not defined; export RMN_VERSION_FULL=_015.2)
endif

RMN_VERSION    = rmn$(RMN_VERSION_FULL)
LIBRMN         = $(RMN_VERSION)
LIBRMNSHARED   = rmnshared$(RMN_VERSION_FULL)

COMM_VERSION   = _40511b
LIBCOMM        = rpn_comm$(COMM_VERSION) $(COMM_stubs)
LIBCOMM_STUBS  = rpn_commstubs$(COMM_VERSION)

LIBHPCSPERF    = hpcsperf

ifneq (,$(VGRIDDESCRIPTORS_VERSION))
VGRID_VERSION  = _$(VGRIDDESCRIPTORS_VERSION)
endif
LIBVGRID       = descrip$(VGRID_VERSION)
RPNPY_LIBS_DEP = $(LIBVGRID) $(LIBRMN) $(LIBCOMM) $(LIBHPCSPERF) $(LIBMASS) $(LAPACK) $(BLAS) $(RTOOLS) $(BINDCPU) $(LLAPI) $(IBM_LD) $(LIBHPC) $(LIBPMAPI)

# RPNPY_LIBS_MERGED = rpnpy_main rpnpy_driver rpnpy_utils rpnpy_tdpack rpnpy_base
# ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
# RPNPY_LIBS_OTHER  =  rpnpy_stubs rpnpy_massvp7_wrap
# else
# RPNPY_LIBS_OTHER  =  rpnpy_stubs
# endif
# RPNPY_LIBS_ALL    = $(RPNPY_LIBS_MERGED) $(RPNPY_LIBS_OTHER)
# RPNPY_LIBS        = rpnpy $(RPNPY_LIBS_OTHER) 
# RPNPY_LIBS_V      = rpnpy_$(RPNPY_VERSION) $(RPNPY_LIBS_OTHER) 

# RPNPY_LIBS_ALL_FILES = $(foreach item,$(RPNPY_LIBS_ALL),$(LIBDIR)/lib$(item).a)
# RPNPY_LIBS_ALL_FILES_PLUS = $(LIBDIR)/librpnpy.a $(RPNPY_LIBS_ALL_FILES) 

# OBJECTS_MERGED_rpnpy = $(foreach item,$(RPNPY_LIBS_MERGED),$(OBJECTS_$(item)))

# RPNPY_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_rpnpy),$(item).[Mm][Oo][Dd])

# RPNPY_ABS        = yyencode yydecode yy2global time2sec_main
# RPNPY_ABS_FILES  = $(foreach item,$(RPNPY_ABS),$(BINDIR)/$(item).Abs )

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(RPNPY_LIBS_V)


##
.PHONY: rpnpy_vfiles rpnpy_version.inc rpnpy_version.h rpnpy_version.py
RPNPY_VFILES = rpnpy_version.inc rpnpy_version.h rpnpy_version.py
rpnpy_vfiles: $(RPNPY_VFILES)
rpnpy_version.inc:
	.rdemkversionfile "rpnpy" "$(RPNPY_VERSION)" $(ROOT)/include f
rpnpy_version.h:
	.rdemkversionfile "rpnpy" "$(RPNPY_VERSION)" $(ROOT)/include c
LASTUPDATE = $(shell date '+%Y-%m-%d %H:%M %Z')
rpnpy_version.py:
	echo "__VERSION__ = '$(RPNPY_VERSION)'" > $(ROOT)/lib/$@
	echo "__LASTUPDATE__ = '$(LASTUPDATE)'" >> $(ROOT)/lib/$@


#---- ARCH Specific overrides -----------------------------------------
ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
LIBMASSWRAP = rpnpy_massvp7_wrap
endif

#---- Abs targets -----------------------------------------------------


#---- Lib target - automated ------------------------------------------


ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnpy/include/Makefile.local.mk [END] ==================)
endif
