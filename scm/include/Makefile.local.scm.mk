ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$scm/include/Makefile.local.scm.mk)
$(info ## )
endif

# ifeq (,$(wildcard $(scm)/VERSION))
#    $(error Not found: $(scm)/VERSION)
# endif
# SCM_VERSION   = $(shell cat $(scm)/VERSION)
# SCM_VERSION_X = x/
SCM_VERSION0  = x/2.0.rc4
SCM_VERSION   = $(notdir $(SCM_VERSION0))
SCM_VERSION_X = $(dir $(SCM_VERSION0))

SCM_SFX = $(RDE_LIBS_SFX)

## Some Shortcut/Alias to Lib Names

ifeq (,$(MODELUTILS_LIBS))
ifeq (,$(modelutils))
$(error ERROR: modelutils not set)
endif
TMPFILELIST = $(wildcard $(modelutils)/include/Makefile.local*.mk)
ifeq (,$(TMPFILELIST))
endif
$(info include $(TMPFILELIST))
include $(TMPFILELIST)
endif

ifeq (,$(RPNPHY_LIBS))
ifeq (,$(rpnphy))
$(error ERROR: rpnphy not set)
endif
TMPFILELIST = $(wildcard $(rpnphy)/include/Makefile.local*.mk)
ifeq (,$(TMPFILELIST))
endif
$(info include $(TMPFILELIST))
include $(TMPFILELIST)
endif

SCM_LIBS_DEP = $(RPNPHY_LIBS) $(RPNPHY_LIBS_DEP) $(NETCDFLIBS) $(SCMEXTRALIBS)

LIBCPLPATH = 
LIBCPL = 
LIBSCMUTILS = scm_utils

SCM_LIB_MERGED_NAME_0 = scm
SCM_LIBS_MERGED_0 = scm_main scm_base
SCM_LIBS_OTHER_0  = $(LIBCPL) $(LIBSCMUTILS)
SCM_LIBS_ALL_0    = $(SCM_LIBS_MERGED_0) $(SCM_LIBS_OTHER_0)
SCM_LIBS_0        = $(SCM_LIB_MERGED_NAME_0) $(SCM_LIBS_OTHER_0)

SCM_LIB_MERGED_NAME = $(foreach item,$(SCM_LIB_MERGED_NAME_0),$(item)$(SCM_SFX))
SCM_LIBS_MERGED = $(foreach item,$(SCM_LIBS_MERGED_0),$(item)$(SCM_SFX))
SCM_LIBS_OTHER  = $(LIBCPL) $(LIBSCMUTILS)$(SCM_SFX)
SCM_LIBS_ALL    = $(SCM_LIBS_MERGED) $(SCM_LIBS_OTHER)
SCM_LIBS        = $(SCM_LIB_MERGED_NAME) $(SCM_LIBS_OTHER)

SCM_LIB_MERGED_NAME_V = $(foreach item,$(SCM_LIB_MERGED_NAME),$(item)_$(SCM_VERSION))
SCM_LIBS_MERGED_V = $(foreach item,$(SCM_LIBS_MERGED),$(item)_$(SCM_VERSION))
SCM_LIBS_OTHER_V  = $(LIBCPL) $(LIBSCMUTILS)$(SCM_SFX)_$(SCM_VERSION)
SCM_LIBS_ALL_V    = $(SCM_LIBS_MERGED_V) $(SCM_LIBS_OTHER_V)
SCM_LIBS_V        = $(SCM_LIB_MERGED_NAME_V) $(SCM_LIBS_OTHER_V)

SCM_LIBS_ALL_FILES      = $(foreach item,$(SCM_LIBS_ALL),$(LIBDIR)/lib$(item).a)
SCM_LIBS_ALL_FILES_PLUS = $(LIBDIR)/lib$(SCM_LIB_MERGED_NAME).a $(SCM_LIBS_ALL_FILES)

OBJECTS_MERGED_scm = $(foreach item,$(SCM_LIBS_MERGED_0),$(OBJECTS_$(item)))

SCM_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_scm),$(item).[Mm][Oo][Dd])


SCM_STD_TARGETS = $(fst2ptxt) $(setfld) $(forcedate)
SCM_ABS       = $(SCM_STD_TARGETS) scm
SCM_ABS_FILES = $(foreach item,$(SCM_STD_TARGETS),$(BINDIR)/$(item)) $(BINDIR)/$(mainscm)

## SCM model Libpath and libs
MODEL3_LIBPATH = $(LIBCPLPATH)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(SCM_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(SCM_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   SCM_ABS_DEP = $(SCM_LIBS_ALL_FILES_PLUS) $(SCM_ABS_DEP)
endif

# export ATM_MODEL_BNDL     = $(SSM_RELDIRBNDL)$(SCM_VERSION)
# export ATM_MODEL_DSTP    := $(shell date '+%Y-%m-%d %H:%M %Z')
# export ATM_MODEL_VERSION  = $(SCM_VERSION)
# export ATM_MODEL_NAME    := SCM

##
.PHONY: scm_vfiles
SCM_VFILES = scm_version.inc scm_version.h
scm_vfiles: $(SCM_VFILES)
scm_version.inc:
	.rdemkversionfile "scm" "$(SCM_VERSION)" . f
scm_version.h:
	.rdemkversionfile "scm" "$(SCM_VERSION)" . c

#---- Architecture-specific elements ---------------------------------
NETCDFLIBS = netcdff netcdf
SCMEXTRALIBS = mpi
ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
   NETCDFLIBS = netcdf
   SCMEXTRALIBS = 
endif
#TODO: needed for eccc-ppp12 load... find less hacky way
# ifeq (ubuntu-14.04-amd64-64,$(ORDENV_PLAT))
ifneq (,$(wildcard /fs/ssm/*))
##    NETCDFLIBS = netcdff netcdf hdf5_hl hdf5 dl m z curl  #RON
##    NETCDFLIBS = netcdff netcdf hdf5hl_fortran hdf5_hl hdf5_fortran hdf5 z curl  #COSP
   NETCDFLIBS = netcdff netcdf hdf5hl_fortran hdf5_hl hdf5_fortran hdf5 dl m z curl  #SuperSet
   SCMEXTRALIBS = mpi
endif


#---- Abs targets -----------------------------------------------------
## SCM model targets (modelutils/scm/rpnphy)
.PHONY: scm fst2ptxt setfld forcedate allbin_scm allbincheck_scm

# .PHONY: scm_vfiles
# SCM_VFILES = scm_version.inc scm_version.h
# scm_vfiles: $(SCM_VFILES)
# scm_version.inc:
# 	.rdemkversionfile "scm" "$(SCM_VERSION)" . f
# scm_version.h:
# 	.rdemkversionfile "scm" "$(SCM_VERSION)" . c

allbin_scm: $(SCM_ABS)
	ls -l $(SCM_ABS_FILES)
allbincheck_scm:
	for item in $(SCM_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

mainscm     = $(ABSPREFIX)mainscm$(ABSPOSTFIX)_$(BASE_ARCH).Abs
mainscm_rel = $(ABSPREFIX)mainscm$(ABSPOSTFIX)_REL_$(BASE_ARCH).Abs
fst2ptxt    = fst2ptxt
setfld      = setfld
forcedate   = forcedate
# libscm      = libscm.a
# libscmutils = lib$(LIBSCMUTILS).a

scm: | scm_rm $(BINDIR)/$(mainscm)
	ls -l $(BINDIR)/$(mainscm)
scm_rm:
	if [[ "x$(SCM_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainscm) ; \
	fi
$(BINDIR)/$(mainscm): $(SCM_ABS_DEP) | $(SCM_VFILES)
	export MAINSUBNAME="scm" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(SCM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(SCM_LIBS_V) $(SCM_LIBS_DEP)" ;\
	$(RBUILD4objMPI)


fst2ptxt: | fst2ptxt_rm $(BINDIR)/$(fst2ptxt)
	ls -l $(BINDIR)/$(fst2ptxt)
fst2ptxt_rm:
	if [[ "x$(FST2PTXT_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(fst2ptxt) ; \
	fi
$(BINDIR)/$(fst2ptxt): $(FST2PTXT_ABS_DEP) | $(FST2PTXT_VFILES)
	export MAINSUBNAME="fst2ptxt" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(SCM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(SCM_LIBS_V) $(SCM_LIBS_DEP)" ;\
	$(RBUILD4objMPI)


setfld: | setfld_rm $(BINDIR)/$(setfld)
	ls -l $(BINDIR)/$(setfld)
setfld_rm:
	if [[ "x$(SETFLD_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(setfld) ; \
	fi
$(BINDIR)/$(setfld): $(SETFLD_ABS_DEP) | $(SETFLD_VFILES)
	export MAINSUBNAME="setfld" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(SCM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(SCM_LIBS_V) $(SCM_LIBS_DEP)" ;\
	$(RBUILD4objMPI)


forcedate: | forcedate_rm $(BINDIR)/$(forcedate)
	ls -l $(BINDIR)/$(forcedate)
forcedate_rm:
	if [[ "x$(FORCEDATE_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(forcedate) ; \
	fi
$(BINDIR)/$(forcedate): $(FORCEDATE_ABS_DEP) | $(FORCEDATE_VFILES)
	export MAINSUBNAME="forcedate" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(SCM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(SCM_LIBS_V) $(SCM_LIBS_DEP)" ;\
	$(RBUILD4objMPI)


#---- Lib target - automated ------------------------------------------
scm_LIB_template1 = \
$$(LIBDIR)/lib$(2)_$$($(3)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

.PHONY: scm_libs
scm_libs: $(OBJECTS_scm) $(SCM_LIBS_ALL_FILES_PLUS) | $(SCM_VFILES)
$(foreach item,$(SCM_LIBS_ALL_0),$(eval $(call scm_LIB_template1,$(item),$(item)$(SCM_SFX),SCM)))
$(foreach item,$(SCM_LIBS_ALL),$(eval $(call LIB_template2,$(item),SCM)))

$(LIBDIR)/lib$(SCM_LIB_MERGED_NAME)_$(SCM_VERSION).a: $(OBJECTS_MERGED_scm) | $(SCM_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_scm); mv $@_$$$$ $@
$(LIBDIR)/lib$(SCM_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(SCM_LIB_MERGED_NAME)_$(SCM_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(SCM_LIB_MERGED_NAME)_$(SCM_VERSION).a $@


ifneq (,$(DEBUGMAKE))
$(info ## ==== $$scm/include/Makefile.local.scm.mk [END] =====================)
endif
