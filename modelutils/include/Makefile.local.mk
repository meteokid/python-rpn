ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$modelutils/include/Makefile.local.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

# ifeq (,$(wildcard $(modelutils)/VERSION))
#    $(error Not found: $(modelutils)/VERSION)
# endif
# MODELUTILS_VERSION0  = $(shell cat $(modelutils)/VERSION | sed 's|x/||')
MODELUTILS_VERSION0  = 1.4-LTS.7
MODELUTILS_VERSION   = $(notdir $(MODELUTILS_VERSION0))
MODELUTILS_VERSION_X = $(dir $(MODELUTILS_VERSION0))

#Component Specific compiler options
MODELUTILS_COMP_RULES_DEBUG = $(modelutils)/include/$(CONST_BASE_ARCH_OLD)/$(CONST_RDE_COMP_ARCH)/Compiler_rules_debug
ifneq (,$(filter intel%,$(COMP_ARCH))$(filter PrgEnv-intel%,$(COMP_ARCH)))
MODELUTILS_DEBUG = -g -traceback
else
MODELUTILS_DEBUG = -g
endif
MODEL1_FFLAGS = $(MODELUTILS_DEBUG)

## Some Shortcut/Alias to Lib Names
LIBHPCSPERF    = modelutils_tmg_stubs$(MODELUTILS_SFX)
RMN_VERSION_NUMBER = 016.2
ifeq (,$(COMM_VERSION))
COMM_VERSION   = _4051609
endif
ifeq (,$(LIBCOMM))
LIBCOMM        = rpn_comm$(COMM_VERSION) $(COMM_stubs)
endif
ifeq (,$(LIBCOMM_STUBS))
LIBCOMM_STUBS  = rpn_commstubs$(COMM_VERSION)
endif
ifeq (,$(VGRID_VERSION))
ifneq (,$(VGRIDDESCRIPTORS_VERSION))
VGRID_VERSION  = _$(VGRIDDESCRIPTORS_VERSION)
endif
endif
# LIBMUGMM       = 
LIBMUTMG       = modelutils_tmg_stubs
LIBMUSTUBS     = modelutils_stubs
# LIBMUVGRID     = 
# ifeq (,$(LIBVGRID))
# ifeq  (,$(VGRID_VERSION))
# LIBVGRID       = $(LIBMUVGRID)
# else
LIBVGRID       = descrip$(VGRID_VERSION)
# endif
# endif
# LIBGMM         = $(LIBMUGMM)
RMN_VERSION    = rmnMP_$(RMN_VERSION_NUMBER)$(FORCE_RMN_VERSION_RC)
LIBRMN         = $(RMN_VERSION)
LIBRMN_SHARED  = rmnshared_$(RMN_VERSION_NUMBER)$(FORCE_RMN_VERSION_RC)
MODELUTILS_LIBS_DEP = $(LIBVGRID) $(LIBRMN) $(LIBCOMM) $(LIBHPCSPERF) $(LIBMASS) $(LAPACK) $(BLAS) $(RTOOLS) $(BINDCPU) $(LLAPI) $(IBM_LD) $(LIBHPC) $(LIBPMAPI)
MODELUTILS_LIBS_SHARED_DEP = $(LIBVGRID) $(LIBRMN_SHARED) $(LIBCOMM) $(LIBHPCSPERF) $(LIBMASS) $(LAPACK) $(BLAS) $(RTOOLS) $(BINDCPU) $(LLAPI) $(IBM_LD) $(LIBHPC) $(LIBPMAPI)

MODELUTILS_LIBS_MERGED_0 = modelutils_main modelutils_driver modelutils_utils modelutils_tdpack modelutils_base
ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
#MODELUTILS_LIBS_OTHER_0  = $(LIBMUSTUBS) modelutils_massvp7_wrap $(LIBMUTMG) $(LIBGMM)
MODELUTILS_LIBS_OTHER_0  = $(LIBMUSTUBS) modelutils_massvp7_wrap $(LIBMUTMG)
else
#MODELUTILS_LIBS_OTHER_0  = $(LIBMUSTUBS) $(LIBMUTMG) $(LIBGMM)
MODELUTILS_LIBS_OTHER_0  = $(LIBMUSTUBS) $(LIBMUTMG)
endif
#MODELUTILS_LIBS_BUILDEXTRA_0  = $(LIBMUVGRID) $(LIBMUGMM)
#MODELUTILS_LIBS_BUILDEXTRA_0  = $(LIBMUVGRID)

MODELUTILS_SFX=$(RDE_BUILDDIR_SFX)
MODELUTILS_LIBS_MERGED = $(foreach item,$(MODELUTILS_LIBS_MERGED_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_OTHER  = $(foreach item,$(MODELUTILS_LIBS_OTHER_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_BUILDEXTRA = $(foreach item,$(MODELUTILS_LIBS_BUILDEXTRA_0),$(item)$(MODELUTILS_SFX))

MODELUTILS_LIBS_ALL_0  = $(MODELUTILS_LIBS_MERGED_0) $(MODELUTILS_LIBS_OTHER_0)
MODELUTILS_LIBS_ALL    = $(MODELUTILS_LIBS_MERGED) $(MODELUTILS_LIBS_OTHER)
MODELUTILS_LIBS_0      = modelutils$(MODELUTILS_SFX)
MODELUTILS_LIBS        = $(MODELUTILS_LIBS_0) $(MODELUTILS_LIBS_OTHER)
MODELUTILS_LIBS_V      = $(MODELUTILS_LIBS_0)_$(MODELUTILS_VERSION) $(MODELUTILS_LIBS_OTHER)
 
ifeq (,$(MAKE_NO_LIBSO))
MODELUTILS_LIBS_SHARED_ALL = $(foreach item,$(MODELUTILS_LIBS_ALL),$(item)-shared)
MODELUTILS_LIBS_SHARED_0   = $(MODELUTILS_LIBS_0)-shared
MODELUTILS_LIBS_SHARED     = $(MODELUTILS_LIBS_SHARED_0) $(MODELUTILS_LIBS_OTHER)
MODELUTILS_LIBS_SHARED_V   = $(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION) $(MODELUTILS_LIBS_OTHER)
endif

#MODELUTILS_LIBS_FL_FILES = $(LIBDIR)/lib$(LIBMUGMM)$(MODELUTILS_SFX).a.fl $(LIBDIR)/lib$(LIBMUGMM)$(MODELUTILS_SFX)_$(MODELUTILS_VERSION).a.fl

MODELUTILS_LIBS_OTHER_FILES = $(foreach item,$(MODELUTILS_LIBS_OTHER) $(MODELUTILS_LIBS_BUILDEXTRA),$(LIBDIR)/lib$(item).a) 
MODELUTILS_LIBS_ALL_FILES   = $(foreach item,$(MODELUTILS_LIBS_ALL) $(MODELUTILS_LIBS_BUILDEXTRA),$(LIBDIR)/lib$(item).a) 
ifeq (,$(MAKE_NO_LIBSO))
MODELUTILS_LIBS_SHARED_FILES   = $(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0).so
endif
MODELUTILS_LIBS_ALL_FILES_PLUS = $(LIBDIR)/lib$(MODELUTILS_LIBS_0).a $(MODELUTILS_LIBS_SHARED_FILES) $(MODELUTILS_LIBS_ALL_FILES) $(MODELUTILS_LIBS_FL_FILES)

OBJECTS_MERGED_modelutils = $(foreach item,$(MODELUTILS_LIBS_MERGED_0),$(OBJECTS_$(item)))

MODELUTILS_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_modelutils),$(item).[Mm][Oo][Dd])

MODELUTILS_ABS        = yyencode yydecode yy2global time2sec_main flipit
MODELUTILS_ABS_FILES  = $(foreach item,$(MODELUTILS_ABS),$(BINDIR)/$(item).Abs)

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(MODELUTILS_LIBS_V)


##
.PHONY: modelutils_vfiles
MODELUTILS_VFILES = modelutils_version.inc modelutils_version.h
modelutils_vfiles: $(MODELUTILS_VFILES)
modelutils_version.inc:
	.rdemkversionfile "modelutils" "$(MODELUTILS_VERSION)" . f
modelutils_version.h:
	.rdemkversionfile "modelutils" "$(MODELUTILS_VERSION)" . c

BUILDNAME = 

#---- ARCH Specific overrides -----------------------------------------
ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
LIBHPCSPERF   = hpcsperf
LIBMASSWRAP_0 = modelutils_massvp7_wrap
LIBMASSWRAP   = $(LIBMASSWRAP_0)$(MODELUTILS_SFX)
endif

#---- Abs targets -----------------------------------------------------

## Modelutils Targets
.PHONY: $(MODELUTILS_ABS)

mainyyencode=yyencode.Abs
yyencode: | yyencode_rm $(BINDIR)/$(mainyyencode)
	ls -l $(BINDIR)/$(mainyyencode)
yyencode_rm:
	rm -f $(BINDIR)/$(mainyyencode)
$(BINDIR)/$(mainyyencode): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="yyencode" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

mainyydecode=yydecode.Abs
yydecode: | yydecode_rm $(BINDIR)/$(mainyydecode)
	ls -l $(BINDIR)/$(mainyydecode)
yydecode_rm:
	rm -f $(BINDIR)/$(mainyydecode)
$(BINDIR)/$(mainyydecode): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="yydecode" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

mainyy2global=yy2global.Abs
yy2global: | yy2global_rm $(BINDIR)/$(mainyy2global)
	ls -l $(BINDIR)/$(mainyy2global)
yy2global_rm:
	rm -f $(BINDIR)/$(mainyy2global)
$(BINDIR)/$(mainyy2global): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="yy2global" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

maintime2sec=time2sec_main.Abs
time2sec_main: time2sec
time2sec: | time2sec_rm $(BINDIR)/$(maintime2sec)
	ls -l $(BINDIR)/$(maintime2sec)
time2sec_rm:
	rm -f $(BINDIR)/$(maintime2sec)
$(BINDIR)/$(maintime2sec): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="time2sec_main" ;\
	export ATM_MODEL_NAME="time2sec $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

mainflipit=flipit.Abs
flipit: | flipit_rm $(BINDIR)/$(mainflipit)
	ls -l $(BINDIR)/$(mainflipit)
flipit_rm:
	rm -f $(BINDIR)/$(mainflipit)
$(BINDIR)/$(mainflipit): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="flipit" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

.PHONY: allbin_modelutils allbincheck_modelutils
allbin_modelutils: | $(MODELUTILS_ABS)
allbincheck_modelutils:
	for item in $(MODELUTILS_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

#---- Lib target - automated ------------------------------------------
modelutils_LIB_template1 = \
$$(LIBDIR)/lib$(2)_$$($(3)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

.PHONY: modelutils_libs
modelutils_libs: $(OBJECTS_modelutils) $(MODELUTILS_LIBS_ALL_FILES_PLUS) | $(MODELUTILS_VFILES)
$(foreach item,$(MODELUTILS_LIBS_ALL_0) $(MODELUTILS_LIBS_BUILDEXTRA_0),$(eval $(call modelutils_LIB_template1,$(item),$(item)$(MODELUTILS_SFX),MODELUTILS)))
$(foreach item,$(MODELUTILS_LIBS_ALL) $(MODELUTILS_LIBS_BUILDEXTRA),$(eval $(call LIB_template2,$(item),MODELUTILS)))

# $(LIBDIR)/lib$(LIBMUGMM)$(MODELUTILS_SFX).a.fl: $(LIBDIR)/lib$(LIBMUGMM)$(MODELUTILS_SFX).a
# 	cd $(LIBDIR) ;\
# 	ln -sf  lib$(LIBMUGMM)$(MODELUTILS_SFX).a \
# 			lib$(LIBMUGMM)$(MODELUTILS_SFX).a.fl
# $(LIBDIR)/lib$(LIBMUGMM)$(MODELUTILS_SFX)_$(MODELUTILS_VERSION).a.fl: $(LIBDIR)/lib$(LIBMUGMM)$(MODELUTILS_SFX)_$(MODELUTILS_VERSION).a
# 	cd $(LIBDIR) ;\
# 	ln -sf  lib$(LIBMUGMM)$(MODELUTILS_SFX)_$(MODELUTILS_VERSION).a \
# 			lib$(LIBMUGMM)$(MODELUTILS_SFX)_$(MODELUTILS_VERSION).a.fl

modelutils_libs_fl: $(MODELUTILS_LIBS_FL_FILES)

$(LIBDIR)/lib$(MODELUTILS_LIBS_0)_$(MODELUTILS_VERSION).a: $(OBJECTS_modelutils) | $(MODELUTILS_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_modelutils); mv $@_$$$$ $@
$(LIBDIR)/lib$(MODELUTILS_LIBS_0).a: $(LIBDIR)/lib$(MODELUTILS_LIBS_0)_$(MODELUTILS_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(MODELUTILS_LIBS_0)_$(MODELUTILS_VERSION).a $@

$(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION).so: $(OBJECTS_modelutils) $(MODELUTILS_LIBS_OTHER_FILES) | $(MODELUTILS_VFILES)
	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_modelutils)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_OTHER) $(MODELUTILS_LIBS_DEP)" ;\
	$(RBUILD4MPI_SO)
	ls -l $@
$(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0).so: $(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION).so
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION).so $@
	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$modelutils/include/Makefile.local.mk [END] ==================)
endif
