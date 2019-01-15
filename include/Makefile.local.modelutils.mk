ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$modelutils/include/Makefile.local.modelutils.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

# ifeq (,$(wildcard $(modelutils)/VERSION))
#    $(error Not found: $(modelutils)/VERSION)
# endif
# MODELUTILS_VERSION0  = $(shell cat $(modelutils)/VERSION | sed 's|x/||')
MODELUTILS_VERSION0  = x/1.5.rc3
MODELUTILS_VERSION   = $(notdir $(MODELUTILS_VERSION0))
MODELUTILS_VERSION_X = $(dir $(MODELUTILS_VERSION0))

MODELUTILS_SFX = $(RDE_LIBS_SFX)

#Component Specific compiler options
MODELUTILS_COMP_RULES_DEBUG = $(modelutils)/include/$(CONST_BASE_ARCH_OLD)/$(CONST_RDE_COMP_ARCH)/Compiler_rules_debug
MODELUTILS_DEBUG = -g -traceback
MODEL1_FFLAGS = $(MODELUTILS_DEBUG)
MODEL1_DEFINE = -DECCCGEM

## Some Shortcut/Alias to Lib Names
RMN_VERSION_NUMBER = 016.2
RMN_VERSION    = rmnMP_$(RMN_VERSION_NUMBER)$(FORCE_RMN_VERSION_RC)
COMM_VERSION   = _4051612
LIBRMN         = $(RMN_VERSION)
LIBRMN_SHARED  = rmnshared_$(RMN_VERSION_NUMBER)$(FORCE_RMN_VERSION_RC)
LIBCOMM        = rpn_comm$(COMM_VERSION) $(COMM_stubs)
LIBCOMM_STUBS  = rpn_commstubs$(COMM_VERSION)

ifeq (,$(VGRID_VERSION))
ifneq (,$(VGRIDDESCRIPTORS_VERSION))
VGRID_VERSION  = _$(VGRIDDESCRIPTORS_VERSION)
endif
endif
LIBVGRID    = descrip$(VGRID_VERSION)

LIBHPCSPERF = 

MODELUTILS_LIBS_DEP = $(LIBVGRID) $(LIBRMN) $(LIBCOMM) $(LIBHPCSPERF) $(LIBMASS) $(LAPACK) $(BLAS) $(RTOOLS) $(BINDCPU) $(LLAPI) $(IBM_LD) $(LIBHPC) $(LIBPMAPI)

LIBMUTMG_0   = modelutils_tmg_stubs
LIBMUSTUBS_0 = modelutils_stubs
MODELUTILS_DUMMYMPISTUBS_0 = modelutils_dummympistubs
MODELUTILS_TESTS_LIB_0     = modelutils_tests
#TODO: make ifort lib conditional
MODELUTILS_IFORT_LIB_0     = modelutils_ov_ifort
MODELUTILS_TDPACK_LIB_0    = modelutils_tdpack

MODELUTILS_LIB_MERGED_NAME_0 = modelutils0
MODELUTILS_LIBS_MERGED_0 = modelutils_main modelutils_driver modelutils_utils $(MODELUTILS_TDPACK_LIB_0) modelutils_base
MODELUTILS_LIBS_OTHER_0  = $(LIBMUSTUBS_0) $(LIBMUTMG_0) $(MODELUTILS_IFORT_LIB_0)
MODELUTILS_LIBS_EXTRA_0  = $(MODELUTILS_DUMMYMPISTUBS_0) $(MODELUTILS_TESTS_LIB_0)
MODELUTILS_LIBS_ALL_0    = $(MODELUTILS_LIBS_MERGED_0) $(MODELUTILS_LIBS_OTHER_0)
MODELUTILS_LIBS_FL_0     = $(MODELUTILS_DUMMYMPISTUBS_0) $(MODELUTILS_IFORT_LIB_0)
MODELUTILS_LIBS_0        = $(MODELUTILS_LIB_MERGED_NAME_0) $(MODELUTILS_LIBS_OTHER_0)

LIBMUTMG   = $(foreach item,$(LIBMUTMG_0),$(item)$(MODELUTILS_SFX))
LIBMUSTUBS = $(foreach item,$(LIBMUSTUBS_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_DUMMYMPISTUBS = $(foreach item,$(MODELUTILS_DUMMYMPISTUBS_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_TESTS_LIB     = $(foreach item,$(MODELUTILS_TESTS_LIB_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_IFORT_LIB     = $(foreach item,$(MODELUTILS_IFORT_LIB_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_TDPACK_LIB    = $(foreach item,$(MODELUTILS_TDPACK_LIB_0),$(item)$(MODELUTILS_SFX))

MODELUTILS_LIB_MERGED_NAME = $(foreach item,$(MODELUTILS_LIB_MERGED_NAME_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_MERGED = $(foreach item,$(MODELUTILS_LIBS_MERGED_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_OTHER  = $(foreach item,$(MODELUTILS_LIBS_OTHER_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_EXTRA  = $(foreach item,$(MODELUTILS_LIBS_EXTRA_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_ALL    = $(foreach item,$(MODELUTILS_LIBS_ALL_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS_FL     = $(foreach item,$(MODELUTILS_LIBS_FL_0),$(item)$(MODELUTILS_SFX))
MODELUTILS_LIBS        = $(foreach item,$(MODELUTILS_LIBS_0),$(item)$(MODELUTILS_SFX))

MODELUTILS_LIB_MERGED_NAME_V = $(foreach item,$(MODELUTILS_LIB_MERGED_NAME),$(item)_$(MODELUTILS_VERSION))
MODELUTILS_LIBS_MERGED_V = $(foreach item,$(MODELUTILS_LIBS_MERGED),$(item)_$(MODELUTILS_VERSION))
MODELUTILS_LIBS_OTHER_V  = $(foreach item,$(MODELUTILS_LIBS_OTHER),$(item)_$(MODELUTILS_VERSION))
MODELUTILS_LIBS_EXTRA_V  = $(foreach item,$(MODELUTILS_LIBS_EXTRA),$(item)_$(MODELUTILS_VERSION))
MODELUTILS_LIBS_ALL_V    = $(foreach item,$(MODELUTILS_LIBS_ALL),$(item)_$(MODELUTILS_VERSION))
MODELUTILS_LIBS_FL_V     = $(foreach item,$(MODELUTILS_LIBS_FL),$(item)_$(MODELUTILS_VERSION))
MODELUTILS_LIBS_V        = $(foreach item,$(MODELUTILS_LIBS),$(item)_$(MODELUTILS_VERSION))

MODELUTILS_LIBS_FL_FILES       = $(foreach item,$(MODELUTILS_LIBS_FL),$(LIBDIR)/lib$(item).a.fl)
MODELUTILS_LIBS_ALL_FILES      = $(foreach item,$(MODELUTILS_LIBS_ALL) $(MODELUTILS_LIBS_EXTRA),$(LIBDIR)/lib$(item).a)
MODELUTILS_LIBS_ALL_FILES_PLUS = $(LIBDIR)/lib$(MODELUTILS_LIB_MERGED_NAME).a $(MODELUTILS_LIBS_ALL_FILES) $(MODELUTILS_LIBS_FL_FILES)

OBJECTS_MERGED_modelutils = $(foreach item,$(MODELUTILS_LIBS_MERGED_0),$(OBJECTS_$(item)))

MODELUTILS_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_modelutils),$(item).[Mm][Oo][Dd])

MODELUTILS_ABS        = yyencode yydecode yy2global time2sec_main flipit test_integrals
MODELUTILS_ABS_FILES  = $(foreach item,$(MODELUTILS_ABS),$(BINDIR)/$(item).Abs)

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(MODELUTILS_LIBS_V)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(MODELUTILS_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(MODELUTILS_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   MODELUTILS_ABS_DEP = $(MODELUTILS_LIBS_ALL_FILES_PLUS) $(RDE_ABS_DEP)
endif

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

#---- Abs targets -----------------------------------------------------

## Modelutils Targets
.PHONY: $(MODELUTILS_ABS)

mainyyencode=yyencode.Abs
yyencode: | yyencode_rm $(BINDIR)/$(mainyyencode)
	ls -l $(BINDIR)/$(mainyyencode)
yyencode_rm:
	if [[ "x$(MODELUTILS_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainyyencode) ; \
	fi
$(BINDIR)/$(mainyyencode): $(MODELUTILS_ABS_DEP) | $(MODELUTILS_VFILES)
	export MAINSUBNAME="yyencode" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

mainyydecode=yydecode.Abs
yydecode: | yydecode_rm $(BINDIR)/$(mainyydecode)
	ls -l $(BINDIR)/$(mainyydecode)
yydecode_rm:
	if [[ "x$(MODELUTILS_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainyydecode) ; \
	fi
$(BINDIR)/$(mainyydecode): $(MODELUTILS_ABS_DEP) | $(MODELUTILS_VFILES)
	export MAINSUBNAME="yydecode" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

mainyy2global=yy2global.Abs
yy2global: | yy2global_rm $(BINDIR)/$(mainyy2global)
	ls -l $(BINDIR)/$(mainyy2global)
yy2global_rm:
	if [[ "x$(MODELUTILS_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainyy2global) ; \
	fi
$(BINDIR)/$(mainyy2global): $(MODELUTILS_ABS_DEP) | $(MODELUTILS_VFILES)
	export MAINSUBNAME="yy2global" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

maintime2sec=time2sec_main.Abs
time2sec_main: time2sec
time2sec: | time2sec_rm $(BINDIR)/$(maintime2sec)
	ls -l $(BINDIR)/$(maintime2sec)
time2sec_rm:
	if [[ "x$(MODELUTILS_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(maintime2sec) ; \
	fi
$(BINDIR)/$(maintime2sec): $(MODELUTILS_ABS_DEP) | $(MODELUTILS_VFILES)
	export MAINSUBNAME="time2sec_main" ;\
	export ATM_MODEL_NAME="time2sec $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

maintest_integrals=test_integrals.Abs
test_integrals: | test_integrals_rm $(BINDIR)/$(maintest_integrals)
	ls -l $(BINDIR)/$(maintest_integrals)
test_integrals_rm:
	if [[ "x$(MODELUTILS_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(maintest_integrals) ; \
	fi
$(BINDIR)/test_integrals.Abs: $(MODELUTILS_ABS_DEP) | $(MODELUTILS_VFILES)
	export MAINSUBNAME="test_integrals" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_TESTS_LIB) $(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS);\
	$(RBUILD4objNOMPI)

mainflipit=flipit.Abs
flipit: | flipit_rm $(BINDIR)/$(mainflipit)
	ls -l $(BINDIR)/$(mainflipit)
flipit_rm:
	if [[ "x$(MODELUTILS_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainflipit) ; \
	fi
$(BINDIR)/$(mainflipit): $(MODELUTILS_ABS_DEP) | $(MODELUTILS_VFILES)
	export MAINSUBNAME="flipit" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
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

LIB_template1fl = \
$$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a.fl: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a $$@

LIB_template2fl = \
$$(LIBDIR)/lib$(1).a.fl: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a.fl ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a.fl $$@


.PHONY: modelutils_libs
modelutils_libs: $(OBJECTS_modelutils) $(MODELUTILS_LIBS_ALL_FILES_PLUS) | $(MODELUTILS_VFILES)
$(foreach item,$(MODELUTILS_LIBS_ALL_0) $(MODELUTILS_LIBS_EXTRA_0),$(eval $(call modelutils_LIB_template1,$(item),$(item)$(MODELUTILS_SFX),MODELUTILS)))
$(foreach item,$(MODELUTILS_LIBS_ALL) $(MODELUTILS_LIBS_EXTRA),$(eval $(call LIB_template2,$(item),MODELUTILS)))
$(foreach item,$(MODELUTILS_LIBS_FL),$(eval $(call LIB_template1fl,$(item),MODELUTILS)))
$(foreach item,$(MODELUTILS_LIBS_FL),$(eval $(call LIB_template2fl,$(item),MODELUTILS)))

modelutils_libs_fl: $(MODELUTILS_LIBS_FL_FILES)

$(LIBDIR)/lib$(MODELUTILS_LIB_MERGED_NAME)_$(MODELUTILS_VERSION).a: $(OBJECTS_MERGED_modelutils) | $(MODELUTILS_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_modelutils); mv $@_$$$$ $@
$(LIBDIR)/lib$(MODELUTILS_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(MODELUTILS_LIB_MERGED_NAME)_$(MODELUTILS_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(MODELUTILS_LIB_MERGED_NAME)_$(MODELUTILS_VERSION).a $@

# $(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION).so: $(OBJECTS_modelutils) $(MODELUTILS_LIBS_OTHER_FILES) | $(MODELUTILS_VFILES)
# 	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_modelutils)" ;\
# 	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_OTHER) $(MODELUTILS_LIBS_DEP)" ;\
# 	$(RBUILD4MPI_SO)
# 	ls -l $@
# $(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0).so: $(LIBDIR)/lib$(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION).so
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(MODELUTILS_LIBS_SHARED_0)_$(MODELUTILS_VERSION).so $@
# 	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$modelutils/include/Makefile.local.modelutils.mk [END] ==================)
endif
