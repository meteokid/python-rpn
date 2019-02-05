ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnphy/include/Makefile.local.rpnphy.mk)
$(info ## )
endif

## RPNPhy definitions

# ifeq (,$(wildcard $(rpnphy)/VERSION))
#    $(error Not found: $(rpnphy)/VERSION)
# endif
# RPNPHY_VERSION0  = $(shell cat $(rpnphy)/VERSION | sed 's|x/||')
RPNPHY_VERSION0  = x/6.0.rc4
RPNPHY_VERSION   = $(notdir $(RPNPHY_VERSION0))
RPNPHY_VERSION_X = $(dir $(RPNPHY_VERSION0))

RPNPHY_SFX = $(RDE_LIBS_SFX)

PHYBINDIR = $(BINDIR)

## Some Shortcut/Alias to Lib Names
RPNPHY_LIBS_DEP = $(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)

LIBCLASS_0       = 
LIBCPLPHY_0      = rpnphy_cpl_stubs
LIBPHYSURFACES_0 = rpnphy_surface
LIBCHM_0         = rpnphy_chm_stubs

RPNPHY_LIB_MERGED_NAME_0 = rpnphy0
RPNPHY_LIBS_MERGED_0 = rpnphy_main rpnphy_api rpnphy_base $(LIBPHYSURFACES_0) rpnphy_series rpnphy_utils
RPNPHY_LIBS_OTHER_0  = $(LIBCLASS_0) $(LIBCHM_0) $(LIBCPLPHY_0)
RPNPHY_LIBS_ALL_0    = $(RPNPHY_LIBS_MERGED_0) $(RPNPHY_LIBS_OTHER_0)
RPNPHY_LIBS_0        = $(RPNPHY_LIB_MERGED_NAME_0) $(RPNPHY_LIBS_OTHER_0)

LIBCLASS       = $(foreach item,$(LIBCLASS_0),$(item)$(RPNPHY_SFX))
LIBCPLPHY      = $(foreach item,$(LIBCPLPHY_0),$(item)$(RPNPHY_SFX))
LIBPHYSURFACES = $(foreach item,$(LIBPHYSURFACES_0),$(item)$(RPNPHY_SFX))
LIBCHM         = $(foreach item,$(LIBCHM_0),$(item)$(RPNPHY_SFX))

RPNPHY_LIB_MERGED_NAME = $(foreach item,$(RPNPHY_LIB_MERGED_NAME_0),$(item)$(RPNPHY_SFX))
RPNPHY_LIBS_MERGED     = $(foreach item,$(RPNPHY_LIBS_MERGED_0),$(item)$(RPNPHY_SFX))
RPNPHY_LIBS_OTHER      = $(LIBCLASS) $(LIBCHM) $(LIBCPLPHY)
RPNPHY_LIBS_ALL        = $(RPNPHY_LIBS_MERGED) $(RPNPHY_LIBS_OTHER)
RPNPHY_LIBS            = $(RPNPHY_LIB_MERGED_NAME) $(RPNPHY_LIBS_OTHER)

RPNPHY_LIB_MERGED_NAME_V = $(foreach item,$(RPNPHY_LIB_MERGED_NAME),$(item)_$(RPNPHY_VERSION))
RPNPHY_LIBS_MERGED_V     = $(foreach item,$(RPNPHY_LIBS_MERGED),$(item)_$(RPNPHY_VERSION))
RPNPHY_LIBS_OTHER_V      = $(RPNPHY_LIBS_OTHER)
RPNPHY_LIBS_ALL_V        = $(RPNPHY_LIBS_MERGED_V) $(RPNPHY_LIBS_OTHER_V)
RPNPHY_LIBS_V            = $(RPNPHY_LIB_MERGED_NAME_V) $(RPNPHY_LIBS_OTHER_V)

RPNPHY_LIBS_ALL_FILES      = $(foreach item,$(RPNPHY_LIBS_ALL),$(LIBDIR)/lib$(item).a)
RPNPHY_LIBS_ALL_FILES_PLUS = $(LIBDIR)/lib$(RPNPHY_LIB_MERGED_NAME).a $(RPNPHY_LIBS_ALL_FILES)

OBJECTS_MERGED_rpnphy = $(foreach item,$(RPNPHY_LIBS_MERGED_0),$(OBJECTS_$(item)))

RPNPHY_MOD_FILES   = $(foreach item,$(FORTRAN_MODULES_rpnphy),$(item).[Mm][Oo][Dd])

# RPNPHY_ABS         = feseri prphynml testphy test_sfclayer
# RPNPHY_ABS_FILES   = $(PHYBINDIR)/$(mainfeseri) $(PHYBINDIR)/$(mainprphynml) $(PHYBINDIR)/$(maintestphy) $(PHYBINDIR)/$(mainsfclayer_test)
RPNPHY_ABS         = feseri prphynml
RPNPHY_ABS_FILES   = $(PHYBINDIR)/$(mainfeseri) $(PHYBINDIR)/$(mainprphynml)

## Base Libpath and libs with placeholders for abs specific libs
#MODEL2_LIBAPPL = $(RPNPHY_LIBS_V)
LIBCHMPATH     = 
MODEL2_LIBPATH = $(LIBCHMPATH)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(RPNPHY_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(RPNPHY_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   RPNPHY_ABS_DEP = $(RPNPHY_LIBS_ALL_FILES_PLUS) $(MODELUTILS_ABS_DEP)
endif

##
.PHONY: rpnphy_vfiles
RPNPHY_VFILES = rpnphy_version.inc rpnphy_version.h
rpnphy_vfiles: $(RPNPHY_VFILES)
rpnphy_version.inc:
	.rdemkversionfile "rpnphy" "$(RPNPHY_VERSION)" . f
rpnphy_version.h:
	.rdemkversionfile "rpnphy" "$(RPNPHY_VERSION)" . c


#---- Abs targets -----------------------------------------------------

## RPNPhy Targets
.PHONY: feseri prphynml testphy test_sfclayer allbin_rpnphy allbincheck_rpnphy

mainfeseri=feseri_$(BASE_ARCH).Abs
feseri: | feseri_rm $(PHYBINDIR)/$(mainfeseri)
	ls -l $(PHYBINDIR)/$(mainfeseri)
feseri_rm:
	if [[ "x$(RPNPHY_ABS_DEP)" == "x" ]] ; then \
		rm -f $(PHYBINDIR)/$(mainfeseri) ; \
	fi
$(PHYBINDIR)/$(mainfeseri): $(RPNPHY_ABS_DEP) $(GEMDYN_ABS_DEP) | $(RPNPHY_VFILES)
	export MAINSUBNAME="feseri" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP) $(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

mainprphynml = prphynml_$(BASE_ARCH).Abs
prphynml: | prphynml_rm $(PHYBINDIR)/$(mainprphynml)
	ls -l $(PHYBINDIR)/$(mainprphynml)
prphynml_rm:
	if [[ "x$(RPNPHY_ABS_DEP)" == "x" ]] ; then \
		rm -f $(PHYBINDIR)/$(mainprphynml) ; \
	fi
$(PHYBINDIR)/$(mainprphynml): $(RPNPHY_ABS_DEP) $(GEMDYN_ABS_DEP) | $(RPNPHY_VFILES)
	export MAINSUBNAME="prphynml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP) $(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)
	ls $@

maintestphy=maintestphy_$(BASE_ARCH).Abs
testphy: | testphy_rm $(PHYBINDIR)/$(maintestphy)
	ls -l $(PHYBINDIR)/$(maintestphy)
testphy_rm:
	if [[ "x$(RPNPHY_ABS_DEP)" == "x" ]] ; then \
		rm -f $(PHYBINDIR)/$(maintestphy) ; \
	fi
$(PHYBINDIR)/$(maintestphy): $(RPNPHY_ABS_DEP) $(GEMDYN_ABS_DEP) | $(RPNPHY_VFILES)
	export MAINSUBNAME="testphy_main" ;\
	export ATM_MODEL_NAME="testphy $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP) $(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

maintest_sfclayer=maintest_sfclayer_$(BASE_ARCH).Abs
test_sfclayer: | test_sfclayer_rm $(PHYBINDIR)/$(maintest_sfclayer)
	ls -l $(PHYBINDIR)/$(maintest_sfclayer)
test_sfclayer_rm:
	if [[ "x$(RPNPHY_ABS_DEP)" == "x" ]] ; then \
		rm -f $(PHYBINDIR)/$(maintest_sfclayer) ; \
	fi
$(PHYBINDIR)/$(maintest_sfclayer): $(RPNPHY_ABS_DEP) $(GEMDYN_ABS_DEP) | $(RPNPHY_VFILES)
	export MAINSUBNAME="test_sfclayer" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP) $(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

allbin_rpnphy: | $(RPNPHY_ABS)
allbincheck_rpnphy:
	for item in $(RPNPHY_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

#---- Lib target - automated ------------------------------------------
rpnphy_LIB_template1 = \
$$(LIBDIR)/lib$(2)_$$($(3)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

.PHONY: rpnphy_libs
rpnphy_libs: $(OBJECTS_rpnphy) $(RPNPHY_LIBS_ALL_FILES_PLUS) | $(RPNPHY_VFILES)
$(foreach item,$(RPNPHY_LIBS_ALL_0),$(eval $(call rpnphy_LIB_template1,$(item),$(item)$(RPNPHY_SFX),RPNPHY)))
$(foreach item,$(RPNPHY_LIBS_ALL),$(eval $(call LIB_template2,$(item),RPNPHY)))

$(LIBDIR)/lib$(RPNPHY_LIB_MERGED_NAME)_$(RPNPHY_VERSION).a: $(OBJECTS_MERGED_rpnphy) | $(RPNPHY_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_rpnphy); mv $@_$$$$ $@
$(LIBDIR)/lib$(RPNPHY_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(RPNPHY_LIB_MERGED_NAME)_$(RPNPHY_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(RPNPHY_LIB_MERGED_NAME)_$(RPNPHY_VERSION).a $@

# $(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION).so: $(OBJECTS_rpnphy) $(RPNPHY_LIBS_OTHER_FILES) | $(RPNPHY_VFILES)
# 	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_rpnphy)" ;\
# 	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_OTHER) $(RPNPHY_LIBS_DEP)" ;\
# 	$(RBUILD4MPI_SO)
# 	ls -l $@
# $(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0).so: $(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION).so
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION).so $@
# 	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnphy/include/Makefile.local.rpnphy.mk [END] ======================)
endif
