ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnphy/include/Makefile.local.mk)
$(info ## )
endif

## RPNPhy definitions

# ifeq (,$(wildcard $(rpnphy)/VERSION))
#    $(error Not found: $(rpnphy)/VERSION)
# endif
# RPNPHY_VERSION0  = $(shell cat $(rpnphy)/VERSION | sed 's|x/||')
RPNPHY_VERSION0  = 5.8-LTS.9
RPNPHY_VERSION   = $(notdir $(RPNPHY_VERSION0))
RPNPHY_VERSION_X = $(dir $(RPNPHY_VERSION0))

PHYBINDIR = $(BINDIR)

## Some Shortcut/Alias to Lib Names
RPNPHY_LIBS_DEP = $(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)
RPNPHY_LIBS_SHARED_DEP = $(MODELUTILS_LIBS_SHARED_V) $(MODELUTILS_LIBS_SHARED_DEP)

LIBCLASS       = 
LIBPHYSURFACES = rpnphy_surface
LIBPHYCONVECT  = rpnphy_convect
LIBCHMPATH     = 
LIBCHM         = rpnphy_chm_stubs
LIBCPLPHY      = rpnphy_cpl_stubs

RPNPHY_LIBS_MERGED_0 = rpnphy_main rpnphy_api rpnphy_base $(LIBPHYSURFACES) $(LIBPHYCONVECT) rpnphy_series rpnphy_utils
RPNPHY_LIBS_OTHER_0  = $(LIBCLASS) $(LIBCHM) $(LIBCPLPHY) 

RPNPHY_SFX=$(RDE_BUILDDIR_SFX)
RPNPHY_LIBS_MERGED = $(foreach item,$(RPNPHY_LIBS_MERGED_0),$(item)$(RPNPHY_SFX))
RPNPHY_LIBS_OTHER  = $(foreach item,$(RPNPHY_LIBS_OTHER_0),$(item)$(RPNPHY_SFX))

RPNPHY_LIBS_ALL_0  = $(RPNPHY_LIBS_MERGED_0) $(RPNPHY_LIBS_OTHER_0)
RPNPHY_LIBS_ALL    = $(RPNPHY_LIBS_MERGED) $(RPNPHY_LIBS_OTHER)

RPNPHY_LIBS_0      = rpnphy$(RPNPHY_SFX)
RPNPHY_LIBS        = $(RPNPHY_LIBS_0) $(RPNPHY_LIBS_OTHER) 
RPNPHY_LIBS_V      = $(RPNPHY_LIBS_0)_$(RPNPHY_VERSION) $(RPNPHY_LIBS_OTHER) 

ifeq (,$(MAKE_NO_LIBSO))
RPNPHY_LIBS_SHARED_ALL = $(foreach item,$(RPNPHY_LIBS_ALL),$(item)-shared)
RPNPHY_LIBS_SHARED_0   = $(RPNPHY_LIBS_0)-shared
RPNPHY_LIBS_SHARED     = $(RPNPHY_LIBS_SHARED_0) $(RPNPHY_LIBS_OTHER) 
RPNPHY_LIBS_SHARED_V   = $(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION) $(RPNPHY_LIBS_OTHER) 
endif

RPNPHY_LIBS_OTHER_FILES = $(foreach item,$(RPNPHY_LIBS_OTHER),$(LIBDIR)/lib$(item).a) 
RPNPHY_LIBS_ALL_FILES   = $(foreach item,$(RPNPHY_LIBS_ALL),$(LIBDIR)/lib$(item).a)
ifeq (,$(MAKE_NO_LIBSO))
RPNPHY_LIBS_SHARED_FILES   = $(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0).so
endif
RPNPHY_LIBS_ALL_FILES_PLUS = $(LIBDIR)/lib$(RPNPHY_LIBS_0).a $(RPNPHY_LIBS_SHARED_FILES) $(RPNPHY_LIBS_ALL_FILES) 

OBJECTS_MERGED_rpnphy = $(foreach item,$(RPNPHY_LIBS_MERGED_0),$(OBJECTS_$(item)))

RPNPHY_MOD_FILES   = $(foreach item,$(FORTRAN_MODULES_rpnphy),$(item).[Mm][Oo][Dd])

RPNPHY_ABS         = feseri prphynml testphy test_sfclayer
RPNPHY_ABS_FILES   = $(PHYBINDIR)/$(mainfeseri) $(PHYBINDIR)/$(mainprphynml) $(PHYBINDIR)/$(maintestphy) $(PHYBINDIR)/$(mainsfclayer_test)

## Base Libpath and libs with placeholders for abs specific libs
#MODEL2_LIBAPPL = $(RPNPHY_LIBS_V)
MODEL2_LIBPATH = $(LIBCHMPATH)


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
	rm -f $(PHYBINDIR)/$(mainfeseri)
$(PHYBINDIR)/$(mainfeseri): | $(RPNPHY_VFILES)
	export MAINSUBNAME="feseri" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

mainprphynml = prphynml_$(BASE_ARCH).Abs
mainprphynmldep = prphynml.o cnv_options.o phy_options.o  sfc_options.o
mainprphynmldepfiles = $(foreach item,$(mainprphynmldep),$(BUILDOBJ)/$(item))
prphynml: | prphynml_rm $(PHYBINDIR)/$(mainprphynml)
	ls -l $(PHYBINDIR)/$(mainprphynml)
prphynml_rm:
	rm -f $(PHYBINDIR)/$(mainprphynml)
$(PHYBINDIR)/$(mainprphynml): $(mainprphynmldep) | $(RPNPHY_VFILES)
	export MAINSUBNAME="prphynml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	export RBUILD_EXTRA_OBJ="$(mainprphynmldepfiles)" ;\
	$(RBUILD4NOMPI)
	ls $@

maintestphy=maintestphy_$(BASE_ARCH).Abs
testphy: | testphy_rm $(PHYBINDIR)/$(maintestphy)
	ls -l $(PHYBINDIR)/$(maintestphy)
testphy_rm:
	rm -f $(PHYBINDIR)/$(maintestphy)
$(PHYBINDIR)/$(maintestphy): | $(RPNPHY_VFILES)
	export MAINSUBNAME="testphy_main" ;\
	export ATM_MODEL_NAME="testphy $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

maintest_sfclayer=maintest_sfclayer_$(BASE_ARCH).Abs
test_sfclayer: | test_sfclayer_rm $(PHYBINDIR)/$(maintest_sfclayer)
	ls -l $(PHYBINDIR)/$(maintest_sfclayer)
test_sfclayer_rm:
	rm -f $(PHYBINDIR)/$(maintest_sfclayer)
$(PHYBINDIR)/$(maintest_sfclayer): | $(RPNPHY_VFILES)
	export MAINSUBNAME="test_sfclayer" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(RPNPHY_VERSION)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_V) $(RPNPHY_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
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

$(LIBDIR)/lib$(RPNPHY_LIBS_0)_$(RPNPHY_VERSION).a: $(OBJECTS_rpnphy) | $(RPNPHY_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_rpnphy); mv $@_$$$$ $@
$(LIBDIR)/lib$(RPNPHY_LIBS_0).a: $(LIBDIR)/lib$(RPNPHY_LIBS_0)_$(RPNPHY_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(RPNPHY_LIBS_0)_$(RPNPHY_VERSION).a $@

$(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION).so: $(OBJECTS_rpnphy) $(RPNPHY_LIBS_OTHER_FILES) | $(RPNPHY_VFILES)
	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_rpnphy)" ;\
	export RBUILD_LIBAPPL="$(RPNPHY_LIBS_OTHER) $(RPNPHY_LIBS_DEP)" ;\
	$(RBUILD4MPI_SO)
	ls -l $@
$(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0).so: $(LIBDIR)/lib$(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION).so
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(RPNPHY_LIBS_SHARED_0)_$(RPNPHY_VERSION).so $@
	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnphy/include/Makefile.local.mk [END] ======================)
endif
