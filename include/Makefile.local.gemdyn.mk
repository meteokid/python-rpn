ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gemdyn/include/Makefile.local.gemdyn.mk)
$(info ## )
endif

## GEM model and GemDyn definitions

# ifeq (,$(wildcard $(gemdyn)/VERSION))
#    $(error Not found: $(gemdyn)/VERSION)
# endif
# GEMDYN_VERSION0  = $(shell cat $(gemdyn)/VERSION | sed 's|x/||')
GEMDYN_VERSION0  = x/5.0.b7rde
GEMDYN_VERSION   = $(notdir $(GEMDYN_VERSION0))
GEMDYN_VERSION_X = $(dir $(GEMDYN_VERSION0))

GEMDYN_SFX = $(RDE_LIBS_SFX)

## Some Shortcut/Alias to Lib Names
GEMDYN_LIBS_DEP = $(RPNPHY_LIBS_V) $(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)

LIBCPLPATH   =
LIBCPL       =
LIBCANONICAL =

GEMDYN_LIB_MERGED_NAME_0 = gemdyn0
GEMDYN_LIBS_MERGED_0 = gemdyn_main gemdyn_base gemdyn_canonical
GEMDYN_LIBS_OTHER_0  = $(LIBCPL) $(LIBCANONICAL)
GEMDYN_LIBS_ALL_0    = $(GEMDYN_LIBS_MERGED_0) $(GEMDYN_LIBS_OTHER_0)
GEMDYN_LIBS_0        = $(GEMDYN_LIB_MERGED_NAME_0) $(GEMDYN_LIBS_OTHER_0)

GEMDYN_LIB_MERGED_NAME = $(foreach item,$(GEMDYN_LIB_MERGED_NAME_0),$(item)$(GEMDYN_SFX))
GEMDYN_LIBS_MERGED = $(foreach item,$(GEMDYN_LIBS_MERGED_0),$(item)$(GEMDYN_SFX))
GEMDYN_LIBS_OTHER  = $(LIBCPL) $(LIBCANONICAL)
GEMDYN_LIBS_ALL    = $(GEMDYN_LIBS_MERGED) $(GEMDYN_LIBS_OTHER)
GEMDYN_LIBS        = $(GEMDYN_LIB_MERGED_NAME) $(GEMDYN_LIBS_OTHER)

GEMDYN_LIB_MERGED_NAME_V = $(foreach item,$(GEMDYN_LIB_MERGED_NAME),$(item)_$(GEMDYN_VERSION))
GEMDYN_LIBS_MERGED_V = $(foreach item,$(GEMDYN_LIBS_MERGED),$(item)_$(GEMDYN_VERSION))
GEMDYN_LIBS_OTHER_V  = $(LIBCPL) $(LIBCANONICAL)
GEMDYN_LIBS_ALL_V    = $(GEMDYN_LIBS_MERGED_V) $(GEMDYN_LIBS_OTHER_V)
GEMDYN_LIBS_V        = $(GEMDYN_LIB_MERGED_NAME_V) $(GEMDYN_LIBS_OTHER_V)

GEMDYN_LIBS_ALL_FILES      = $(foreach item,$(GEMDYN_LIBS_ALL),$(LIBDIR)/lib$(item).a)
GEMDYN_LIBS_ALL_FILES_PLUS = $(LIBDIR)/lib$(GEMDYN_LIB_MERGED_NAME).a $(GEMDYN_LIBS_ALL_FILES)

OBJECTS_MERGED_gemdyn = $(foreach item,$(GEMDYN_LIBS_MERGED_0),$(OBJECTS_$(item)))

GEMDYN_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_gemdyn),$(item).[Mm][Oo][Dd])

GEMDYN_ABS        = gem_monitor_end gem_monitor_output toc2nml gemgrid checkdmpart prgemnml split3df
GEMDYN_ABS_FILES  = $(BINDIR)/gem_monitor_output $(BINDIR)/gem_monitor_end $(BINDIR)/toc2nml $(BINDIR)/gemgrid_$(BASE_ARCH).Abs $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs $(BINDIR)/gemprnml_$(BASE_ARCH).Abs $(BINDIR)/split3df_$(BASE_ARCH).Abs

## GEM model Libpath and libs
#MODEL3_LIBAPPL = $(GEMDYN_LIBS_V)
MODEL3_LIBPATH = $(LIBCPLPATH)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(GEMDYN_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(GEMDYN_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   GEMDYN_ABS_DEP = $(GEMDYN_LIBS_ALL_FILES_PLUS) $(RPNPHY_ABS_DEP)
endif

##
.PHONY: gemdyn_vfiles
GEMDYN_VFILES = gemdyn_version.inc gemdyn_version.h
gemdyn_vfiles: $(GEMDYN_VFILES)
gemdyn_version.inc:
	.rdemkversionfile "gemdyn" "$(GEMDYN_VERSION)" . f
gemdyn_version.h:
	.rdemkversionfile "gemdyn" "$(GEMDYN_VERSION)" . c

#---- Abs targets -----------------------------------------------------

## GemDyn Targets

.PHONY: prgemnml gemgrid checkdmpart split3df toc2nml monitor sometools allbin allbincheck

mainprgemnml=gemprnml_$(BASE_ARCH).Abs
prgemnml: | prgemnml_rm $(BINDIR)/$(mainprgemnml)
	ls -l $(BINDIR)/$(mainprgemnml)
prgemnml_rm:
	if [[ "x$(GEMDYN_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainprgemnml) ; \
	fi
$(BINDIR)/$(mainprgemnml): $(GEMDYN_ABS_DEP) | $(GEMDYN_VFILES)
	export MAINSUBNAME="prgemnml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)
	ls $@

maingemgrid=gemgrid_$(BASE_ARCH).Abs
gemgrid: | gemgrid_rm $(BINDIR)/$(maingemgrid)
	ls -l $(BINDIR)/$(maingemgrid)
gemgrid_rm:
	if [[ "x$(GEMDYN_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(maingemgrid) ; \
	fi
$(BINDIR)/$(maingemgrid): $(GEMDYN_ABS_DEP) | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemgrid" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)

maincheckdmpart=checkdmpart_$(BASE_ARCH).Abs
checkdmpart: | checkdmpart_rm $(BINDIR)/$(maincheckdmpart)
	ls -lL $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs
checkdmpart_rm:
	if [[ "x$(GEMDYN_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(maincheckdmpart) ; \
	fi
$(BINDIR)/$(maincheckdmpart): $(GEMDYN_ABS_DEP) | $(GEMDYN_VFILES)
	export MAINSUBNAME="checkdmpart" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)

mainsplit3df=split3df_$(BASE_ARCH).Abs
split3df: | split3df_rm $(BINDIR)/$(mainsplit3df)
	ls -lL $(BINDIR)/$(mainsplit3df)
split3df_rm:
	if [[ "x$(GEMDYN_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/$(mainsplit3df) ; \
	fi
$(BINDIR)/$(mainsplit3df): $(GEMDYN_ABS_DEP) | $(GEMDYN_VFILES)
	export MAINSUBNAME="split3df" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)


toc2nml: | toc2nml_rm $(BINDIR)/toc2nml
	ls -lL $(BINDIR)/toc2nml
toc2nml_rm:
	if [[ "x$(GEMDYN_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/toc2nml ; \
	fi
$(BINDIR)/toc2nml: $(GEMDYN_ABS_DEP) | $(GEMDYN_VFILES)
	export MAINSUBNAME="toc2nml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)

gem_monitor_end: $(BINDIR)/gem_monitor_end
	ls -lL $(BINDIR)/$@
gem_monitor_end.c:
	if [[ ! -f $@ ]] ; then rdeco $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_end: gem_monitor_end.c
	mybidon=gem_monitor_end_123456789 ;\
	if [[ ! -f gem_monitor_end.c ]] ; then rdeco gem_monitor_end.c ; fi ;\
	cat gem_monitor_end.c | sed 's/main_gem_monitor_end/main/' > $${mybidon}.c ;\
	$(RDECC) -o $@ -src $${mybidon}.c && rm -f $${mybidon}.[co]

gem_monitor_output: $(BINDIR)/gem_monitor_output
	ls -lL $(BINDIR)/$@
gem_monitor_output.c:
	if [[ ! -f $@ ]] ; then rdeco $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_output: gem_monitor_output.c
	mybidon=gem_monitor_output_123456789 ;\
	if [[ ! -f gem_monitor_output.c ]] ; then rdeco gem_monitor_output.c ; fi ;\
	cat gem_monitor_output.c | sed 's/main_gem_monitor_output/main/' > $${mybidon}.c ;\
	$(RDECC) -o $@ -src $${mybidon}.c && rm -f $${mybidon}.[co]

monitor: | $(BINDIR)/gem_monitor_end $(BINDIR)/gem_monitor_output
sometools: | prgemnml gemgrid toc2nml

allbin_gemdyn: | $(GEMDYN_ABS)
allbincheck_gemdyn:
	for item in $(GEMDYN_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

#---- Lib target - automated ------------------------------------------
gemdyn_LIB_template1 = \
$$(LIBDIR)/lib$(2)_$$($(3)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

.PHONY: gemdyn_libs
gemdyn_libs: $(OBJECTS_gemdyn) $(GEMDYN_LIBS_ALL_FILES_PLUS) | $(GEMDYN_VFILES)
$(foreach item,$(GEMDYN_LIBS_ALL_0),$(eval $(call gemdyn_LIB_template1,$(item),$(item)$(GEMDYN_SFX),GEMDYN)))
$(foreach item,$(GEMDYN_LIBS_ALL),$(eval $(call LIB_template2,$(item),GEMDYN)))

$(LIBDIR)/lib$(GEMDYN_LIB_MERGED_NAME)_$(GEMDYN_VERSION).a: $(OBJECTS_MERGED_gemdyn) | $(GEMDYN_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_gemdyn); mv $@_$$$$ $@
$(LIBDIR)/lib$(GEMDYN_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(GEMDYN_LIB_MERGED_NAME)_$(GEMDYN_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s lib$(GEMDYN_LIB_MERGED_NAME)_$(GEMDYN_VERSION).a $@

# $(LIBDIR)/lib$(GEMDYN_LIBS_SHARED_0)_$(GEMDYN_VERSION).so: $(OBJECTS_gemdyn) $(GEMDYN_LIBS_OTHER_FILES) | $(GEMDYN_VFILES)
# 	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_gemdyn)" ;\
# 	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_OTHER) $(GEMDYN_LIBS_DEP)" ;\
# 	$(RBUILD4MPI_SO)
# 	ls -l $@
# $(LIBDIR)/lib$(GEMDYN_LIBS_SHARED_0).so: $(LIBDIR)/lib$(GEMDYN_LIBS_SHARED_0)_$(GEMDYN_VERSION).so
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(GEMDYN_LIBS_SHARED_0)_$(GEMDYN_VERSION).so $@
# 	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$gemdyn/include/Makefile.local.gemdyn.mk [END] =====================)
endif
