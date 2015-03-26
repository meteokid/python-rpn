ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnphy/include/Makefile.local.mk)
$(info ## )
endif

## GEM model and GemDyn definitions

# ifeq (,$(wildcard $(gemdyn)/VERSION))
#    $(error Not found: $(gemdyn)/VERSION)
# endif
# GEMDYN_VERSION0  = $(shell cat $(gemdyn)/VERSION | sed 's|x/||')
GEMDYN_VERSION0  = x/4.8.a3
GEMDYN_VERSION   = $(notdir $(GEMDYN_VERSION0))
GEMDYN_VERSION_X = $(dir $(GEMDYN_VERSION0))


## Some Shortcut/Alias to Lib Names
GEMDYN_LIBS_DEP = $(RPNPHY_LIBS_V) $(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)

LIBCPLPATH  = 
LIBCPL      =

GEMDYN_LIBS_MERGED = gemdyn_main gemdyn_base gemdyn_adw
GEMDYN_LIBS_OTHER  = $(LIBCPL)
GEMDYN_LIBS_ALL    = $(GEMDYN_LIBS_MERGED) $(GEMDYN_LIBS_OTHER)
GEMDYN_LIBS        = gemdyn $(GEMDYN_LIBS_OTHER) 
GEMDYN_LIBS_V      = gemdyn_$(GEMDYN_VERSION) $(GEMDYN_LIBS_OTHER) 

GEMDYN_LIBS_ALL_FILES = $(foreach item,$(GEMDYN_LIBS_ALL),$(LIBDIR)/lib$(item).a)
GEMDYN_LIBS_ALL_FILES_PLUS = $(LIBDIR)/libgemdyn.a $(GEMDYN_LIBS_ALL_FILES) 

OBJECTS_MERGED_gemdyn = $(foreach item,$(GEMDYN_LIBS_MERGED),$(OBJECTS_$(item)))

GEMDYN_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_gemdyn),$(item).[Mm][Oo][Dd])

GEMDYN_ABS        = gem_monitor_end gem_monitor_output toc2nml gemgrid checkdmpart prgemnml split3df
GEMDYN_ABS_FILES  = $(BINDIR)/gem_monitor_output $(BINDIR)/gem_monitor_end $(BINDIR)/toc2nml $(BINDIR)/gemgrid_$(BASE_ARCH).Abs $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs $(BINDIR)/gemprnml_$(BASE_ARCH).Abs $(BINDIR)/split3df_$(BASE_ARCH).Abs

## GEM model Libpath and libs
#MODEL3_LIBAPPL = $(GEMDYN_LIBS_V)
MODEL3_LIBPATH = $(LIBCPLPATH)


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
	rm -f $(BINDIR)/$(mainprgemnml)
$(BINDIR)/$(mainprgemnml): | $(GEMDYN_VFILES)
	export MAINSUBNAME="prgemnml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

maingemgrid=gemgrid_$(BASE_ARCH).Abs
gemgrid: | gemgrid_rm $(BINDIR)/$(maingemgrid)
	ls -l $(BINDIR)/$(maingemgrid)
gemgrid_rm:
	rm -f $(BINDIR)/$(maingemgrid)
$(BINDIR)/$(maingemgrid): | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemgrid" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

maincheckdmpart=checkdmpart_$(BASE_ARCH).Abs
checkdmpart: | checkdmpart_rm $(BINDIR)/$(maincheckdmpart)
	ls -lL $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs
checkdmpart_rm:
	rm -f $(BINDIR)/$(maincheckdmpart)
$(BINDIR)/$(maincheckdmpart): | $(GEMDYN_VFILES)
	export MAINSUBNAME="checkdmpart" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)

mainsplit3df=split3df_$(BASE_ARCH).Abs
split3df: | split3df_rm $(BINDIR)/$(mainsplit3df)
	ls -lL $(BINDIR)/$(mainsplit3df)
split3df_rm:
	rm -f $(BINDIR)/$(mainsplit3df)
$(BINDIR)/$(mainsplit3df): | $(GEMDYN_VFILES)
	export MAINSUBNAME="split3df" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)


toc2nml: | toc2nml_rm $(BINDIR)/toc2nml
	ls -lL $(BINDIR)/toc2nml
toc2nml_rm:
	rm -f $(BINDIR)/toc2nml
$(BINDIR)/toc2nml: | $(GEMDYN_VFILES)
	export MAINSUBNAME="toc2nml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

gem_monitor_end: $(BINDIR)/gem_monitor_end
	ls -lL $(BINDIR)/$@
gem_monitor_end.c: 
	if [[ ! -f $@ ]] ; then rdeco $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_end: gem_monitor_end.c
	mybidon=gem_monitor_end_123456789 ;\
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

.PHONY: gemdyn_libs
gemdyn_libs: $(OBJECTS_gemdyn) $(GEMDYN_LIBS_ALL_FILES_PLUS) | $(GEMDYN_VFILES)
$(foreach item,$(GEMDYN_LIBS_ALL),$(eval $(call LIB_template1,$(item),GEMDYN)))
$(foreach item,$(GEMDYN_LIBS_ALL),$(eval $(call LIB_template2,$(item),GEMDYN)))

$(LIBDIR)/libgemdyn_$(GEMDYN_VERSION).a: $(OBJECTS_gemdyn) | $(GEMDYN_VFILES)
	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_gemdyn); mv $@_$$$$ $@
$(LIBDIR)/libgemdyn.a: $(LIBDIR)/libgemdyn_$(GEMDYN_VERSION).a
	cd $(LIBDIR) ; rm -f $@ ;\
	ln -s libgemdyn_$(GEMDYN_VERSION).a $@


ifneq (,$(DEBUGMAKE))
$(info ## ==== $$gemdyn/include/Makefile.local.mk [END] =====================)
endif
