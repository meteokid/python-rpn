## ====================================================================
## File: $gemdyn/include/Makefile.local.mk
##

## GEM model and GemDyn definitions

ifeq (,$(wildcard $(gemdyn)/VERSION))
   $(error Not found: $(gemdyn)/VERSION)
endif
GEMDYN_VERSION = $(shell cat $(gemdyn)/VERSION)

## Some Shortcut/Alias to Lib Names
#GEMDYNLIBS     = gemdyn$(GEMDYN_VERSION)
#GEMLIBS        = $(GEMDYNLIBS)

LIBCPLPATH  = 
#LIBCPL      = gemdyn_cpl_stubs
LIBCPL      =

GEMDYN_LIBS_MERGED = gemdyn_main gemdyn_base gemdyn_adw
GEMDYN_LIBS_OTHER  = $(LIBCPL)
GEMDYN_LIBS_ALL    = $(GEMDYN_LIBS_MERGED) $(GEMDYN_LIBS_OTHER)
GEMDYN_LIBS        = gemdyn $(GEMDYN_LIBS_OTHER) 
GEMDYN_LIBS_V      = gemdyn_$(GEMDYN_VERSION) $(GEMDYN_LIBS_OTHER) 

GEMDYN_LIBS_ALL_FILES = $(foreach item,$(GEMDYN_LIBS_ALL),lib$(item).a)
GEMDYN_LIBS_ALL_FILES_PLUS = libgemdyn.a $(GEMDYN_LIBS_ALL_FILES) 

OBJECTS_MERGED_gemdyn = $(foreach item,$(GEMDYN_LIBS_MERGED),$(OBJECTS_$(item)))
#OBJECTS_gemdyn_cpl_stubs      = cpl_stubs.o

GEMDYN_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_gemdyn),$(item).[Mm][Oo][Dd])

GEM_ABS       = gemntr gemdm
GEM_ABS_FILES = $(BINDIR)/$(mainntr) $(BINDIR)/$(maindm)

GEMDYN_ABS        = gem_monitor_end gem_monitor_output toc2nml gemgrid checkdmpart prgemnml split3df $(GEM_ABS)
GEMDYN_ABS_FILES  = $(BINDIR)/gem_monitor_output $(BINDIR)/gem_monitor_end $(BINDIR)/toc2nml $(BINDIR)/gemgrid_$(BASE_ARCH).Abs $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs $(BINDIR)/gemprnml_$(BASE_ARCH).Abs $(BINDIR)/split3df_$(BASE_ARCH).Abs $(BINDIR)/$(mainntr) $(BINDIR)/$(maindm) $(GEM_ABS_FILES)

## GEM model Libpath and libs
MODEL3_LIBAPPL = $(GEMDYN_LIBS_V)
MODEL3_LIBPATH = $(LIBCPLPATH)


##
.PHONY: gem_vfiles gemdyn_vfiles
GEM_VFILES = gem_version.inc gem_version.h
gem_vfiles: $(GEM_VFILES)
GEMDYN_VFILES = gemdyn_version.inc gemdyn_version.h $(GEM_VFILES)
gemdyn_vfiles: $(GEMDYN_VFILES)
gem_version.inc:
	.rdemkversionfile "gem" "$(GEMDYN_VERSION)" . f
gem_version.h:
	.rdemkversionfile "gem" "$(GEMDYN_VERSION)" . c
gemdyn_version.inc:
	.rdemkversionfile "gemdyn" "$(GEMDYN_VERSION)" . f
gemdyn_version.h:
	.rdemkversionfile "gemdyn" "$(GEMDYN_VERSION)" . c

#---- Abs targets -----------------------------------------------------

## GEM model targets (modelutils/gemdyn/rpnphy)
.PHONY: gem gemntr gemdm gem_nompi gemntr_nompi gemdm_nompi allbin_gem allbincheck_gem

allbin_gem: $(GEM_ABS)
	ls -l $(GEM_ABS_FILES)
allbincheck_gem:
	for item in $(GEM_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

gem: | gemntr gemdm

mainntr = $(ABSPREFIX)maingemntr$(ABSPOSTFIX)_$(BASE_ARCH).Abs
mainntr_rel = $(ABSPREFIX)maingemntr_REL_$(BASE_ARCH).Abs
gemntr: | gemntr_rm $(BINDIR)/$(mainntr)
	ls -l $(BINDIR)/$(mainntr)
gemntr_rm:
	rm -f $(BINDIR)/$(mainntr)
$(BINDIR)/$(mainntr): | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemntr" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	$(RBUILD4objMPI)

maindm  = $(ABSPREFIX)maingemdm$(ABSPOSTFIX)_$(BASE_ARCH).Abs
maindm_rel  = $(ABSPREFIX)maingemdm_REL_$(BASE_ARCH).Abs
gemdm: | gemdm_rm $(BINDIR)/$(maindm)
	ls -l $(BINDIR)/$(maindm)
gemdm_rm:
	rm -f $(BINDIR)/$(maindm)
$(BINDIR)/$(maindm): | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemdm" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	$(RBUILD4objMPI)

gem_nompi: | gemntr_nompi gemdm_nompi

gemntr_nompi: | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemntr" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export RBUILD_COMM_STUBS=rpn_commstubs$(COMM_VERSION) ;\
	$(RBUILD4objNOMPI)

gemdm_nompi: | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemdm" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export RBUILD_COMM_STUBS=rpn_commstubs$(COMM_VERSION) ;\
	$(RBUILD4objMPI)


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
	export RBUILD_COMM_STUBS=rpn_commstubs$(COMM_VERSION) ;\
	$(RBUILD4objNOMPI)

maingemgrid=gemgrid_$(BASE_ARCH).Abs
gemgrid: | gemgrid_rm $(BINDIR)/$(maingemgrid)
	ls -l $(BINDIR)/$(maingemgrid)
gemgrid_rm:
	rm -f $(BINDIR)/$(maingemgrid)
$(BINDIR)/$(maingemgrid): | $(GEMDYN_VFILES)
	export MAINSUBNAME="gemgrid" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export RBUILD_COMM_STUBS=rpn_commstubs$(COMM_VERSION) ;\
	$(RBUILD4objNOMPI)

maincheckdmpart=checkdmpart_$(BASE_ARCH).Abs
checkdmpart: | checkdmpart_rm $(BINDIR)/$(maincheckdmpart)
	ls -lL $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs
checkdmpart_rm:
	rm -f $(BINDIR)/$(maincheckdmpart)
$(BINDIR)/$(maincheckdmpart): | $(GEMDYN_VFILES)
	export MAINSUBNAME="checkdmpart" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	$(RBUILD4objMPI)

mainsplit3df=split3df_$(BASE_ARCH).Abs
split3df: | split3df_rm $(BINDIR)/$(mainsplit3df)
	ls -lL $(BINDIR)/$(mainsplit3df)
split3df_rm:
	rm -f $(BINDIR)/$(mainsplit3df)
$(BINDIR)/$(mainsplit3df): | $(GEMDYN_VFILES)
	export MAINSUBNAME="split3df" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	$(RBUILD4objMPI)


toc2nml: | toc2nml_rm $(BINDIR)/toc2nml
	ls -lL $(BINDIR)/toc2nml
toc2nml_rm:
	rm -f $(BINDIR)/toc2nml
$(BINDIR)/toc2nml: | $(GEMDYN_VFILES)
	export MAINSUBNAME="toc2nml" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export RBUILD_COMM_STUBS=rpn_commstubs$(COMM_VERSION) ;\
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

#----  SSM build Support ------------------------------------------------

GEMDYN_SSMALL_NAME  = gemdyn_$(GEMDYN_VERSION)_all
GEMDYN_SSMARCH_NAME = gemdyn+$(COMP_ARCH)_$(GEMDYN_VERSION)_$(SSMARCH)
GEMDYN_SSMALL_FILES  = $(GEMDYN_SSMALL_NAME).ssm
GEMDYN_SSMARCH_FILES = $(GEMDYN_SSMARCH_NAME).ssm

.PHONY: gemdyn_ssm gemdyn_ssm_all.ssm rm_gemdyn_ssm_all.ssm gemdyn_ssm_all rm_gemdyn_ssm_all gemdyn_ssm_arch.ssm rm_gemdyn_ssm_arch.ssm gemdyn_ssm_arch gemdyn_ssm_arch_rm
gemdyn_ssm: gemdyn_ssm_all.ssm gemdyn_ssm_arch.ssm
rm_gemdyn_ssm: rm_gemdyn_ssm_all.ssm rm_gemdyn_ssm_all rm_gemdyn_ssm_arch.ssm gemdyn_ssm_arch_rm

gemdyn_ssm_all.ssm: $(GEMDYN_SSMALL_FILES)
$(GEMDYN_SSMALL_FILES): gemdyn_ssm_all rm_gemdyn_ssm_all.ssm $(SSMDEPOTDIR)/$(GEMDYN_SSMALL_NAME).ssm
rm_gemdyn_ssm_all.ssm:
	rm -f $(SSMDEPOTDIR)/$(GEMDYN_SSMALL_NAME).ssm
$(SSMDEPOTDIR)/$(GEMDYN_SSMALL_NAME).ssm:
	cd $(BUILD) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gemdyn_ssm_all: rm_gemdyn_ssm_all $(BUILD)/$(GEMDYN_SSMALL_NAME)
rm_gemdyn_ssm_all:
	rm -rf $(BUILD)/$(GEMDYN_SSMALL_NAME)
$(BUILD)/$(GEMDYN_SSMALL_NAME):
	rm -rf $@ ; mkdir -p $@ ; \
	rsync -av --exclude-from=$(DIRORIG_gemdyn)/.ssm.d/exclude $(DIRORIG_gemdyn)/ $@/ ; \
	echo "Dependencies (s.ssmuse.dot): " > $@/BUILDINFO ; \
	cat $@/ssmusedep.bndl >> $@/BUILDINFO ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) "all ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control

gemdyn_ssm_arch.ssm: $(GEMDYN_SSMARCH_FILES)
$(GEMDYN_SSMARCH_FILES): gemdyn_ssm_arch rm_gemdyn_ssm_arch.ssm $(SSMDEPOTDIR)/$(GEMDYN_SSMARCH_NAME).ssm
rm_gemdyn_ssm_arch.ssm:
	rm -f $(SSMDEPOTDIR)/$(GEMDYN_SSMARCH_NAME).ssm
$(SSMDEPOTDIR)/$(GEMDYN_SSMARCH_NAME).ssm:
	cd $(BUILD) ; tar czvf $@ $(basename $(notdir $@))
	ls -l $@

gemdyn_ssm_arch: gemdyn_ssm_arch_rm $(BUILD)/$(GEMDYN_SSMARCH_NAME)
gemdyn_ssm_arch_rm:
	rm -rf $(BUILD)/$(GEMDYN_SSMARCH_NAME)
$(BUILD)/$(GEMDYN_SSMARCH_NAME):
	mkdir -p $@/include/$(EC_ARCH) $@/lib/$(EC_ARCH) $@/bin/$(BASE_ARCH) ; \
	cp $(GEMDYN_MOD_FILES) $@/include/$(EC_ARCH) ; \
	rsync -av $(wildcard libgemdyn*.a libgemdyn*.a.fl libgemdyn*.so) $@/lib/$(EC_ARCH)/ ; \
	.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile gemdyn $(GEMDYN_VERSION) $@/include/$(EC_ARCH) sh ; \
	.rdemkversionfile gem $(GEMDYN_VERSION) $@/include/$(EC_ARCH) f ; \
	.rdemkversionfile gem $(GEMDYN_VERSION) $@/include/$(EC_ARCH) c ; \
	.rdemkversionfile gem $(GEMDYN_VERSION) $@/include/$(EC_ARCH) sh ; \
	cd $(BINDIR) ; \
	cp $(GEMDYN_ABS_FILES) $@/bin/$(BASE_ARCH) ; \
	mv $@/bin/$(BASE_ARCH)/$(mainntr) $@/bin/$(BASE_ARCH)/$(mainntr_rel) ;\
	mv $@/bin/$(BASE_ARCH)/$(maindm)  $@/bin/$(BASE_ARCH)/$(maindm_rel) ;\
	cp -R $(DIRORIG_gemdyn)/.ssm.d $@/ ; \
	.rdemk_ssm_control gemdyn $(GEMDYN_VERSION) "$(SSMORDARCH) ; $(SSMARCH) ; $(BASE_ARCH)" $@/BUILDINFO $@/DESCRIPTION > $@/.ssm.d/control 

## ====================================================================
