## ====================================================================
## File: $gemdyn/include/Makefile.local.mk
##

## GEM model and GemDyn definitions
BINDIR    = $(PWD)
# PROF       = prof_003
# CPLLIBPATH = /users/dor/armn/mod/cpl/v_$(CPL_VERSION)/lib/$(EC_ARCH)
# LIBPATHPOST  = $(CHMLIBPATH)/lib/$(EC_ARCH) $(CHMLIBPATH)/$(EC_ARCH) $(CHMLIBPATH) $(CPLLIBPATH)/lib/$(EC_ARCH) $(CPLLIBPATH) $(PROFLIBPATH)
# LIBS_PRE = $(GEMLIBS) $(PHY) $(CLASSLIBS) $(CHM) $(PATCH) $(CPL)

## GEM model targets
mainntr = $(ABSPREFIX)maingemntr$(ABSPOSTFIX)_$(BASE_ARCH).Abs
maindm  = $(ABSPREFIX)maingemdm$(ABSPOSTFIX)_$(BASE_ARCH).Abs
GEM_BINLIST = $(mainntr) $(maindm)

.PHONY: gem gemntr gemdm gem_nompi gemntr_nompi gemdm_nompi allbin_gem allbincheck_gem
gem: gemntr gemdm
gem_nompi: gemntr_nompi gemdm_nompi
gemntr_nompi:
	$(MAKE) gemntr COMM_stubs=rpn_commstubs$(COMM_VERSION) MPI=
gemdm_nompi:
	$(MAKE) gemdm COMM_stubs=rpn_commstubs$(COMM_VERSION) MPI=
gemntr: $(BINDIR)/$(mainntr)
$(BINDIR)/$(mainntr): $(LIBLOCALDEP)
	export ATM_MODEL_NAME=GEMNTR ; MAINSUBNAME=gemntr ; $(RBUILD3MPI)
	ls -l $@
gemdm: $(BINDIR)/$(maindm)
$(BINDIR)/$(maindm): $(LIBLOCALDEP)
	export ATM_MODEL_NAME=GEMDM ;  MAINSUBNAME=gemdm ; $(RBUILD3MPI)
	ls -l $@

allbin_gem: 
	status=0 ;\
	for item in $(GEM_BINLIST); do \
		$(MAKE) $(BINDIR)/$${item} || status=1 ;\
	done ;\
	exit $${status}

allbincheck_gem:
	status=0 ;\
	for item in $(GEM_BINLIST); do \
		if [[ ! -f $(BINDIR)/$${item} ]] ; then \
			echo "$@: Missing $${item} in $(BINDIR)" ;\
			status=1 ;\
		fi ;\
	done ;\
	exit $${status}


## GemDyn Targets
GEMDYN_BINLIST = gemprnml_$(BASE_ARCH).Abs gemgrid_$(BASE_ARCH).Abs checkdmpart_$(BASE_ARCH).Abs split3df_$(BASE_ARCH).Abs toc2nml gem_monitor_end gem_monitor_output
.PHONY: prgemnml gemgrid checkdmpart split3df toc2nml gem_monitor_end gem_monitor_output allbin_gemdyn allbincheck_gemdyn

prgemnml: $(BINDIR)/gemprnml_$(BASE_ARCH).Abs
$(BINDIR)/gemprnml_$(BASE_ARCH).Abs: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=prgemnml ;  MAINSUBNAME=prgemnml ; $(RBUILD3NOMPI)
	ls -l $@

gemgrid: $(BINDIR)/gemgrid_$(BASE_ARCH).Abs
$(BINDIR)/gemgrid_$(BASE_ARCH).Abs: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=gemgrid ;  MAINSUBNAME=gemgrid ; COMM_stubs1=rpn_commstubs ; $(RBUILD3NOMPI)
	ls -l $@

checkdmpart: $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs
$(BINDIR)/checkdmpart_$(BASE_ARCH).Abs: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=checkdmpart ; MAINSUBNAME=checkdmpart ; $(RBUILD3MPI)
	ls -l $@

split3df: $(BINDIR)/split3df_$(BASE_ARCH).Abs
$(BINDIR)/split3df_$(BASE_ARCH).Abs: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=split3df ;  MAINSUBNAME=split3df ; $(RBUILD3MPI)
	ls -l $@

toc2nml: $(BINDIR)/toc2nml
$(BINDIR)/toc2nml: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=toc2nml ;  MAINSUBNAME=toc2nml ; $(RBUILD3NOMPI)
	ls -l $@

gem_monitor_end: $(BINDIR)/gem_monitor_end
gem_monitor_output: $(BINDIR)/gem_monitor_output
$(BINDIR)/gem_monitor_end: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=gem_monitor_end ; MAINSUBNAME=gem_monitor_end ; $(RBUILD3NOMPI_C)
	ls -l $@

$(BINDIR)/gem_monitor_output: $(LIBLOCALDEP)
	export ATM_MODEL_NAME=gem_monitor_output ; MAINSUBNAME=gem_monitor_output ; $(RBUILD3NOMPI_C)
	ls -l $@

allbin_gemdyn: | allbin_gem
	status=0 ;\
	for item in $(GEMDYN_BINLIST); do \
		$(MAKE) $(BINDIR)/$${item} || status=1 ;\
	done ;\
	exit $${status}

allbincheck_gemdyn: | allbincheck_gem
	status=0 ;\
	for item in $(GEMDYN_BINLIST); do \
		if [[ ! -f $(BINDIR)/$${item} ]] ; then \
			echo "$@: Missing $${item} in $(BINDIR)" ;\
			status=1 ;\
		fi ;\
	done ;\
	exit $${status}

clean_gemdyn:
	cd gemdyn ;
	chmod -R u+w . 2> /dev/null || true
	for mydir in `find . -type d` ; do \
		for ext in $(INCSUFFIXES) $(SRCSUFFIXES) .o .[mM][oO][dD]; do \
			rm -f $${mydir}/*$${ext} 2>/dev/null ;\
		done ;\
	done

distclean_gemdyn: | clean_gemdyn
	for item in $(GEMDYN_BINLIST); do \
		rm -f $(BINDIR)/$${item} ;\
	done

## ====================================================================
