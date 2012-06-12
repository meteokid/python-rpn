## ====================================================================
## File: $gemdyn/include/Makefile.local.mk
##

include $(modelutils)/include/recettes
include $(gemdyn)/include/build_components

## GEM model and GemDyn definitions

mainntr = $(ABSPREFIX)maingemntr$(ABSPOSTFIX)_$(BASE_ARCH).Abs
maindm  = $(ABSPREFIX)maingemdm$(ABSPOSTFIX)_$(BASE_ARCH).Abs

MONBINDIR = $(PWD)
BINDIR    = $(MONBINDIR)


## Some Shortcut/Alias to Lib Names
GEMLIBS     = gemdyn

V4D         = v4d_stubs
#V4D         = "gemdyn_v4d prof_003"

CHMLIBPATH  = 
CHM         = $(CHM_VERSION)
#CHMLIBPATH  = $(ARMNLIB)/modeles/CHM/v_$(CHM_VERSION)
#CHM         = chm

#PHY         = rpnphy_stubs
PHY         = rpnphy

#PROF          = prof_stubs
PROF          = prof_003
#PROFLIBPATH   = $(ARMNLIB)/lib/$(BASE_ARCH)
PROFLIBPATH   =

#CPL         = cpl_stubs
CPL        = cpl_$(CPL_VERSION)
CPLLIBPATH = /users/dor/armn/mod/cpl/v_$(CPL_VERSION)


## GEM model Libpath and libs
#LIBPATH = $(PWD) $(LIBPATH_PRE) $(LIBPATHEXTRA) $(LIBSYSPATHEXTRA) $(LIBPATHOTHER) $(LIBPATH_POST)
LIBPATHPOST  = $(CHMLIBPATH)/lib/$(EC_ARCH) $(CPLLIBPATH)/lib/$(EC_ARCH) $(PROFLIBPATH)

#OTHERS  = $(COMM) $(VGRID) $(UTIL) $(LLAPI) $(IBM_LD)
#LIBAPPL = $(LIBS_PRE) $(MODELUTILSLIBS) $(OTHERS) $(LIBS_POST)
#LIBSYS  = $(LIBSYS_PRE) $(LIBSYSEXTRA) $(LIBSYS_POST)
LIBS_PRE = $(LIBSGEM) $(V4D) $(PHY) $(CHM) $(PATCH) $(CPL)

#LIBS     = $(LIBAPPL)


## GEM model targets (modelutils/gemdyn/rpnphy)
.PHONY: gem gemntr gemdm gem4d gem_nompi gemntr_nompi gemdm_nompi allbin_gem allbincheck_gem
gem: gemntr gemdm

allbin_gem: $(BINDIR)/$(mainntr) $(BINDIR)/$(maindm)
	ls -l $(BINDIR)/$(mainntr) $(BINDIR)/$(maindm)
allbincheck_gem:
	if [[ \
		&& -f $(BINDIR)/$(mainntr) \
		&& -f $(BINDIR)/$(maindm) \
		]] ; then \
		exit 0 ;\
	fi ;\
	exit 1

gem4d:
	$(MAKE) gem V4D="gemdyn_v4d $(PROF)" PROFLIBPATH=$(ARMNLIB)/lib/$(BASE_ARCH)

gem_nompi: gemntr_nompi gemdm_nompi

gemntr_nompi:
	$(MAKE) gemntr COMM_stubs=rpn_commstubs$(COMM_VERSION) MPI=

gemdm_nompi:
	$(MAKE) gemdm COMM_stubs=rpn_commstubs$(COMM_VERSION) MPI=

$(BINDIR)/$(mainntr): gemntr
	cp $(mainntr) $@
gemntr:
	$(RBUILD2O) ;\
	mv $@ $(PWD)/$(mainntr) 2>/dev/null || true ;\
	chmod u+x $(PWD)/$(mainntr) ;\
	ls -lL $(PWD)/$(mainntr) ;\
	echo DYN_VERSION   = $(ATM_DYN_VERSION);\
	echo PHY_VERSION   = $(ATM_PHY_VERSION);\
	echo CHM_VERSION   = $(CHM_VERSION);\
	echo CPL_VERSION   = $(CPL_VERSION);\
	echo VGRID_VERSION = $(VGRID_VERSION);\
	echo RMN_VERSION   = $(RMN_VERSION);\
	echo COMM_VERSION  = $(COMM_VERSION)

$(BINDIR)/$(maindm): gemdm
	cp $(maindm) $@
gemdm:
	$(RBUILD2O) ;\
	mv $@ $(PWD)/$(maindm) 2>/dev/null || true ;\
	chmod u+x $(PWD)/$(maindm) ;\
	ls -lL $(PWD)/$(maindm) ;\
	echo DYN_VERSION   = $(ATM_DYN_VERSION);\
	echo PHY_VERSION   = $(ATM_PHY_VERSION);\
	echo CHM_VERSION   = $(CHM_VERSION);\
	echo CPL_VERSION   = $(CPL_VERSION);\
	echo VGRID_VERSION = $(VGRID_VERSION);\
	echo RMN_VERSION   = $(RMN_VERSION);\
	echo COMM_VERSION  = $(COMM_VERSION)


## GemDyn Targets
.PHONY: libdyn
libdyn: rmpo $(OBJECTS)

.PHONY: prgemnml gemgrid toc2nml monitor sometools allbin allbincheck
prgemnml: $(BINDIR)/gemprnml_$(BASE_ARCH).Abs
	ls -lL $(BINDIR)/gemprnml_$(BASE_ARCH).Abs
prgemnml.ftn90: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
$(LCLPO)/prgemnml.o: prgemnml.ftn90
$(BINDIR)/gemprnml_$(BASE_ARCH).Abs: $(LCLPO)/prgemnml.o
	cd $(LCLPO) ;\
	makemodelbidon prgemnml > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	$(RBUILD) -obj prgemnml.o bidon.o -o $@ -libpath $(LIBPATH) -libappl "gemdyn_main gemdyn modelutils" -librmn $(RMN_VERSION) -libsys $(LIBSYS)
	/bin/rm -f $(LCLPO)/bidon.o 2>/dev/null || true

gemgrid: $(BINDIR)/gemgrid_$(BASE_ARCH).Abs
	ls -lL $(BINDIR)/gemgrid_$(BASE_ARCH).Abs
gemgrid.ftn90: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
$(LCLPO)/gemgrid.o: gemgrid.ftn90
$(BINDIR)/gemgrid_$(BASE_ARCH).Abs: $(LCLPO)/gemgrid.o
	cd $(LCLPO) ;\
	makemodelbidon gemgrid > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	$(RBUILD) -obj gemgrid.o bidon.o -o $@ -libpath $(LIBPATH) -libappl "gemdyn modelutils $(COMM) rpn_commstubs$(COMM_VERSION)" -librmn $(RMN_VERSION) -libsys $(LIBSYS)
	/bin/rm -f $(LCLPO)/bidon.o 2>/dev/null || true 

toc2nml: $(BINDIR)/toc2nml
	ls -lL $(BINDIR)/toc2nml
toc2nml.ftn90: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
$(LCLPO)/toc2nml.o: toc2nml.ftn90
$(BINDIR)/toc2nml: $(LCLPO)/toc2nml.o
	cd $(LCLPO) ;\
	makemodelbidon toc2nml > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	$(RBUILD) -obj toc2nml.o bidon.o -o $@ -libpath $(LIBPATH) -libappl "gemdyn_main descrip" -librmn $(RMN_VERSION) -libsys $(LIBSYS)
	/bin/rm -f $(LCLPO)/bidon.o 2>/dev/null || true 

monitor: $(BINDIR)/gem_monitor_end $(BINDIR)/gem_monitor_output
	echo BINDIR=$(BINDIR)
	ls -lL $(BINDIR)/gem_monitor_*

gem_monitor_end.c: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_end: gem_monitor_end.c
	$(MAKE) gem_monitor_end.o
	$(RBUILD) -obj gem_monitor_end.o -o $@ -conly
	rm -f gem_monitor_end.o 2>/dev/null || true

gem_monitor_output.c: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_output: gem_monitor_output.c
	$(MAKE) gem_monitor_output.o
	$(RBUILD) -obj gem_monitor_output.o -o $@ -conly
	rm -f gem_monitor_output.o 2>/dev/null || true

sometools: prgemnml gemgrid toc2nml

allbin_gemdyn: monitor toc2nml gemgrid prgemnml #gemabs

allbincheck_gemdyn:
	if [[ \
		&& -f $(BINDIR)/gemprnml_$(BASE_ARCH).Abs \
		&& -f $(BINDIR)/gemgrid_$(BASE_ARCH).Abs \
		&& -f $(BINDIR)/toc2nml \
		&& -f $(BINDIR)/gem_monitor_end \
		&& -f $(BINDIR)/gem_monitor_output \
		]] ; then \
		exit 0 ;\
	fi ;\
	exit 1

## Dependencies not handled properly by r.make_exp
out_vref_mod.o: out_vref_mod.cdk90      cstv.cdk     dimout.cdk   glb_ld.cdk \
                grd.cdk                 grid.cdk     level.cdk    lun.cdk    \
                                         out.cdk     type.cdk     ver.cdk

nest_blending.o: nest_blending.cdk90   glb_ld.cdk nest.cdk
nest_blending_ad.o: nest_blending_ad.cdk90   glb_ld.cdk nest.cdk

## ====================================================================
