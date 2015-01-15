## ====================================================================
## File: $gemdyn/include/Makefile.local.mk
##

#include $(modelutils)/include/recettes
#include $(gemdyn)/include/build_components

## GEM model and GemDyn definitions

mainntr = $(ABSPREFIX)maingemntr$(ABSPOSTFIX)_$(BASE_ARCH).Abs
maindm  = $(ABSPREFIX)maingemdm$(ABSPOSTFIX)_$(BASE_ARCH).Abs

MONBINDIR = $(PWD)
BINDIR    = $(MONBINDIR)


## Some Shortcut/Alias to Lib Names
GEMDYN_VERSION = 
GEMDYNLIBS  = gemdyn$(GEMDYN_VERSION)
GEMLIBS     = $(GEMDYNLIBS)

OBJECTS_gemdyn = $(OBJECTS_gemdyn_base) $(OBJECTS_gemdyn_adw)
OBJECTS_libgemdyn_cpl_stubs = cpl_stubs.o
OBJECTS_libgemdyn_itf_cpl_stubs = itf_cpl_stubs.o
OBJECTS_libgemdyn_cpl_prof_stubs = prof_stubs.o
GEMDYN_LIBS = gemdyn_main gemdyn gemdyn_itf_cpl_stubs gemdyn_cpl_stubs gemdyn_prof_stubs
GEMDYN_LIBS_V = $(foreach item,$(GEMDYN_LIBS),$(item)_$(GEMDYN_VERSION).a)
GEMDYN_LIBS_FILES = $(foreach item,$(GEMDYN_LIBS),lib$(item).a)
GEMDYN_MOD_FILES = $(foreach item,$(FORTRAN_MODULES_gemdyn),$(item).[Mm][Oo][Dd])
GEMDYN_ABS = gem_monitor_end gem_monitor_output toc2nml gemgrid checkdmpart prgemnml split3df
GEMDYN_ABS_FILES = $(BINDIR)/gem_monitor_output $(BINDIR)/gem_monitor_end $(BINDIR)/toc2nml $(BINDIR)/gemgrid_$(BASE_ARCH).Abs $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs $(BINDIR)/gemprnml_$(BASE_ARCH).Abs $(BINDIR)/split3df_$(BASE_ARCH).Abs  

GEM_ABS = gemntr gemdm
GEM_ABS_FILES = $(BINDIR)/$(mainntr) $(BINDIR)/$(maindm)

CHMLIBPATH  = 
CHM         = $(CHM_VERSION) $(CHMLIBS)
#CHMLIBPATH  = $(ARMNLIB)/modeles/CHM/v_$(CHM_VERSION)
#CHM         = chm

#PHY         = rpnphy_stubs
#CLASSLIBS   = rpnphy_class
#PHYSURFACELIBS = rpnphy_surface
#PHYCONVECTLIBS = rpnphy_convect
#CHMLIBS = rpnphy_chm_stubs
PHY         = $(PHYLIBS)

#PROF          = prof_stubs
PROF          = prof_003
#PROFLIBPATH   = $(ARMNLIB)/lib/$(BASE_ARCH)
PROFLIBPATH   =

CPL         = cpl_stubs itf_cpl_stubs
#CPL        = cpl_$(CPL_VERSION)
CPLLIBPATH = /users/dor/armn/mod/cpl/v_$(CPL_VERSION)/lib/$(EC_ARCH)


## GEM model Libpath and libs
#LIBPATH = $(PWD) $(LIBPATH_PRE) $(LIBPATHEXTRA) $(LIBSYSPATHEXTRA) $(LIBPATHOTHER) $(LIBPATH_POST)
LIBPATHPOST  = $(CHMLIBPATH)/lib/$(EC_ARCH) $(CHMLIBPATH)/$(EC_ARCH) $(CHMLIBPATH) $(CPLLIBPATH)/lib/$(EC_ARCH) $(CPLLIBPATH) $(PROFLIBPATH)

#OTHERS  = $(COMM) $(VGRID) $(UTIL) $(LLAPI) $(IBM_LD)
#LIBAPPL = $(LIBS_PRE) $(MODELUTILSLIBS) $(OTHERS) $(LIBS_POST)
#LIBSYS  = $(LIBSYS_PRE) $(LIBSYSEXTRA) $(LIBSYS_POST)
LIBS_PRE = $(GEMLIBS) $(PHY) $(CLASSLIBS) $(CHM) $(PATCH) $(CPL)

#LIBS     = $(LIBAPPL)


## GEM model targets (modelutils/gemdyn/rpnphy)
.PHONY: gem gemntr gemdm gem_nompi gemntr_nompi gemdm_nompi allbin_gem allbincheck_gem
gem: $(GEM_ABS)

allbin_gem: $(GEM_ABS_FILES)
	ls -l $(GEM_ABS_FILES)
allbincheck_gem:
	for item in $(GEM_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

gem_nompi: gemntr_nompi gemdm_nompi

gemntr_nompi:
	$(MAKE) gemntr COMM_stubs=rpn_commstubs$(COMM_VERSION) MPI=

gemdm_nompi:
	$(MAKE) gemdm COMM_stubs=rpn_commstubs$(COMM_VERSION) MPI=

$(BINDIR)/$(mainntr): gemntr
	if [[ -r $(mainntr) ]] ; then cp $(mainntr) $@ 2>/dev/null ; fi
gemntr:
	export ATM_MODEL_NAME="GEMNTR" ; $(RBUILD2Oa) ;\
	if [[ x$(BINDIR) == x ]] ; then \
		cp $@.Abs $(PWD)/$(mainntr) 2>/dev/null || true ;\
		chmod u+x $(PWD)/$(mainntr) ;\
		ls -lL $(PWD)/$(mainntr) ;\
	else \
		cp $@.Abs $(BINDIR)/$(mainntr) 2>/dev/null || true ;\
		chmod u+x $(BINDIR)/$(mainntr) ;\
		ls -lL $(BINDIR)/$(mainntr) ;\
	fi

$(BINDIR)/$(maindm): gemdm
	if [[ -r $(maindm) ]] ; then cp $(maindm) $@ 2>/dev/null ; fi
gemdm:
	export ATM_MODEL_NAME="GEMDM" ; $(RBUILD2Oa) ;\
	if [[ x$(BINDIR) == x ]] ; then \
		cp $@.Abs $(PWD)/$(maindm) || true ;\
		chmod u+x $(PWD)/$(maindm) ;\
		ls -lL $(PWD)/$(maindm) ;\
	else \
		cp $@.Abs $(BINDIR)/$(maindm) 2>/dev/null || true ;\
		chmod u+x $(BINDIR)/$(maindm) ;\
		ls -lL $(BINDIR)/$(maindm) ;\
	fi


## GemDyn Targets
.PHONY: libdyn
libdyn: rmpo $(OBJECTS)

mainprgemnml=gemprnml_$(BASE_ARCH).Abs
.PHONY: prgemnml gemgrid checkdmpart split3df toc2nml monitor sometools allbin allbincheck
# prgemnml: $(BINDIR)/gemprnml_$(BASE_ARCH).Abs
# 	ls -lL $(BINDIR)/gemprnml_$(BASE_ARCH).Abs
# prgemnml.ftn90: 
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
# $(LCLPO)/prgemnml.o: prgemnml.ftn90
# $(BINDIR)/gemprnml_$(BASE_ARCH).Abs: $(LCLPO)/prgemnml.o
prgemnml: 
	export ATM_MODEL_NAME="prgemnml" ;\
	makemodelbidon prgemnml > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	cd $(LCLPO) ;\
	$(RBUILD) -obj *.o -o $(mainprgemnml) $(OMP) \
		-libpath $(PWD) $(LIBPATH) \
		-libappl "gemdyn_main gemdyn $(MODELUTILSLIBS) $(OTHERS)" \
		-librmn $(RMN_VERSION) \
		-libsys $(LIBSYS) \
		-codebeta $(CODEBETA)  \
		-optf "=$(LFLAGS)" ;\
	/bin/mv $(mainprgemnml) $(BINDIR)/$(mainprgemnml) ;\
	 /bin/rm -f $(mainprgemnml)_* bidon.o


# gemgrid: $(BINDIR)/gemgrid_$(BASE_ARCH).Abs
# 	ls -lL $(BINDIR)/gemgrid_$(BASE_ARCH).Abs
# gemgrid.ftn90: 
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
# $(LCLPO)/gemgrid.o: gemgrid.ftn90
# $(BINDIR)/gemgrid_$(BASE_ARCH).Abs: $(LCLPO)/gemgrid.o
gemgrid:
	export ATM_MODEL_NAME="gemgrid" ;\
	makemodelbidon gemgrid > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	cd $(LCLPO) ;\
	$(RBUILD) -obj *.o -o gemgrid_$(BASE_ARCH).Abs $(OMP) \
		-libpath $(PWD) $(LIBPATH) \
		-libappl "gemdyn_main gemdyn $(MODELUTILSLIBS) $(OTHERS) rpn_commstubs$(COMM_VERSION)" \
		-librmn $(RMN_VERSION) 
		-libsys $(LIBSYS) \
		-codebeta $(CODEBETA) \
		-optf "=$(LFLAGS)" ;\
	/bin/mv gemgrid_$(BASE_ARCH).Abs $(BINDIR)/gemgrid_$(BASE_ARCH).Abs ;\
	/bin/rm -f $(LCLPO)/bidon.o $(LCLPO)/gemgrid.o 2>/dev/null || true 


checkdmpart: $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs
	ls -lL $(BINDIR)/checkdmpart_$(BASE_ARCH).Abs
# checkdmpart.ftn90: 
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
# $(LCLPO)/checkdmpart.o: checkdmpart.ftn90
$(BINDIR)/checkdmpart_$(BASE_ARCH).Abs: #$(LCLPO)/checkdmpart.o
	export ATM_MODEL_NAME="checkdmpart" ; makemodelbidon checkdmpart > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	cd $(LCLPO) ;\
	$(RBUILD) -obj *.o -o $@ $(OMP) $(MPI) \
		-libpath $(PWD) $(LIBPATH) \
		-libappl gemdyn_main $(LIBAPPL) \
		-librmn $(RMN_VERSION) \
		-libsys $(LIBSYS) \
		-codebeta $(CODEBETA) \
		-optf "=$(LFLAGS)"
	/bin/rm -f $(LCLPO)/bidon.o $(LCLPO)/checkdmpart.o 2>/dev/null || true 


split3df: $(BINDIR)/split3df_$(BASE_ARCH).Abs
	ls -lL $(BINDIR)/split3df_$(BASE_ARCH).Abs
# split3df.cdk90: 
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
# $(LCLPO)/split3df.o: split3df.cdk90
$(BINDIR)/split3df_$(BASE_ARCH).Abs: #$(LCLPO)/split3df.o
	export ATM_MODEL_NAME="split3df" ; makemodelbidon split3df > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	cd $(LCLPO) ;\
	$(RBUILD) -obj *.o -o $@ $(OMP) -mpi \
		-libpath $(PWD) $(LIBPATH) \
		-libappl gemdyn_main $(LIBAPPL) \
		-libsys $(LIBSYS) \
		-optf "=$(LFLAGS)"
	/bin/rm -f $(LCLPO)/bidon.o $(LCLPO)/split3df.o 2>/dev/null || true 


toc2nml: $(BINDIR)/toc2nml
	ls -lL $(BINDIR)/toc2nml
# toc2nml.cdk90: 
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ 2>/dev/null || true ; fi ;\
# 	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ 2>/dev/null || true ; fi
# $(LCLPO)/toc2nml.o: toc2nml.cdk90
$(BINDIR)/toc2nml: #$(LCLPO)/toc2nml.o
	export ATM_MODEL_NAME="toc2nml" ; makemodelbidon toc2nml > bidon.f90 ; $(MAKE) bidon.o ; rm -f bidon.f90 ;\
	cd $(LCLPO) ;\
	$(RBUILD) -obj *.o -o $@ $(OMP) \
		-libpath $(PWD) $(LIBPATH) \
		-libappl "gemdyn_main $(VGRID) $(UTIL)" \
		-librmn $(RMN_VERSION) \
		-libsys $(LIBSYS) \
		-codebeta $(CODEBETA) \
		-optf "=$(LFLAGS)" 
	/bin/rm -f $(LCLPO)/bidon.o 2>/dev/null || true 

monitor: $(BINDIR)/gem_monitor_end $(BINDIR)/gem_monitor_output

gem_monitor_end: $(BINDIR)/gem_monitor_end
	ls -lL $(BINDIR)/$@
gem_monitor_end.c: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_end: gem_monitor_end.c
	$(MAKE) gem_monitor_end.o
	$(RBUILD) $(OMP) -obj gem_monitor_end.o -o $@ -conly
	rm -f gem_monitor_end.o 2>/dev/null || true

gem_monitor_output: $(BINDIR)/gem_monitor_output
	ls -lL $(BINDIR)/$@
gem_monitor_output.c: 
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/src/main/$@ $@ || true ; fi
	if [[ ! -f $@ ]] ; then cp $(gemdyn)/main/$@ $@ || true ; fi
$(BINDIR)/gem_monitor_output: gem_monitor_output.c
	$(MAKE) gem_monitor_output.o
	$(RBUILD) $(OMP) -obj gem_monitor_output.o -o $@ -conly
	rm -f gem_monitor_output.o 2>/dev/null || true

sometools: prgemnml gemgrid toc2nml

allbin_gemdyn: $(GEMDYN_ABS)
#allbin_gemdyn:  toc2nml gemgrid checkdmpart prgemnml split3df #gemabs
#allbin_gemdyn: monitor toc2nml gemgrid prgemnml split3df #gemabs


allbincheck_gemdyn:
	for item in $(GEMDYN_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0


## ====================================================================
