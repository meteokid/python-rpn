ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: Makefile.user.mk)
$(info ## )
endif

## For details RDE Makefile variables see:
## https://wiki.cmc.ec.gc.ca/wiki/RDE/1.0/Ref#Makefile_Vars.2C_Rules_and_targets

# VERBOSE = 1
# OPTIL   = 2
# OMP     = -openmp
# MPI     = -mpi
# LFLAGS  =     # User's flags passed to the linker
ifneq (,$(filter intel%,$(COMP_ARCH))$(filter PrgEnv-intel%,$(COMP_ARCH)))
   FFLAGS  = -C -g -traceback -ftrapuv #-warn all -warn nointerfaces
   CFLAGS  = -C -g -traceback -ftrapuv #-fp-model precise -Wall
else
   FFLAGS  = -C -g -traceback
   CFLAGS  = -C -g -traceback
endif
# LIBAPPL = 
# LIBPATH_USER = 
# COMP_RULES_FILE = 
# PROFIL  = -prof   # set to -prof to enable the profiler (default = "")


## Optionally for scm the following options can also be modified
##
## For details SCM additional Makefile variables see:
## https://wiki.cmc.ec.gc.ca/wiki/GEM/5.0/dev'
##
## Note: MODELUTILS_COMP_RULES_DEBUG is the Compiler_rules file
##       used to produce the debug libs

# BUILDNAME          = my_exp_name
# RMN_VERSION_NUMBER = 015.2
# COMM_VERSION       = _4051606
# VGRID_VERSION      = _$(VGRIDDESCRIPTORS_VERSION)
# COMP_RULES_FILE    = $(MODELUTILS_COMP_RULES_DEBUG)


## ==== For SCM developers ==========================================
## code developpement should mandatory be done with (3 steps)

## Step 1: uncomment the following lines

# ifneq (,$(filter intel%,$(COMP_ARCH))$(filter PrgEnv-intel%,$(COMP_ARCH)))
#    FFLAGS = -C -g -traceback -ftrapuv -warn all -warn nointerfaces
#    CFLAGS = -C -g -traceback -ftrapuv -fp-model precise #-Wall
# else
#    FFLAGS = -C -g -traceback
#    CFLAGS = -C -g -traceback
# endif

## Step 2: Add the following line (uncommented) in "Makefile.user.root.mk"
# export RDE_LOCAL_LIBS_ONLY=1

## Step 3: Execute the following commands
## make dep       #mandatory
## make -j9 libs  #mandatory
## make -j9 bins

## ==== For SCM developers [END] ====================================

ifneq (,$(ATM_MODEL_USERLIBS))
ifeq (,$(COMP_RULES_FILE))
ifeq (,$(wildcard $(HOME)/userlibs/$(EC_ARCH)/Compiler_rules))
ifneq (,$(wildcard $(ATM_MODEL_USERLIBS)/$(EC_ARCH)/Compiler_rules))
COMP_RULES_FILE = $(ATM_MODEL_USERLIBS)/$(EC_ARCH)/Compiler_rules
endif
endif
endif
endif

## Sample MPI abs targets

mympiabsname: | mympiabsname_rm $(BINDIR)/mympiabsname
	ls -l $(BINDIR)/mympiabsname
mympiabsname_rm:
	if [[ "x$(SCM_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/mympiabsname ; \
	fi
$(BINDIR)/mympiabsname: $(SCM_ABS_DEP) | $(SCM_VFILES)
	export MAINSUBNAME="my_main_sub_name" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(SCM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(SCMDYN_LIBS_V) $(SCMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)
	ls $@

## Sample no-MPI abs targets

myabsname: | myabsname_rm $(BINDIR)/myabsname
	ls -l $(BINDIR)/myabsname
myabsname_rm:
	if [[ "x$(SCM_ABS_DEP)" == "x" ]] ; then \
		rm -f $(BINDIR)/myabsname ; \
	fi
$(BINDIR)/myabsname: $(SCM_ABS_DEP) | $(SCM_VFILES)
	export MAINSUBNAME="my_main_sub_name" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(SCM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(SCMDYN_LIBS_V) $(SCMDYN_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS="$(LIBCOMM_STUBS) $(MODELUTILS_DUMMYMPISTUBS)";\
	$(RBUILD4objNOMPI)
	ls $@


ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
