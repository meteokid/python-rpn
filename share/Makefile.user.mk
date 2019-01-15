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
# ifneq (,$(filter intel%,$(COMP_ARCH))$(filter PrgEnv-intel%,$(COMP_ARCH)))
# FFLAGS  = -C -g -traceback -ftrapuv #-warn all
# CFLAGS  = -C -g -traceback -ftrapuv -fp-model precise #-Wall
# else
# FFLAGS  = -C -g -traceback
# CFLAGS  = -C -g -traceback 
# endif
# LIBAPPL = 
# LIBPATH_USER = 
# COMP_RULES_FILE = 
# PROFIL  = -prof   # set to -prof to enable the profiler (default = "")


## Optionally for gem the following options can also be modified
##
## For details GEM additional Makefile variables see:
## https://wiki.cmc.ec.gc.ca/wiki/GEM/4.8/dev'
## 
## Note: MODELUTILS_COMP_RULES_DEBUG is the Compiler_rules file
##       used to produce the debug libs

# BUILDNAME          = my_exp_name
# RMN_VERSION_NUMBER = 015.2
# COMM_VERSION       = _4051606
# VGRID_VERSION      = _$(VGRIDDESCRIPTORS_VERSION)
# COMP_RULES_FILE    = $(MODELUTILS_COMP_RULES_DEBUG)


## For GEM developpers:
## code developpement should mandatory be done with
## the following options: (uncomment the lines below)

# FFLAGS     = -C -g -traceback
# MODELUTILS_SFX = -d
# RPNPHY_SFX = -d
# GEMDYN_SFX = -d


## To build with a local version of all libraries
## 
## Remove files (include or w/t module/sub/function) with: 
## rderm filename.ext
## do NOT create a stopping stub (prefer to catch at load time than run time)
##
## make dep         #mandatory
## make -j objects  #mandatory
## make modelutils_libs rpnphy_libs gemdyn_libs  #mandatory
## make allbin_gem # allbin_modelutils allbin_rpnphy allbin_gemdyn 

# MODELUTILS_VERSION=$(USER)
# RPNPHY_VERSION=$(USER)
# GEMDYN_VERSION=$(USER)

ifneq (,$(ATM_MODEL_USERLIBS))
ifeq (,$(COMP_RULES_FILE))
ifeq (,$(wildcard $(HOME)/userlibs/$(EC_ARCH)/Compiler_rules))
ifneq (,$(wildcard $(ATM_MODEL_USERLIBS)/$(EC_ARCH)/Compiler_rules))
COMP_RULES_FILE = $(ATM_MODEL_USERLIBS)/$(EC_ARCH)/Compiler_rules
endif
endif
endif
endif

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
