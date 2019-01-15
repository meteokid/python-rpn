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

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
