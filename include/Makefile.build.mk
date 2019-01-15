ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: Makefile.build.mk)
$(info ## )
endif

SHELL = /bin/bash

## ==== Basic definitions

ifneq (,$(MYTIME))
   MYTIMEX = set -x ; time
endif

ifeq (,$(CONST_BUILD))
   ifneq (,$(DEBUGMAKE))
      $(info include $(MAKEFILE_CONST))
   endif
   include $(MAKEFILE_CONST)
endif

BUILD    := $(ROOT)/$(CONST_BUILD)
BUILDPRE := $(ROOT)/$(CONST_BUILDPRE)
BUILDOBJ := $(ROOT)/$(CONST_BUILDOBJ)
BUILDMOD := $(ROOT)/$(CONST_BUILDMOD)
BUILDLIB := $(ROOT)/$(CONST_BUILDLIB)
BUILDBIN := $(ROOT)/$(CONST_BUILDBIN)
BINDIR   := $(BUILDBIN)

BUILDSRC := $(ROOT)/$(CONST_BUILDSRC)

#------------------------------------------------------------------------
# WARNING: Avoid using VPATH to find source files when working only
#          with modified source files.
#          VPATH cause wrong file include when a include file is modified
#          but not the source file including it. The fortran preprocessor
#          then includes the include file located in the same dir as the
#          source file, not the modified one in the build dir.
#          Copying the source file to the build dir does not solve all problems
#          since recursive include files produce the same problem.
#          For that reason avoid using VPATH when working only with modified
#          source files. Instead allow the default make rules to checkout
#          all files related to dependencies and inverse dependencies.
ifeq (1,$(RDE_USE_FULL_VPATH))
   VPATH           := $(CONST_VPATH)
   SRCPATH_INCLUDE := $(CONST_SRCPATH_INCLUDE) $(CONST_SRCPATH)
   SRCPATH_INCLUDE_OVERRIDES := 
else
   VPATH    := $(ROOT)/$(CONST_BUILDSRC)
   SRCPATH_INCLUDE := 
   SRCPATH_INCLUDE_OVERRIDES := $(ROOT)/include/$(CONST_MAKEFILE_USER_COMPARCH) $(ROOT)/include/$(CONST_MAKEFILE_USER_BASEARCH) $(ROOT) $(ROOT)/include
endif
SRCPATH  := $(CONST_SRCPATH)
#------------------------------------------------------------------------

export RDE_EXP_ROOT := $(ROOT)
MAKEFILEDEP := $(CONST_MAKEFILE_DEP)
MAKEFILEUSERLIST := $(wildcard $(ROOT)/$(CONST_MAKEFILE_USER) $(ROOT)/$(CONST_MAKEFILE_USER_BASEARCH) $(ROOT)/$(CONST_MAKEFILE_USER_COMPARCH))

ifeq (,$(rde))
   $(error FATAL ERROR: rde is not defined)
endif
ifeq (,$(ROOT))
   $(error FATAL ERROR: ROOT is not defined)
endif
ifeq ($(ROOT),$(BUILD))
   $(error FATAL ERROR: BUILD == ROOT)
endif
# ifeq (,$(VPATH))
#    $(error FATAL ERROR: VPATH is not defined)
# endif

CPP = /lib/cpp
# ASFLAGS =
# DOC =
AR = r.ar -arch $(ARCH)

## ==== Legacy
EC_MKL = $(RDE_MKL)

# FORCE_RMN_VERSION_RC =
# #RMN_VERSION = rmn_015.2$(FORCE_RMN_VERSION_RC)
# RMN_VERSION = rmn$(FORCE_RMN_VERSION_RC)

LIBPATH = $(PWD) $(LIBPATH_PRE) $(BUILDLIB) $(LIBPATHEXTRA) $(LIBSYSPATHEXTRA) $(LIBPATHOTHER) $(LIBPATH_POST)
#LIBAPPL = $(LIBS_PRE) $(LIBLOCAL) $(LIBOTHERS) $(LIBEXTRA) $(LIBS_POST)
LIBAPPL = $(LIBS_PRE) $(LIBOTHERS) $(LIBEXTRA) $(LIBS_POST)
LIBSYS  = $(LIBSYS_PRE) $(LIBSYSOTHERS) $(LIBSYSEXTRA) $(LIBSYS_POST)

## ==== Compiler rules override
# You may take a copy of the compiler rules file and modify it
#    cp $(s.get_compiler_rules) Compiler_rules_${COMP_ARCH}
#    and set it in Makefile.user.mk:
#       COMP_RULES_FILE = $(ROOT)/Compiler_rules_$(COMP_ARCH)

#COMP_RULES_FILE =

## ==== Compiler/linker options
OPTIL  := 2
OMP    := -openmp
MPI    := -mpi
#DEBUG  := -debug
#PROFIL := -prof
#SHARED_O_DYNAMIC := -dynamic
#SHARED_O_DYNAMIC := -shared

## Compiler
#MODEL: MODEL_FFLAGS, MODEL_CFLAGS, MODEL_CPPFLAGS
#USER:  FFLAGS, CFLAGS, CPPFLAGS

RDE_FFLAGS = $(OMP) $(MPI) -O $(OPTIL) $(DEBUG) $(SHARED_O_DYNAMIC) $(PROFIL) $(RDE_FFLAGS_ARCH) $(RDE_FFLAGS_COMP)
RDE_CFLAGS = $(OMP) $(MPI) -O $(OPTIL) $(DEBUG) $(SHARED_O_DYNAMIC) $(PROFIL) $(RDE_CFLAGS_ARCH) $(RDE_CFLAGS_COMP)

MODEL_FFLAGS1  = $(MODEL_FFLAGS) $(MODEL1_FFLAGS) $(MODEL2_FFLAGS) $(MODEL3_FFLAGS) $(MODEL4_FFLAGS) $(MODEL5_FFLAGS)
MODEL_CFLAGS1  = $(MODEL_CFLAGS) $(MODEL1_CFLAGS) $(MODEL2_CFLAGS) $(MODEL3_CFLAGS) $(MODEL4_CFLAGS) $(MODEL5_CFLAGS)
MODEL_CPPFLAGS1= $(MODEL_CPPFLAGS) $(MODEL1_CPPFLAGS) $(MODEL2_CPPFLAGS) $(MODEL3_CPPFLAGS) $(MODEL4_CPPFLAGS) $(MODEL5_CPPFLAGS)

RDEALL_FFLAGS = $(RDE_FFLAGS) $(MODEL_FFLAGS1) $(FFLAGS) $(RDE_OPTF_MODULE) $(COMPF) $(FCOMPF)
RDEALL_CFLAGS = $(RDE_CFLAGS) $(MODEL_CPPFLAGS1) $(MODEL_CFLAGS1) $(CPPFLAGS) $(CFLAGS) $(COMPF) $(CCOMPF)

## Linker
#MODEL: MODEL_LFLAGS
#USER:  LFLAGS

RDE_LFLAGS       = $(OMP) $(MPI) $(DEBUG) $(SHARED_O_DYNAMIC) $(PROFIL) $(RDE_LFLAGS_ARCH) $(RDE_LFLAGS_COMP)
RDE_LFLAGS_NOMPI = $(OMP) $(DEBUG) $(SHARED_O_DYNAMIC) $(PROFIL) $(RDE_LFLAGS_ARCH) $(RDE_LFLAGS_COMP) $(RDE_MKL_NOMPI)

MODEL_LFLAGS1    = $(MODEL_LFLAGS) $(MODEL1_LFLAGS) $(MODEL2_LFLAGS) $(MODEL3_LFLAGS) $(MODEL4_LFLAGS) $(MODEL5_LFLAGS)

RDEALL_LFLAGS       = $(RDE_LFLAGS) $(MODEL_LFLAGS1) $(LFLAGS)
RDEALL_LFLAGS_NOMPI = $(RDE_LFLAGS_NOMPI) $(MODEL_LFLAGS1) $(LFLAGS)

## ==== Defines
#MODEL: MODEL_DEFINE
#USER:  DEFINE

RDE_DEFINES   = $(RDE_DEFINES_ARCH) $(RDE_DEFINES_COMP)

MODEL_DEFINE1 = $(MODEL_DEFINE) $(MODEL1_DEFINE) $(MODEL2_DEFINE) $(MODEL3_DEFINE) $(MODEL4_DEFINE) $(MODEL5_DEFINE)

# RDEALL_DEFINES_NAMES = $(RDE_DEFINES) $(MODEL_DEFINE1) $(DEFINE)
# RDEALL_DEFINES       = $(foreach item,$(RDEALL_DEFINES_NAMES),-D$(item))
RDEALL_DEFINES       = $(RDE_DEFINES) $(MODEL_DEFINE1) $(DEFINE)

## ==== Includes
#MODEL: MODEL_INCLUDE_PRE MODEL_INCLUDE MODEL_INCLUDE_POST
#USER:  INCLUDES_PRE INCLUDES INCLUDES_POST

RDE_INCLUDE0 := $(CONST_RDEINC)
RDE_INCLUDE_PRE = $(BUILDPRE)
RDE_INCLUDE_MOD = $(BUILDMOD)
#RDE_INCLUDE     = $(RDE_INCLUDE_COMP) $(RDE_INCLUDE_ARCH) $(RDE_INCLUDE0) $(PWD) $(RDE_INCLUDE_PRE) $(RDE_INCLUDE_MOD)
RDE_INCLUDE     = $(RDE_INCLUDE_COMP) $(RDE_INCLUDE_ARCH) $(RDE_INCLUDE0)

INCLUDES1      = $(INCLUDES_PRE) $(INCLUDES) $(INCLUDES_POST)
#RDE_INCLUDE1   = $(RDE_INCLUDE_PRE) $(RDE_INCLUDE) $(RDE_INCLUDE_POST)
MODEL_INCLUDE1 = $(MODEL_INCLUDE_PRE) $(MODEL_INCLUDE) $(MODEL1_INCLUDE) $(MODEL2_INCLUDE) $(MODEL3_INCLUDE) $(MODEL4_INCLUDE) $(MODEL5_INCLUDE) $(SRCPATH_INCLUDE) $(MODEL_INCLUDE_POST)

#RDEALL_INCLUDE_NAMES = $(PWD) $(RDE_INCLUDE_PRE) $(RDE_INCLUDE_MOD) $(INCLUDES1) $(MODEL_INCLUDE1) $(RDE_INCLUDE1)
RDEALL_INCLUDE_NAMES = $(PWD) $(SRCPATH_INCLUDE_OVERRIDES) $(RDE_INCLUDE_PRE) $(RDE_INCLUDE_MOD) $(INCLUDES1) $(MODEL_INCLUDE1) $(RDE_INCLUDE)  $(RDE_INCLUDE_POST)
RDEALL_INCLUDES      = $(foreach item,$(RDEALL_INCLUDE_NAMES),-I$(item))

## ==== Libpath
#MODEL: MODEL_LIBPATH_PRE MODEL_LIBPATH MODEL_LIBPATH_POST
#USER:  LIBPATH_PRE LIBPATH_USER LIBPATH_POST
#LEGACY:LIBDIR LIBPATHEXTRA LIBSYSPATHEXTRA LIBPATHOTHER

RDE_LIBPATH_LEGACY = $(LIBDIR) $(LIBPATHEXTRA) $(LIBSYSPATHEXTRA) $(LIBPATHOTHER)
RDE_LIBPATH        = $(RDE_LIBPATH_COMP) $(RDE_LIBPATH_ARCH) $(PWD) $(BUILDLIB) $(RDE_LIBPATH_LEGACY)
#NOTE: BUILDLIB also apears as: LIBDIR in RDE_LIBPATH_LEGACY

LIBPATH1       = $(LIBPATH_PRE) $(LIBPATH_USER) $(LIBPATH_POST)
RDE_LIBPATH1   = $(RDE_LIBPATH_PRE) $(RDE_LIBPATH) $(RDE_LIBPATH_POST)
MODEL_LIBPATH1 = $(MODEL_LIBPATH_PRE) $(MODEL_LIBPATH)  $(MODEL1_LIBPATH)  $(MODEL2_LIBPATH)  $(MODEL3_LIBPATH)  $(MODEL4_LIBPATH)  $(MODEL5_LIBPATH) $(MODEL_LIBPATH_POST)

RDEALL_LIBPATH_NAMES = $(LIBPATH1) $(MODEL_LIBPATH1) $(RDE_LIBPATH1)
RDEALL_LIBPATH       = $(foreach item,$(RDEALL_LIBPATH_NAMES),-L$(item))

## ==== Libs
#MODEL: MODEL_LIBPRE MODEL_LIBAPPL MODEL_LIBPOST
#       MODEL_LIBSYSPRE MODEL_LIBSYS MODEL_LIBSYSPOST
#USER:  LIBS_PRE LIBAPPL LIBS_POST LIBRMN LIBSYS_PRE LIBSYS LIBSYS_POST
#LEGACY:RMN_VERSION LIBOTHERS LIBSYSUTIL LIBSYSEXTRA, CODEBETA
#       LIBCOMM LIBVGRID LIBEZSCINT LIBUTIL 
#       LIBMASS LAPACK BLAS RTOOLS BINDCPU LIBHPCSPERF LLAPI IBM_LD
#       LIBHPC LIBPMAPI

#LIBOTHERS          = $(LIBCOMM) $(LIBVGRID) $(LIBEZSCINT) $(LIBUTIL)
RDE_LIBAPPL_LEGACY = $(LIBOTHERS)
RDE_LIBAPPL1       = $(RDE_LIBAPPL) $(RDE_LIBAPPL_LEGACY)

#LIBSYSUTIL         = $(LIBMASS) $(LAPACK) $(BLAS) $(RTOOLS) $(BINDCPU)  $(LIBHPCSPERF) $(LLAPI) $(IBM_LD)
#LIBSYSEXTRA        = $(LIBHPC) $(LIBPMAPI)
RDE_LIBSYS_LEGACY  = $(LIBSYSUTIL) $(LIBSYSEXTRA)
RDE_LIBSYS         = $(RDE_LIBSYS_LEGACY) $(LIBSYSUTIL)

RDEALL_LIBAPPL_PRE  = $(LIBS_PRE) $(MODEL_LIBPRE) $(RDE_LIBPRE)
RDEALL_LIBAPPL_POST = $(LIBS_POST) $(MODEL_LIBPOST) $(RDE_LIBPOST)
RDEALL_LIBAPPL      =  $(RDEALL_LIBAPPL_PRE) $(LIBAPPL) $(MODEL_LIBAPPL) $(MODEL5_LIBAPPL) $(MODEL4_LIBAPPL) $(MODEL3_LIBAPPL) $(MODEL2_LIBAPPL) $(MODEL1_LIBAPPL) $(RDE_LIBAPPL1) $(RDEALL_LIBAPPL_POST)

# LIBRMN = $(RMN_VERSION)

RDEALL_LIBSYS_PRE   = $(LIBSYS_PRE) $(MODEL_LIBSYSPRE) $(RDE_LIBSYSPRE)
RDEALL_LIBSYS_POST  = $(LIBSYS_POST) $(MODEL_LIBSYSPOST) $(RDE_LIBSYSPOST)
RDEALL_LIBSYS       = $(RDEALL_LIBSYS_PRE) $(LIBSYS) $(MODEL_LIBSYS) $(MODEL5_LIBSYS) $(MODEL4_LIBSYS) $(MODEL3_LIBSYS) $(MODEL2_LIBSYS) $(MODEL1_LIBSYS) $(RDE_LIBSYS) $(RDEALL_LIBSYS_POST)

# RDEALL_LIBS_NAMES = $(RDEALL_LIBAPPL) $(LIBRMN) $(RDEALL_LIBSYS)
RDEALL_LIBS_NAMES = $(RDEALL_LIBAPPL) $(RDEALL_LIBSYS) $(RDE_LIB_OVERRIDES)
RDEALL_LIBS       = $(foreach item,$(RDEALL_LIBS_NAMES),-l$(item))

## ==== Constants for $(MAKEFILEDEP)

## all libs tagets have a dependency on LIBDEP_ALL
LIBDEP_ALL =  $(MAKEFILEDEP)
## Modules are created in MODDIR
MODDIR = $(BUILDMOD)
## Local Libraries are created in LIBDIR
LIBDIR = $(BUILDLIB)
## Local Abs are created in BINDIR
BINDIR = $(BUILDBIN)

## ==== Pkg Building Macros

rdeuc = $(shell echo $(1) | tr 'a-z' 'A-Z')

LIB_template1 = \
$$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

LIB_template2 = \
$$(LIBDIR)/lib$(1).a: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a $$@

## ==== Arch specific and Local/user definitions, targets and overrides
#TODO: move these back into separated files RDEINC/ARCH/COMP/Makefile.comp.mk
ifneq (,$(filter Linux_x86-64,$(RDE_BASE_ARCH))$(filter linux26-%,$(RDE_BASE_ARCH))$(filter rhel-%,$(RDE_BASE_ARCH))$(filter ubuntu-%,$(RDE_BASE_ARCH))$(filter sles-%,$(RDE_BASE_ARCH)))
RDE_DEFINES_ARCH = -DLINUX_X86_64
LAPACK      = lapack
BLAS        = blas
LIBMASSWRAP =
LIBMASS     = $(LIBMASSWRAP) massv_p4

#RDE_OPTF_MODULE = -module $(BUILDMOD)
ifneq (,$(filter pgi%,$(COMP_ARCH)))
RDE_OPTF_MODULE = -module $(BUILDMOD)
endif
ifneq (,$(filter gfortran%,$(COMP_ARCH)))
RDE_OPTF_MODULE = -J $(BUILDMOD)
endif
ifneq (,$(filter intel%,$(COMP_ARCH))$(filter PrgEnv-intel%,$(COMP_ARCH)))
RDE_OPTF_MODULE = -module $(BUILDMOD)
endif

ifneq (,$(filter intel%,$(COMP_ARCH))$(filter PrgEnv-intel%,$(COMP_ARCH)))
LAPACK      =
BLAS        =
RDE_FP_MODEL= -fp-model source
RDE_INTEL_DIAG_DISABLE = -diag-disable 7713 -diag-disable 10212 -diag-disable 5140
RDE_FFLAGS_COMP = $(RDE_INTEL_DIAG_DISABLE) $(RDE_MKL) $(RDE_FP_MODEL)
RDE_CFLAGS_COMP = $(RDE_INTEL_DIAG_DISABLE) $(RDE_MKL) $(RDE_FP_MODEL)
RDE_LFLAGS_COMP = $(RDE_INTEL_DIAG_DISABLE) $(RDE_MKL) $(RDE_FP_MODEL)
endif
ifneq (,$(filter intel%,$(COMP_ARCH)))
RDE_MKL     = -mkl
endif
ifneq (,$(filter PrgEnv-intel%,$(COMP_ARCH)))
RDE_MKL_NOMPI   = -mkl
endif
ifneq (,$(filter pgi%,$(COMP_ARCH)))
RDE_KIEEE = -Kieee
RDE_FFLAGS_COMP = $(RDE_KIEEE)
endif
endif

#Default target if no other is specified
.PHONY: nothingtobedone
nothingtobedone:
	@echo "Nothing to be done"

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   RDE_ABS_DEP      = $(RDE_LIB_OVERRIDES_NEW_FILE)
   RDE_LIB_OVERRIDES= $(RDE_LIB_OVERRIDES_NEW)
endif

LOCALMAKEFILES0 := $(foreach mydir,$(CONST_SRCPATH_INCLUDE) $(SRCPATH_INCLUDE_OVERRIDES),$(wildcard $(mydir)/Makefile.local*.mk))
RDEBUILDMAKEFILES = $(wildcard \
   $(ROOT)/Makefile.rules.mk \
   $(ROOT)/$(MAKEFILEDEP) \
   $(LOCALMAKEFILES0) \
   $(MAKEFILEUSERLIST))
ifneq (,$(RDEBUILDMAKEFILES))
   ifneq (,$(DEBUGMAKE))
      $(foreach item, $(RDEBUILDMAKEFILES), $(info include $(item)))
      # $(info include $(RDEBUILDMAKEFILES))
   endif
   include $(RDEBUILDMAKEFILES)
endif

ifneq (,$(findstring s,$(MAKEFLAGS)))
   VERBOSE :=
endif
ifneq (,$(VERBOSE))
ifeq (0,$(VERBOSE))
   VERBOSEV  :=
   VERBOSEVL :=
else
   VERBOSEV  := -v
   VERBOSEVL := -verbose
endif
endif

RDEBUILDNAMEFILE = $(wildcard $(ROOT)/.rde.buildname)
ifneq (,$(RDEBUILDNAMEFILE))
   ifneq (,$(DEBUGMAKE))
      $(info include $(RDEBUILDNAMEFILE))
   endif
   include $(RDEBUILDNAMEFILE)
endif

ifneq (,$(BUILDNAME))
   export RDE_BUILDNAME=$(BUILDNAME)
# else
#    #TODO:
endif

RDE_COMP_RULES_FILE_USER =
ifneq (,$(COMP_RULES_FILE))
   ifneq (,$(wildcard $(ROOT)/$(COMP_RULES_FILE)))
      RDE_COMP_RULES_FILE_USER = --comprules=$(ROOT)/$(COMP_RULES_FILE)
   endif
   ifneq (,$(wildcard $(COMP_RULES_FILE)))
      RDE_COMP_RULES_FILE_USER = --comprules=$(COMP_RULES_FILE)
   endif
   $(info RDE_COMP_RULES_FILE_USER =$(RDE_COMP_RULES_FILE_USER))
endif

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   RBUILD4objMPI    = $(RBUILD4MPI)
   RBUILD4objNOMPI  = $(RBUILD4NOMPI)
   RBUILD4objMPI_C  = $(RBUILD4MPI_C)
   RBUILD4objNOMPI_C= $(RBUILD4NOMPI_C)
   RDE_BUILDDIR_SFX = -$(USER)
   ifeq (,$(RDE_BUILDNAME))
      export RDE_BUILDNAME=$(USER)$(shell date '+%Y%m%d.%H%M%S')
   endif
endif

## ==== Targets
ifneq (,$(DEBUGMAKE))
   $(info Definitions done, Starting Targets section)
endif

#.DEFAULT:
#	@rdeco -q -i $@ || true \;

.DEFAULT:
	@if [[ x$$(echo $@ | cut -c1-9) == x_invdep_. ]] ; then \
	   echo > /dev/null ;\
	elif [[ x"$(filter $(suffix $@),$(CONST_RDESUFFIX) .mk)" != x"" ]] ; then \
	   rdeco -q -i $@  && echo "Checking out: $@" || (echo "ERROR: File Not found (Makefile.build.mk:rdeco): $@" && exit 1);\
	else \
	   echo "ERROR: No such target: $@" 1>&2 ; \
	   exit 1 ; \
	fi

.PHONY: objexp #TODO

#Produire les objets de tous les fichiers de l'experience qu'ils soient checkout ou non
objects: $(OBJECTS)

libs: $(OBJECTS) $(ALL_LIBS) $(RDE_LIBS_USER_EXTRA)
	ls -al $(ALL_LIBS) $(RDE_LIBS_USER_EXTRA)

bins: $(RDE_BINS_USER_EXTRA)
	ls -al $(RDE_BINS_USER_EXTRA)

# #TODO: get .o .mod from lib again after make clean?
# #TODO: should we keep .mod after make clean?

# clean:
# 	chmod -R u+w . $(BUILDMOD) $(BUILDPRE) 2> /dev/null || true ;\
# 	rm -f $(foreach mydir,. * */* */*/* */*/*/* */*/*/*/*,$(foreach exte,$(INCSUFFIXES) $(SRCSUFFIXES) .o .[mM][oO][dD],$(mydir)/*$(exte))) 2>/dev/null || true  ;\
# 	rm -f $(foreach mydir,. * */* */*/* */*/*/* */*/*/*/*,$(foreach mydir0,$(BUILDMOD) $(BUILDPRE),$(mydir0)/$(mydir)/*)) 2>/dev/null || true

# check_inc_dup: links
# 	echo "Checking for duplicated include files" ;\
# 	pfcheck_dup -r --src=$(VPATH) --ext="$(INCSUFFIXES)" . $(INCLUDES) $(EC_INCLUDE_PATH) #$(shell s.generate_ec_path --include)

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.build.mk [END] =======================================)
endif
