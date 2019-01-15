## ====================================================================
## File: RDEINC/Makefile
##

SHELL = /bin/bash

## ==== Basic definitions
rdeuc = $(shell echo $(1) | tr 'a-z' 'A-Z')

ifneq (,$(DEBUGMAKE))
   VERBOSE = 2
   MYTIME  = time
endif
ifneq (,$(MYTIME))
   MYTIMEX = set -x ; time
endif

ROOT := $(PWD)
ifeq (,$(ROOT))
ROOT := $(CURDIR)
endif

ifeq (,$(rde))
   $(error FATAL ERROR: rde is not defined; Maybe you forgot to load the RDE env)
endif
ifeq (,$(wildcard $(shell rdevar srcpathfile)))
   $(error FATAL ERROR: Not in a properly set RDE development dir, Source file list not found, maybe you forgot to do "rde addsrc" or equivalent, see "rde help" for more details)
endif
MAKEFILE_CONST := $(shell rdevar Makefile_const)
MAKEFILE_CONST_REDONE := 0

ifeq (,$(wildcard $(MAKEFILE_CONST)))
   $(info Building $(MAKEFILE_CONST))
   $(info $(shell $(MYTIMEX) rdevar --mkfile))
   MAKEFILE_CONST_REDONE := 1
endif
ifeq (,$(CONST_BUILD))
   ifneq (,$(DEBUGMAKE))
      $(info include $(MAKEFILE_CONST))
   endif
   include $(MAKEFILE_CONST)
endif

MAKEFILE_CONST_REDO := 0
ifeq (0,$(MAKEFILE_CONST_REDONE))
   ifneq ($(rde_version),$(CONST_RDE_VERSION))
      $(info WARNING: $(MAKEFILE_CONST) found for another version of RDE: $(CONST_RDE_VERSION))
      MAKEFILE_CONST_REDO := 1
   endif
   ifeq (0,$(MAKEFILE_CONST_REDO))
      MAKEFILE_CONST_AGE_MAX := 10800
      #MAKEFILE_CONST_AGE := $(shell let age=`date +%s`-$(CONST_RDE_LASTUPDATED) ; echo $$age)
      #MAKEFILE_CONST_REDO := $(shell [[ $(MAKEFILE_CONST_AGE) -lt $(MAKEFILE_CONST_AGE_MAX) ]] && echo 1 || echo 0)
      MAKEFILE_CONST_REDO := $(shell let age=`date +%s`-$(CONST_RDE_LASTUPDATED) ; [[ $${age} -lt $(MAKEFILE_CONST_AGE_MAX) ]] && echo 0 || echo 1)

   endif
endif

ifeq (1,$(MAKEFILE_CONST_REDO))
   $(info Building $(MAKEFILE_CONST))
   $(info $(shell $(MYTIMEX) rdevar --mkfile))
   ifneq (,$(DEBUGMAKE))
      $(info include $(MAKEFILE_CONST))
   endif
   include $(MAKEFILE_CONST)
endif

ifeq (,$(RDE_DEP_SUPP_FILE))
   RDE_DEP_SUPP_FILE := $(rde)/etc/rdedep.supp
endif

BUILD    := $(ROOT)/$(CONST_BUILD)
BUILDPRE := $(ROOT)/$(CONST_BUILDPRE)
BUILDOBJ := $(ROOT)/$(CONST_BUILDOBJ)
BUILDMOD := $(ROOT)/$(CONST_BUILDMOD)
BUILDLIB := $(ROOT)/$(CONST_BUILDLIB)
BUILDBIN := $(ROOT)/$(CONST_BUILDBIN)
BINDIR   = $(BUILDBIN)

MAKEFILEDEP := $(CONST_MAKEFILE_DEP)
MAKEFILEUSERLIST := $(wildcard $(ROOT)/$(CONST_MAKEFILE_ROOT) $(ROOT)/$(CONST_MAKEFILE_ROOT_BASEARCH) $(ROOT)/$(CONST_MAKEFILE_ROOT_COMPARCH))

INCSUFFIXES := $(CONST_RDESUFFIXINC)
SRCSUFFIXES := $(CONST_RDESUFFIXSRC)
RDE_BASE_ARCH := $(CONST_RDE_BASE_ARCH)
RDE_BASE_ARCH_EQUIV := $(CONST_RDE_BASE_ARCH_EQUIV)
RDE_ARCH_LIST := $(CONST_RDE_LINKED_ARCH_LIST)
ifeq (,$(RDE_ARCH_LIST))
RDE_ARCH_LIST := $(CONST_RDE_KNOWN_BASE_ARCH_LIST)
endif

RDE_INCLUDE0 := $(CONST_RDEINC)

CLOCK_SKEW_MAX := 1

ifeq (,$(ROOT))
   $(error FATAL ERROR: ROOT is not defined)
endif
ifeq ($(ROOT),$(BUILD))
   $(error FATAL ERROR: BUILD == ROOT)
endif

ifneq (,$(findstring s,$(MAKEFLAGS)))
   VERBOSE = 
endif
ifneq (,$(VERBOSE))
ifeq (0,$(VERBOSE))
   VERBOSEV  :=
else
   VERBOSEV  := -v
   ifeq (2,$(VERBOSE))
      VERBOSEV2 := -v
      VERBOSEVV := -v -v
   endif
endif
endif
ifeq (,$(DEBUGMAKE))
   NOPRINTDIR := --no-print-directory
endif

MAKE_LINKS = $(MYTIMEX) .rdeupdate_build_links.ksh $(VERBOSEVV)
#MYMAKE_VARS = ROOT=$(ROOT) RDE_BASE_ARCH=$(RDE_BASE_ARCH) BINDIR=$(BINDIR) DEBUGMAKE=$(DEBUGMAKE) MYTIME="$(MYTIME)" MAKEFILE_CONST=$(MAKEFILE_CONST)
MYMAKE_VARS = ROOT=$(ROOT) RDE_BASE_ARCH=$(RDE_BASE_ARCH) BINDIR=$(BINDIR) MAKEFILE_CONST=$(MAKEFILE_CONST)
#MYMAKE_VARS = ROOT=$(ROOT) RDE_BASE_ARCH=$(RDE_BASE_ARCH) MAKEFILE_CONST=$(MAKEFILE_CONST)

RDE_TARGET_OBJ = obj
RDE_TARGET_OBJ_FORCES = obj_forced
RDE_TARGET_OBJ_TOUCH = obj_touch
RDE_TARGET_CLEAN = clean
RDE_TARGET_BUILDCLEAN = buildclean
RDE_TARGET_BUILDCLEAN_PLUS = buildclean+
RDE_TARGET_DISTCLEAN = distclean

RDE_BUILDCLEAN_LIST = $(BUILDSRC) $(BUILDPRE) $(BUILDOBJ) $(BUILDMOD) $(BUILDLIB) $(BUILDBIN) $(BUILDSSM) $(BUILD)

ifneq (,$(MAKEFILEUSERLIST))
   ifneq (,$(DEBUGMAKE))
      $(info include $(MAKEFILEUSERLIST))
   endif
   include $(MAKEFILEUSERLIST)
endif

## ==== Targets

RDE_TARGET_ALL = all
$(RDE_TARGET_ALL): obj

.SUFFIXES:

.DEFAULT: 
	$(MAKE) $(NOPRINTDIR) sanity
	$(MAKE_LINKS) ;\
	cd $(BUILDOBJ) ;\
   if [[ $$? != 0 ]] ; then echo "ERROR: cannot move into build dir: $(BUILDOBJ)" ; exit 1 ; fi ;\
	$(MYTIME) $(MAKE) $(NOPRINTDIR) $@ $(MYMAKE_VARS)

.PHONY: sanity links dep deplocal objloc obj_forced obj_touch clean rmpo cleanbuild cleanbuild0 cleanall

.PHONY: help
help:
	@cat $(rde)/etc/rde_make_help.txt
	@echo "Raw list of Makefile targets:"
	@echo
	@rdelsmake --targets
	@echo "=============================================================="


.PHONY: tags
# Build tags file for emacs/vi dependiong on ${EDITOR:-vi} env.var.
tags:
	echo "Building the tag file for ${EDITOR:-vi} code navigation"
	echo "You may set the EDITOR env. var. to make the tag file for that editor"
	$(MYTIME) rdemktags -v

.PHONY: deplocal
deplocal: #$(MAKEFILE_CONST)
	rm -f $(MAKEFILEDEP) 2>/dev/null || true
	$(MYTIME) rdemkdep -v --supp $(RDE_DEP_SUPP_FILE) --arch $(RDE_BASE_ARCH_EQUIV)
.PHONY: dep

ifneq (,$(filter aix-%,$(RDE_BASE_ARCH))$(filter AIX-%,$(RDE_BASE_ARCH)))
dep: deplocal
else
dep: #$(MAKEFILE_CONST)
	rm -f Makefile.dep.*.mk 2>/dev/null || true
	$(MYTIME) rdemkdep -v --supp $(RDE_DEP_SUPP_FILE) --arch "$(RDE_ARCH_LIST)"
endif

#$(MAKEFILE_CONST) 
.PHONY: sanity
sanity: $(MAKEFILEDEP) Makefile.build.mk Makefile.rules.mk $(BUILD) $(BUILDOBJ) chkclock
chkclock:
	rdechkclock -v -t $(CLOCK_SKEW_MAX)|| exit 1
.PHONY: links
links: sanity
	$(MAKE_LINKS)

.PHONY: Makefile.const
Makefile.const: 
	$(info Building Makefile Constants: $(MAKEFILE_CONST))
	$(MYTIME) rdevar --mkfile
$(MAKEFILE_CONST): $(CONST_SRCPATHFILE) $(CONST_RDE_CONFIG_FILE)
	$(info Building Makefile Constants: $(MAKEFILE_CONST))
	$(MYTIME) rdevar --mkfile


$(MAKEFILEDEP):
	$(MYTIME) rdemkdep -v --supp $(RDE_DEP_SUPP_FILE) --arch $(RDE_BASE_ARCH_EQUIV)
Makefile.build.mk:
	$(MYTIME) rdemklink $(VERBOSEV2)
Makefile.rules.mk:
	$(MYTIME) rdemklink $(VERBOSEV2)
$(BUILD):
	$(MYTIME) rdemklink $(VERBOSEV2)
$(BUILDOBJ):
	$(MYTIME) rdemklink $(VERBOSEV2)

.PHONY: $(RDE_TARGET_OBJ)
#Compiler uniquement les fichiers  modifies.
$(RDE_TARGET_OBJ):  sanity
	$(MAKE_LINKS) ;\
	cd $(BUILDOBJ) ;\
   if [[ $$? != 0 ]] ; then echo "ERROR: cannot move into build dir: $(BUILDOBJ)" ; exit 1 ; fi ;\
	localfiles="" ;\
	for myfile in $(foreach exte,$(SRCSUFFIXES),$(wildcard *$(exte))) ; do \
		if [[ -L $${myfile} ]] ; then \
			localfiles="$${localfiles} $${myfile%.*}.o _invdep_.$${myfile}" ;\
		fi ;\
	done ;\
	for myfile in $(foreach exte,$(INCSUFFIXES),$(wildcard *$(exte))) ; do \
		if [[ -L $${myfile} ]] ; then \
			localfiles="$${localfiles} _invdep_.$${myfile}" ;\
		fi ;\
	done ;\
	$(MYTIME) $(MAKE) $(NOPRINTDIR) $${localfiles} $(MYMAKE_VARS)

.PHONY: $(RDE_TARGET_OBJ_FORCES)
$(RDE_TARGET_OBJ_FORCES): $(RDE_TARGET_OBJ_TOUCH) 
	$(MYTIME) $(MAKE) $(NOPRINTDIR) $(RDE_TARGET_OBJ) $(MYMAKE_VARS)
.PHONY: $(RDE_TARGET_OBJ_TOUCH)
$(RDE_TARGET_OBJ_TOUCH):
	touch `ls *.f* *.c* *.h* *.F* *.t* 2>/dev/null` 2>/dev/null || true

.PHONY: rmpo
rmpo: $(RDE_TARGET_CLEAN)
.PHONY: $(RDE_TARGET_CLEAN)
$(RDE_TARGET_CLEAN): 
	chmod -R u+w $(BUILDOBJ)/*.o $(BUILDMOD)/*.[Mm][Oo][Dd] 2>/dev/null || true
	rm -rf       $(BUILDOBJ)/*.o $(BUILDMOD)/*.[Mm][Oo][Dd] 2>/dev/null || true
	cd $(BUILDPRE) ;\
   if [[ $$? != 0 ]] ; then echo "ERROR: cannot move into build dir: $(BUILDOBJ)" ; exit 1 ; fi ;\
	for myfile in $(foreach exte,$(SRCSUFFIXES) $(INCSUFFIXES),$(wildcard *$(exte))) ; do \
		if [[ ! -L $${myfile} ]] ; then \
			chmod u+w $${myfile} ; rm -f  $${myfile};\
		fi ;\
	done
	rm -f $(MAKEFILE_CONST)

.PHONY: $(RDE_TARGET_BUILDCLEAN_PLUS)
$(RDE_TARGET_BUILDCLEAN_PLUS): 
	chmod u+w -R $(BUILD)/
	rm -f $$(find $(BUILD)/ -type f) 2>/dev//null || true
	rm -f $(MAKEFILE_CONST)

.PHONY: $(RDE_TARGET_BUILDCLEAN)
$(RDE_TARGET_BUILDCLEAN): 
	for mydir in $(RDE_BUILDCLEAN_LIST) ;do\
	   chmod u+w $${mydir}/* $${mydir}/.[a-zA-Z0-9_-]* 2>/dev/null || true ;\
		rm -f     $${mydir}/* $${mydir}/.[a-zA-Z0-9_-]* 2>/dev/null || true ;\
	done
	rm -f $(MAKEFILE_CONST)

.PHONY: $(RDE_TARGET_DISTCLEAN)
$(RDE_TARGET_DISTCLEAN): $(RDE_TARGET_BUILDCLEAN)
	for item in `rdevar RDEROOTLINKS` ; do \
		if [[ -L $${item} ]] ; then \
			rm -f $${item} 2>/dev/null || true; \
		fi ;\
	done
	chmod u+w .`rdevar RDEROOTFILES` 2>/dev/null || true
	rm -f `rdevar RDEROOTFILES` 2>/dev/null || true
	chmod -R u+w `rdevar RDEDIRS` 2>/dev/null || true
	rm -rf `rdevar RDEDIRS` 2>/dev/null || true
	for item in $(RDE_ARCH_LIST) ; do \
		buildlink=$$(rdevar --arch $${item} build) ;\
		if [[ -d $${buildlink:-__no-such-link__} ]] ; then \
			rm -rf $${buildlink:-__no-such-link__}/* ;\
		fi ;\
		rm -f $${buildlink:-__no-such-link__} ;\
	done

## ====================================================================
