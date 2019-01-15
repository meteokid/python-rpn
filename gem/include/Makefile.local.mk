ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/include/Makefile.local.mk)
$(info ## )
endif

# ifeq (,$(wildcard $(gem)/VERSION))
#    $(error Not found: $(gem)/VERSION)
# endif
# GEM_VERSION   = $(shell cat $(gem)/VERSION)
# GEM_VERSION_X = x/
GEM_VERSION0  = 4.8-LTS.13
GEM_VERSION   = $(notdir $(GEM_VERSION0))
GEM_VERSION_X = $(dir $(GEM_VERSION0))

GEM_SFX=$(RDE_BUILDDIR_SFX)

GEM_ABS_SHARED = gemdm
GEM_ABS_SHARED_FILES = $(BINDIR)/$(maindm)

GEM_ABS       = gemdm
GEM_ABS_FILES = $(BINDIR)/$(maindm)

LIBHPCSPERF=

.PHONY: gem_vfiles
GEM_VFILES = gem_version.inc gem_version.h
gem_vfiles: $(GEM_VFILES)
gem_version.inc:
	.rdemkversionfile "gem" "$(GEM_VERSION)" . f
gem_version.h:
	.rdemkversionfile "gem" "$(GEM_VERSION)" . c

# export ATM_MODEL_BNDL     = $(SSM_RELDIRBNDL)$(GEM_VERSION)
# export ATM_MODEL_DSTP    := $(shell date '+%Y-%m-%d %H:%M %Z')
# export ATM_MODEL_VERSION  = $(GEM_VERSION)
# export ATM_MODEL_NAME    := GEMDM

#---- Abs targets -----------------------------------------------------
## GEM model targets (modelutils/gemdyn/rpnphy)
.PHONY: gem gemdm gem_nompi gemdm_nompi allbin_gem allbincheck_gem gem_shared gemdm_shared

allbin_gem: $(GEM_ABS)
	ls -l $(GEM_ABS_FILES)
allbincheck_gem:
	for item in $(GEM_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

gem: | gemdm prgemnml prphynml gemgrid checkdmpart

maindm      = $(ABSPREFIX)maingemdm$(ABSPOSTFIX)_$(BASE_ARCH).Abs
maindm_rel  = $(ABSPREFIX)maingemdm_REL_$(BASE_ARCH).Abs
gemdm: | gemdm_rm $(BINDIR)/$(maindm)
	ls -l $(BINDIR)/$(maindm)
gemdm_rm:
	rm -f $(BINDIR)/$(maindm)
$(BINDIR)/$(maindm): | $(GEM_VFILES)
	export MAINSUBNAME="gemdm" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
	$(RBUILD4objMPI)

gem_shared: | gemdm_shared

maindm_shared  = $(ABSPREFIX)maingemdm_shared$(ABSPOSTFIX)_$(BASE_ARCH).Abs
gemdm_shared: | gemdm_shared_rm $(BINDIR)/$(maindm_shared)
	ls -l $(BINDIR)/$(maindm_shared)
gemdm_shared_rm:
	rm -f $(BINDIR)/$(maindm_shared)
$(BINDIR)/$(maindm_shared): | $(GEM_VFILES)
	export MAINSUBNAME="gemdm" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
	export ATM_MODEL_VERSION="$(GEM_VERSION)" ;\
	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_SHARED_V) $(GEMDYN_LIBS_SHARED_DEP)" ;\
	$(RBUILD4objMPI)


# gem_nompi: | gemdm_nompi

# gemdm_nompi: | $(GEMDYN_VFILES)
# 	rm -f $(BINDIR)/$(maindm)
# 	export MAINSUBNAME="gemdm" ;\
# 	export ATM_MODEL_NAME="$${MAINSUBNAME} $(BUILDNAME)" ;\
# 	export ATM_MODEL_VERSION="$(GEMDYN_VERSION)" ;\
# 	export RBUILD_LIBAPPL="$(GEMDYN_LIBS_V) $(GEMDYN_LIBS_DEP)" ;\
# 	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
# 	$(RBUILD4objMPI)
# 	ls -l $(BINDIR)/$(maindm)

## ====================================================================
