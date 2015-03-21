ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gemdyn/Makefile.user.mk)
$(info ## )
endif
#VERBOSE=1

SSM_NAME := gemdyn

SSM_X          = $(GEMDYN_VERSION_X)
SSM_DEPOT_DIR := $(HOME)/SsmDepot
SSM_BASE      := $(HOME)/SsmBundles
SSM_BASE_DOM   = $(SSM_BASE)/GEM/d/$(SSM_X)$(SSM_NAME)
SSM_BASE_BNDL  = $(SSM_BASE)/GEM/$(SSM_X)$(SSM_NAME)

ATM_MODEL_ISOFFICIAL := true
RBUILD_EXTRA_OBJ0    := 

COMPONENTS        := $(SSM_NAME)
COMPONENTS_UC     := $(foreach item,$(COMPONENTS),$(call rdeuc,$(item)))

COMPONENTS2       := modelutils rpnphy $(COMPONENTS)
COMPONENTS2_UC    := $(foreach item,$(COMPONENTS2),$(call rdeuc,$(item)))

COMPONENTS_VFILES := $(foreach item,$(COMPONENTS2_UC),$($(item)_VFILES))

#------------------------------------

MYSSMINCLUDEMK = $(wildcard $(RDE_INCLUDE0)/Makefile.ssm.mk $(gemdyn)/include/Makefile.ssm.mk)
ifneq (,$(MYSSMINCLUDEMK))
   ifneq (,$(DEBUGMAKE))
      $(info include $(MYSSMINCLUDEMK))
   endif
   include $(MYSSMINCLUDEMK)
endif

#------------------------------------

.PHONY: components_objects
components_objects: $(COMPONENTS_VFILES) $(OBJECTS)


.PHONY: components_libs
COMPONENTS_LIBS_FILES = $(foreach item,$(COMPONENTS_UC),$($(item)_LIBS_ALL_FILES_PLUS))
components_libs: $(COMPONENTS_VFILES) $(OBJECTS) $(COMPONENTS_LIBS_FILES)
	ls -l $(COMPONENTS_LIBS_FILES)


COMPONENTS_ABS  := $(foreach item,$(COMPONENTS_UC),$($(item)_ABS))
COMPONENTS_ABS_FILES  := $(foreach item,$(COMPONENTS_UC),$($(item)_ABS_FILES))
.PHONY: components_abs components_abs_check
components_abs: $(COMPONENTS_ABS)
	ls -l $(COMPONENTS_ABS_FILES)


COMPONENTS_SSM_ALL  := $(foreach item,$(COMPONENTS_UC),$($(item)_SSMALL_FILES))
COMPONENTS_SSM_ARCH := $(foreach item,$(COMPONENTS_UC),$($(item)_SSMARCH_FILES))
COMPONENTS_SSM := $(COMPONENTS_SSM_ALL) $(COMPONENTS_SSM_ARCH)
.PHONY: components_ssm
components_ssm: $(COMPONENTS_SSM)
components_ssm_all: $(COMPONENTS_SSM_ALL)
components_ssm_arch: $(COMPONENTS_SSM_ARCH)


components_install:
	if [[ x$(CONFIRM_INSTALL) != xyes ]] ; then \
		echo "Please use: make $@ CONFIRM_INSTALL=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v $(COMPONENTS_UNINSTALL) \
			--dest=$(SSM_BASE_DOM)/$(SSM_NAME)_$(GEMDYN_VERSION) \
			--bndl=$(SSM_BASE_BNDL)/$(GEMDYN_VERSION).bndl \
			--pre=$(gemdyn)/ssmusedep_all.bndl \
			--post=$(gemdyn)/ssmusedep_post.bndl \
			--base=$(SSM_BASE) \
			$(SSM_NAME){_,+*_}$(GEMDYN_VERSION)_*.ssm

components_uninstall:
	if [[ x$(UNINSTALL_CONFIRM) != xyes ]] ; then \
		echo "Please use: make $@ UNINSTALL_CONFIRM=yes" ;\
		exit 1;\
	fi
	cd $(SSM_DEPOT_DIR) ;\
	rdessm-install -v $(COMPONENTS_UNINSTALL) \
			--dest=$(SSM_BASE_DOM)/$(SSM_NAME)_$(GEMDYN_VERSION) \
			--bndl=$(SSM_BASE_BNDL)/$(GEMDYN_VERSION).bndl \
			--base=$(SSM_BASE) \
			--uninstall

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
