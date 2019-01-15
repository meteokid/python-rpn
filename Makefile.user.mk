ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/Makefile.user.mk)
$(info ## )
endif

# export ATM_MODEL_ISOFFICIAL := true
export BUILDNAME = 
RBUILD_EXTRA_OBJ0    := 

ifeq (,$(vgrid))
COMP_VGRID = vgrid
endif
ifeq (,$(rpncomm))
COMP_RPNCOMM = rpncomm
endif

COMPONENTS        := $(RDECOMPONENTS)
COMPONENTS_UC     := $(foreach item,$(COMPONENTS),$(call rdeuc,$(item)))

COMPONENTS2       := $(RDECOMPONENTS) $(COMPONENTS)
COMPONENTS2_UC    := $(foreach item,$(COMPONENTS2),$(call rdeuc,$(item)))

COMPONENTS_VFILES := $(foreach item,$(COMPONENTS2_UC),$($(item)_VFILES))

# SRCPATH_INCLUDE := $(CONST_SRCPATH_INCLUDE) $(CONST_SRCPATH)
# VPATH           := $(CONST_VPATH) #$(ROOT)/$(CONST_BUILDSRC)

#------------------------------------

#TODO: need target to check all bndl or even better to update bndl from local components's version numbers
#TODO: need target to link to CMC/Science bndl
## for item in $(ls */ssmusedep*-${RDENETWORK}) ; do ln -sf ${item##*/} ${item%-*} ; do


MYSSMINCLUDEMK0 = $(foreach item,$(COMPONENTS),$($(item))/include/Makefile.ssm.mk)
MYSSMINCLUDEMK = $(wildcard $(RDE_INCLUDE0)/Makefile.ssm.mk $(MYSSMINCLUDEMK0))
ifneq (,$(MYSSMINCLUDEMK))
   ifneq (,$(DEBUGMAKE))
      $(info include $(MYSSMINCLUDEMK))
   endif
   include $(MYSSMINCLUDEMK)
endif

ifeq (,$(DIRORIG_gem))
DIRORIG_gem = $(gem)
endif

#------------------------------------
ifeq (-d,$(RDE_BUILDDIR_SFX))
COMP_RULES_FILE = $(MODELUTILS_COMP_RULES_DEBUG)
#ifeq (intel13sp1u2,$(CONST_RDE_COMP_ARCH))
#FFLAGS  = -C -ftrapuv #-warn all -warn nounused 
#else
#FFLAGS  = -C
#endif
endif

.PHONY: vfiles obj lib abs ssm ssmall ssmarch
vfiles: components_vfiles
objects: components_objects
libs: components_libs
abs: components_abs
ssm: components_ssm
ssmall: components_ssm_all
ssmarch: components_ssm_arch

.PHONY: components_vfiles
components_vfiles: $(COMPONENTS_VFILES)
	ls $(COMPONENTS_VFILES) | sort -u


.PHONY: components_objects
COMPONENTS_OBJECTS = $(foreach item,$(COMPONENTS),$(OBJECTS_$(item)))
components_objects: $(COMPONENTS_VFILES) $(COMPONENTS_OBJECTS)


.PHONY: components_libs
COMPONENTS_LIBS_FILES = $(foreach item,$(COMPONENTS_UC),$($(item)_LIBS_ALL_FILES_PLUS))
components_libs: $(COMPONENTS_VFILES) $(COMPONENTS_OBJECTS) $(COMPONENTS_LIBS_FILES)
	ls -l $(COMPONENTS_LIBS_FILES)
	ls -lL $(COMPONENTS_LIBS_FILES)


COMPONENTS_ABS  := $(foreach item,$(COMPONENTS_UC),$($(item)_ABS))
COMPONENTS_ABS_FILES  := $(foreach item,$(COMPONENTS_UC),$($(item)_ABS_FILES))
.PHONY: components_abs components_abs_check
components_abs: $(COMPONENTS_ABS)
	ls -l $(COMPONENTS_ABS_FILES)

COMPONENTS_SSM_ALL  := $(foreach item,$(COMPONENTS_UC),$($(item)_SSMALL_FILES))
COMPONENTS_SSM_ARCH := $(foreach item,$(COMPONENTS_UC),$($(item)_SSMARCH_FILES))
COMPONENTS_SSM := $(COMPONENTS_SSM_ALL) $(COMPONENTS_SSM_ARCH)
.PHONY: components_ssm
components_ssm: components_ssm_all components_ssm_arch
components_ssm_all: $(COMPONENTS_SSM_ALL)
	for i in $(COMPONENTS) ; do \
		if [[ -f $(ROOT)/$$i/Makefile ]] ; then \
			cd $(ROOT)/$$i && \
			make -f Makefile $${i}_ssm_all.ssm || true ; \
		fi ; \
	done
components_ssm_arch: $(COMPONENTS_SSM_ARCH)
	for i in $(COMPONENTS) ; do \
		if [[ -f $(ROOT)/$$i/Makefile ]] ; then \
			cd $(ROOT)/$$i && \
			make -f Makefile $${i}_ssm_arch.ssm || true ; \
		fi ; \
	done

COMPONENTS_INSTALL_ALL := $(foreach item,$(COMPONENTS_UC),$($(item)_INSTALL))
COMPONENTS_UNINSTALL_ALL := $(foreach item,$(COMPONENTS_UC),$($(item)_UNINSTALL))
$(info COMPONENTS_INSTALL_ALL=$(COMPONENTS_INSTALL_ALL))
.PHONY: components_install components_uninstall
components_install: $(COMPONENTS_INSTALL_ALL)
	for i in $(COMPONENTS) ; do \
		if [[ -f $(ROOT)/$$i/Makefile ]] ; then \
			cd $(ROOT)/$$i && \
			make -f Makefile $${i}_install CONFIRM_INSTALL=$(CONFIRM_INSTALL) || true ; \
		fi ; \
	done
components_uninstall: $(COMPONENTS_UNINSTALL_ALL)
	for i in $(COMPONENTS) ; do \
		if [[ -f $(ROOT)/$$i/Makefile ]] ; then \
			cd $(ROOT)/$$i && \
			make -f Makefile $${i}_uninstall UNINSTALL_CONFIRM=$(UNINSTALL_CONFIRM) || true ; \
		fi ; \
	done

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
