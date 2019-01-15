ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/Makefile.user.mk)
$(info ## )
endif

#Internal compiler error on: 
#(ifort) make adx_int_rhs.o adx_int_winds_glb.o adx_tracers_interp.o itf_ens_hzd.o yy2global.o FFLAGS='-g -traceback'

export ATM_MODEL_ISOFFICIAL := true
export BUILDNAME = 
RBUILD_EXTRA_OBJ0    := 

ifeq (,$(vgrid))
COMP_VGRID = vgrid
endif
ifeq (,$(rpncomm))
COMP_RPNCOMM = rpncomm
endif

COMPONENTS        := $(COMP_VGRID) $(COMP_RPNCOMM) modelutils rpnphy gemdyn gem
COMPONENTS_UC     := $(foreach item,$(COMPONENTS),$(call rdeuc,$(item)))

COMPONENTS2       := $(COMP_VGRID) $(COMP_RPNCOMM) modelutils rpnphy gemdyn $(COMPONENTS)
COMPONENTS2_UC    := $(foreach item,$(COMPONENTS2),$(call rdeuc,$(item)))

COMPONENTS_VFILES := $(foreach item,$(COMPONENTS2_UC),$($(item)_VFILES))

SRCPATH_INCLUDE := $(CONST_SRCPATH_INCLUDE)
VPATH           := $(CONST_VPATH) #$(ROOT)/$(CONST_BUILDSRC)

#------------------------------------

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

.PHONY: vfiles2 obj2 lib2 abs2
vfiles2: components_vfiles
obj2: components_objects
lib2: components_libs
abs2: components_abs

.PHONY: components_vfiles
components_vfiles: $(COMPONENTS_VFILES)


.PHONY: components_objects
components_objects: $(COMPONENTS_VFILES) $(OBJECTS)


.PHONY: components_libs
COMPONENTS_LIBS_FILES = $(foreach item,$(COMPONENTS_UC),$($(item)_LIBS_ALL_FILES_PLUS))
components_libs: $(COMPONENTS_VFILES) $(OBJECTS) $(COMPONENTS_LIBS_FILES)
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
components_ssm: $(COMPONENTS_SSM)
components_ssm_all: $(COMPONENTS_SSM_ALL)
components_ssm_arch: $(COMPONENTS_SSM_ARCH)


COMPONENTS_INSTALL_ALL := $(foreach item,$(COMPONENTS_UC),$($(item)_INSTALL))
COMPONENTS_UNINSTALL_ALL := $(foreach item,$(COMPONENTS_UC),$($(item)_UNINSTALL))
.PHONY: components_install components_uninstall
components_install: $(COMPONENTS_INSTALL_ALL)
components_uninstall: $(COMPONENTS_UNINSTALL_ALL)

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
