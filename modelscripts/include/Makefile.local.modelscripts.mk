ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$modelscripts/include/Makefile.local.modelscripts.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

ifeq (,$(wildcard $(modelscripts)/VERSION))
   $(error Not found: $(modelscripts)/VERSION)
endif
MODELSCRIPTS_VERSION0  = $(shell cat $(modelscripts)/VERSION)
MODELSCRIPTS_VERSION   = $(notdir $(MODELSCRIPTS_VERSION0))
MODELSCRIPTS_VERSION_X = $(dir $(MODELSCRIPTS_VERSION0))

MODELSCRIPTS_SFX = $(RDE_LIBS_SFX)

MODELSCRIPTS_LIB_MERGED_NAME_0 = 
MODELSCRIPTS_LIBS_MERGED_0 = 
MODELSCRIPTS_LIBS_OTHER_0  = 
MODELSCRIPTS_LIBS_EXTRA_0  = 
MODELSCRIPTS_LIBS_ALL_0    = 
MODELSCRIPTS_LIBS_FL_0     = 
MODELSCRIPTS_LIBS_0        = 

MODELSCRIPTS_LIB_MERGED_NAME = $(foreach item,$(MODELSCRIPTS_LIB_MERGED_NAME_0),$(item)$(MODELSCRIPTS_SFX))
MODELSCRIPTS_LIBS_MERGED = $(foreach item,$(MODELSCRIPTS_LIBS_MERGED_0),$(item)$(MODELSCRIPTS_SFX))
MODELSCRIPTS_LIBS_OTHER  = $(foreach item,$(MODELSCRIPTS_LIBS_OTHER_0),$(item)$(MODELSCRIPTS_SFX))
MODELSCRIPTS_LIBS_EXTRA  = $(foreach item,$(MODELSCRIPTS_LIBS_EXTRA_0),$(item)$(MODELSCRIPTS_SFX))
MODELSCRIPTS_LIBS_ALL    = $(foreach item,$(MODELSCRIPTS_LIBS_ALL_0),$(item)$(MODELSCRIPTS_SFX))
MODELSCRIPTS_LIBS_FL     = $(foreach item,$(MODELSCRIPTS_LIBS_FL_0),$(item)$(MODELSCRIPTS_SFX))
MODELSCRIPTS_LIBS        = $(foreach item,$(MODELSCRIPTS_LIBS_0),$(item)$(MODELSCRIPTS_SFX))

MODELSCRIPTS_LIB_MERGED_NAME_V = $(foreach item,$(MODELSCRIPTS_LIB_MERGED_NAME),$(item)_$(MODELSCRIPTS_VERSION))
MODELSCRIPTS_LIBS_MERGED_V = $(foreach item,$(MODELSCRIPTS_LIBS_MERGED),$(item)_$(MODELSCRIPTS_VERSION))
MODELSCRIPTS_LIBS_OTHER_V  = $(foreach item,$(MODELSCRIPTS_LIBS_OTHER),$(item)_$(MODELSCRIPTS_VERSION))
MODELSCRIPTS_LIBS_EXTRA_V  = $(foreach item,$(MODELSCRIPTS_LIBS_EXTRA),$(item)_$(MODELSCRIPTS_VERSION))
MODELSCRIPTS_LIBS_ALL_V    = $(foreach item,$(MODELSCRIPTS_LIBS_ALL),$(item)_$(MODELSCRIPTS_VERSION))
MODELSCRIPTS_LIBS_FL_V     = $(foreach item,$(MODELSCRIPTS_LIBS_FL),$(item)_$(MODELSCRIPTS_VERSION))
MODELSCRIPTS_LIBS_V        = $(foreach item,$(MODELSCRIPTS_LIBS),$(item)_$(MODELSCRIPTS_VERSION))

MODELSCRIPTS_LIBS_FL_FILES       = $(foreach item,$(MODELSCRIPTS_LIBS_FL),$(LIBDIR)/lib$(item).a.fl)
MODELSCRIPTS_LIBS_ALL_FILES      = $(foreach item,$(MODELSCRIPTS_LIBS_ALL) $(MODELSCRIPTS_LIBS_EXTRA),$(LIBDIR)/lib$(item).a)
MODELSCRIPTS_LIBS_ALL_FILES_PLUS =  ## $(LIBDIR)/lib$(MODELSCRIPTS_LIB_MERGED_NAME).a $(MODELSCRIPTS_LIBS_ALL_FILES) $(MODELSCRIPTS_LIBS_FL_FILES)

OBJECTS_MERGED_modelscripts = $(foreach item,$(MODELSCRIPTS_LIBS_MERGED_0),$(OBJECTS_$(item)))

MODELSCRIPTS_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_modelscripts),$(item).[Mm][Oo][Dd])

MODELSCRIPTS_ABS        = 
MODELSCRIPTS_ABS_FILES  = 

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(MODELSCRIPTS_LIBS_V)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(MODELSCRIPTS_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(MODELSCRIPTS_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   MODELSCRIPTS_ABS_DEP = $(MODELSCRIPTS_LIBS_ALL_FILES_PLUS) $(RDE_ABS_DEP)
endif

##
.PHONY: modelscripts_vfiles
MODELSCRIPTS_VFILES = 
modelscripts_vfiles: $(MODELSCRIPTS_VFILES)

BUILDNAME = 

#---- ARCH Specific overrides -----------------------------------------

#---- Abs targets -----------------------------------------------------

## Modelscripts Targets
.PHONY: $(MODELSCRIPTS_ABS)

.PHONY: allbin_modelscripts allbincheck_modelscripts
allbin_modelscripts: | $(MODELSCRIPTS_ABS)
allbincheck_modelscripts:
	for item in $(MODELSCRIPTS_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

#---- Lib target - automated ------------------------------------------
modelscripts_LIB_template1 = \
$$(LIBDIR)/lib$(2)_$$($(3)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

LIB_template1fl = \
$$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a.fl: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a $$@

LIB_template2fl = \
$$(LIBDIR)/lib$(1).a.fl: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a.fl ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a.fl $$@

.PHONY: modelscripts_libs
modelscripts_libs: $(OBJECTS_modelscripts) $(MODELSCRIPTS_LIBS_ALL_FILES_PLUS) | $(MODELSCRIPTS_VFILES)
$(foreach item,$(MODELSCRIPTS_LIBS_ALL_0) $(MODELSCRIPTS_LIBS_EXTRA_0),$(eval $(call modelscripts_LIB_template1,$(item),$(item)$(MODELSCRIPTS_SFX),MODELSCRIPTS)))
$(foreach item,$(MODELSCRIPTS_LIBS_ALL) $(MODELSCRIPTS_LIBS_EXTRA),$(eval $(call LIB_template2,$(item),MODELSCRIPTS)))
$(foreach item,$(MODELSCRIPTS_LIBS_FL),$(eval $(call LIB_template1fl,$(item),MODELSCRIPTS)))
$(foreach item,$(MODELSCRIPTS_LIBS_FL),$(eval $(call LIB_template2fl,$(item),MODELSCRIPTS)))

modelscripts_libs_fl: $(MODELSCRIPTS_LIBS_FL_FILES)

$(LIBDIR)/lib$(MODELSCRIPTS_LIB_MERGED_NAME)_$(MODELSCRIPTS_VERSION).a: $(OBJECTS_MERGED_modelscripts) | $(MODELSCRIPTS_VFILES)
# 	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_modelscripts); mv $@_$$$$ $@
# $(LIBDIR)/lib$(MODELSCRIPTS_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(MODELSCRIPTS_LIB_MERGED_NAME)_$(MODELSCRIPTS_VERSION).a
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(MODELSCRIPTS_LIB_MERGED_NAME)_$(MODELSCRIPTS_VERSION).a $@

# $(LIBDIR)/lib$(MODELSCRIPTS_LIBS_SHARED_0)_$(MODELSCRIPTS_VERSION).so: $(OBJECTS_modelscripts) $(MODELSCRIPTS_LIBS_OTHER_FILES) | $(MODELSCRIPTS_VFILES)
# 	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_modelscripts)" ;\
# 	export RBUILD_LIBAPPL="$(MODELSCRIPTS_LIBS_OTHER) $(MODELSCRIPTS_LIBS_DEP)" ;\
# 	$(RBUILD4MPI_SO)
# 	ls -l $@
# $(LIBDIR)/lib$(MODELSCRIPTS_LIBS_SHARED_0).so: $(LIBDIR)/lib$(MODELSCRIPTS_LIBS_SHARED_0)_$(MODELSCRIPTS_VERSION).so
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(MODELSCRIPTS_LIBS_SHARED_0)_$(MODELSCRIPTS_VERSION).so $@
# 	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$modelscripts/include/Makefile.local.modelscripts.mk [END] ==================)
endif
