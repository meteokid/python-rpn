ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rde/include/Makefile.local.rde.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

ifeq (,$(wildcard $(rde)/VERSION))
   $(error Not found: $(rde)/VERSION)
endif
RDE_VERSION0  = $(shell cat $(rde)/VERSION)
RDE_VERSION   = $(notdir $(RDE_VERSION0))
RDE_VERSION_X = $(dir $(RDE_VERSION0))

RDE_SFX = $(RDE_LIBS_SFX)

RDE_LIB_MERGED_NAME_0 = 
RDE_LIBS_MERGED_0 = 
RDE_LIBS_OTHER_0  = 
RDE_LIBS_EXTRA_0  = 
RDE_LIBS_ALL_0    = 
RDE_LIBS_FL_0     = 
RDE_LIBS_0        = 

RDE_LIB_MERGED_NAME = $(foreach item,$(RDE_LIB_MERGED_NAME_0),$(item)$(RDE_SFX))
RDE_LIBS_MERGED = $(foreach item,$(RDE_LIBS_MERGED_0),$(item)$(RDE_SFX))
RDE_LIBS_OTHER  = $(foreach item,$(RDE_LIBS_OTHER_0),$(item)$(RDE_SFX))
RDE_LIBS_EXTRA  = $(foreach item,$(RDE_LIBS_EXTRA_0),$(item)$(RDE_SFX))
RDE_LIBS_ALL    = $(foreach item,$(RDE_LIBS_ALL_0),$(item)$(RDE_SFX))
RDE_LIBS_FL     = $(foreach item,$(RDE_LIBS_FL_0),$(item)$(RDE_SFX))
RDE_LIBS        = $(foreach item,$(RDE_LIBS_0),$(item)$(RDE_SFX))

RDE_LIB_MERGED_NAME_V = $(foreach item,$(RDE_LIB_MERGED_NAME),$(item)_$(RDE_VERSION))
RDE_LIBS_MERGED_V = $(foreach item,$(RDE_LIBS_MERGED),$(item)_$(RDE_VERSION))
RDE_LIBS_OTHER_V  = $(foreach item,$(RDE_LIBS_OTHER),$(item)_$(RDE_VERSION))
RDE_LIBS_EXTRA_V  = $(foreach item,$(RDE_LIBS_EXTRA),$(item)_$(RDE_VERSION))
RDE_LIBS_ALL_V    = $(foreach item,$(RDE_LIBS_ALL),$(item)_$(RDE_VERSION))
RDE_LIBS_FL_V     = $(foreach item,$(RDE_LIBS_FL),$(item)_$(RDE_VERSION))
RDE_LIBS_V        = $(foreach item,$(RDE_LIBS),$(item)_$(RDE_VERSION))

RDE_LIBS_FL_FILES       = $(foreach item,$(RDE_LIBS_FL),$(LIBDIR)/lib$(item).a.fl)
RDE_LIBS_ALL_FILES      = $(foreach item,$(RDE_LIBS_ALL) $(RDE_LIBS_EXTRA),$(LIBDIR)/lib$(item).a)
RDE_LIBS_ALL_FILES_PLUS =  ## $(LIBDIR)/lib$(RDE_LIB_MERGED_NAME).a $(RDE_LIBS_ALL_FILES) $(RDE_LIBS_FL_FILES)

OBJECTS_MERGED_rde = $(foreach item,$(RDE_LIBS_MERGED_0),$(OBJECTS_$(item)))

RDE_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_rde),$(item).[Mm][Oo][Dd])

RDE_ABS        = 
RDE_ABS_FILES  = 

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(RDE_LIBS_V)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(RDE_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(RDE_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   RDE_ABS_DEP = $(RDE_LIBS_ALL_FILES_PLUS) $(RDE_ABS_DEP)
endif

##
.PHONY: rde_vfiles
RDE_VFILES = 
rde_vfiles: $(RDE_VFILES)

BUILDNAME = 

#---- ARCH Specific overrides -----------------------------------------

#---- Abs targets -----------------------------------------------------

## Rde Targets
.PHONY: $(RDE_ABS)

.PHONY: allbin_rde allbincheck_rde
allbin_rde: | $(RDE_ABS)
allbincheck_rde:
	for item in $(RDE_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

#---- Lib target - automated ------------------------------------------
rde_LIB_template1 = \
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

.PHONY: rde_libs
rde_libs: $(OBJECTS_rde) $(RDE_LIBS_ALL_FILES_PLUS) | $(RDE_VFILES)
$(foreach item,$(RDE_LIBS_ALL_0) $(RDE_LIBS_EXTRA_0),$(eval $(call rde_LIB_template1,$(item),$(item)$(RDE_SFX),RDE)))
$(foreach item,$(RDE_LIBS_ALL) $(RDE_LIBS_EXTRA),$(eval $(call LIB_template2,$(item),RDE)))
$(foreach item,$(RDE_LIBS_FL),$(eval $(call LIB_template1fl,$(item),RDE)))
$(foreach item,$(RDE_LIBS_FL),$(eval $(call LIB_template2fl,$(item),RDE)))

rde_libs_fl: $(RDE_LIBS_FL_FILES)

$(LIBDIR)/lib$(RDE_LIB_MERGED_NAME)_$(RDE_VERSION).a: $(OBJECTS_MERGED_rde) | $(RDE_VFILES)
# 	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_rde); mv $@_$$$$ $@
# $(LIBDIR)/lib$(RDE_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(RDE_LIB_MERGED_NAME)_$(RDE_VERSION).a
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(RDE_LIB_MERGED_NAME)_$(RDE_VERSION).a $@

# $(LIBDIR)/lib$(RDE_LIBS_SHARED_0)_$(RDE_VERSION).so: $(OBJECTS_rde) $(RDE_LIBS_OTHER_FILES) | $(RDE_VFILES)
# 	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_rde)" ;\
# 	export RBUILD_LIBAPPL="$(RDE_LIBS_OTHER) $(RDE_LIBS_DEP)" ;\
# 	$(RBUILD4MPI_SO)
# 	ls -l $@
# $(LIBDIR)/lib$(RDE_LIBS_SHARED_0).so: $(LIBDIR)/lib$(RDE_LIBS_SHARED_0)_$(RDE_VERSION).so
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(RDE_LIBS_SHARED_0)_$(RDE_VERSION).so $@
# 	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rde/include/Makefile.local.rde.mk [END] ==================)
endif
