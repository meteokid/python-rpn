ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$migdep/include/Makefile.local.migdep.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

ifeq (,$(wildcard $(migdep)/VERSION))
   $(error Not found: $(migdep)/VERSION)
endif
MIGDEP_VERSION0  = $(shell cat $(migdep)/VERSION)
MIGDEP_VERSION   = $(notdir $(MIGDEP_VERSION0))
MIGDEP_VERSION_X = $(dir $(MIGDEP_VERSION0))

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$migdep/include/Makefile.local.migdep.mk [END] ==================)
endif
