ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$mig/Makefile.user.root.mk)
$(info ## )
endif

components_install:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)
components_uninstall:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)

export RDE_USE_FULL_VPATH=1

RDE_TARGET_DISTCLEAN = distclean0
.PHONY: distclean extrafilesclean
extrafilesclean:
	## rm -rf GEM_cfg*  # should these be removed as well?... only if not in git
	rm -rf .rde* BINMOD
	rm -f  .linkit.log .ssmuse_gem suite PREP RUNMOD
	rm -f  */ssmusedep*bndl gem/ATM_MODEL_*

distclean: $(RDE_TARGET_DISTCLEAN) extrafilesclean


ifneq (,$(DEBUGMAKE))
$(info ## ==== $$mig/Makefile.user.mk [END] ========================================)
endif
