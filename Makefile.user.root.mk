ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/Makefile.user.root.mk)
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
distclean: $(RDE_TARGET_DISTCLEAN) extrafilesclean


ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
