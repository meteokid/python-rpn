ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$gem/Makefile.user.root.mk)
$(info ## )
endif

components_install:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)
components_uninstall:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
