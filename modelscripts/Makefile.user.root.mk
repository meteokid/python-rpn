ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$modelscripts/Makefile.user.root.mk)
$(info ## )
endif

components_install:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)
components_uninstall:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$modelscripts/Makefile.user.mk [END] ========================================)
endif
