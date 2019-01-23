ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$scm/Makefile.user.root.mk)
$(info ## )
endif

components_install:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)
components_uninstall:
	$(MYTIME) $(MAKE) -f Makefile.build.mk $(NOPRINTDIR) $@ $(MYMAKE_VARS)

# export RDE_USE_FULL_VPATH=1

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$scm/Makefile.user.mk [END] ========================================)
endif
