ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: Makefile.user.root.mk)
$(info ## )
endif

## The RDE_LOCAL_LIBS_ONLY=1 option will force
## compilation of whole code,
## production of libraries and
## building of the binaries with the local libs.

# export RDE_LOCAL_LIBS_ONLY=1

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.build.mk [END] =======================================)
endif
