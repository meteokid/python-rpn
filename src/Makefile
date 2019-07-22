# Top-level makefile for building the shared library dependencies.

all: sharedlibs

include include/libs.mk

sharedlibs: $(LIBRMN_SHARED) $(LIBDESCRIP_SHARED) $(LIBBURPC_SHARED)
	# Copy extra libraries needed for runtime
	[ -z "$(EXTRA_LIBS)" ] || cp -L $(EXTRA_LIBS) $(SHAREDLIB_DIR)
