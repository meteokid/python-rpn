# Top-level makefile for building the shared library dependencies.

# Final location to install the shared libraries.
# If this Makefile is invoked on the command-line, then assume an in-place
# build.
ifeq ($(SHAREDLIB_DIR),)
SHAREDLIB_DIR=$(PWD)/../lib/rpnpy/_sharedlibs
endif

# Location to do the build.
# If invoked on the command-line, then build in-place in this directory.
ifeq ($(BUILDDIR),)
BUILDDIR=$(PWD)
endif

all: sharedlibs

include include/libs.mk

sharedlibs: $(LIBRMN_SHARED) $(LIBDESCRIP_SHARED) $(LIBBURPC_SHARED)
	# Copy extra libraries needed for runtime
	[ -z "$(EXTRA_LIBS)" ] || cp -L $(EXTRA_LIBS) $(SHAREDLIB_DIR)
