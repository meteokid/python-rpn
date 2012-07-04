SHELL = /bin/bash

include Makefile.config.mk

#---------------------
export VERSION := $(shell if [[ -f $(PWD)/VERSION ]] ; then cat $(PWD)/VERSION ; fi)
export TOPDIR  := $(PWD)
export NOPRINTDIR := --no-print-directory
export modelutils := $(shell echo $${modelutils:-$(PWD)})

ifeq ($(PWD),$(BUILDDIRROOT))
   $(error FATAL ERROR, BUILDDIRROOT cannot be PWD)
	$(MAKE) -f Makefile.base.mk error
endif
ifeq ($(PWD),$(DISTDIR))
   $(error FATAL ERROR, DISTDIR cannot be PWD)
	$(MAKE) -f Makefile.base.mk error
endif

#MAKEFILES = Makefile.base.mk

.DEFAULT: | VERSION
	version=$(VERSION) ;\
	if [[ x$(VERSION) == x ]] ; then \
		version=$(shell if [[ -f $(PWD)/VERSION ]] ; then cat VERSION ; fi) ;\
	fi ;\
	$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $@ VERSION=$${version}

default: | VERSION
	$(MAKE) $(NOPRINTDIR) $(MFLAGS) all

.PHONY: force_all
force_all: distclean buildall distall force_allssmpkgs

.PHONY: clean distclean
clean:
	chmod -R u+w $(BUILDDIRROOT) 2> /dev/null || true ;\
	rm -rf $(BUILDDIRROOT) 2> /dev/null || true
mostlyclean:
	chmod -R u+w $(DISTDIR) 2> /dev/null || true ;\
	rm -rf $(DISTDIR) 2> /dev/null || true
distclean: | clean mostlyclean


# .PHONY: check
# check:


.PHONY: versiontag
versiontag:
	rm -f VERSION ;\
	$(MAKE) $(NOPRINTDIR) $(MFLAGS) VERSION
VERSION:
	version=$(VERSION) ;\
	if [[ x$(VERSION) == x ]] ; then \
		version=$(shell if [[ x$(VERSION) == x ]] ; then mu.git_tag_echo | cut -d_ -f2 ; fi) ;\
	fi ;\
	if [[ x$${version} == x ]] ; then \
		echo "FATAL ERROR, Cannot get version info" ;\
		exit 1 ;\
	fi ;\
	echo $${version} > $@ ;\
	echo VERSION=$${version} ;\
