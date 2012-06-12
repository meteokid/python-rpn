SHELL = /bin/bash

TOPDIR = $(PWD)
SRCDIR = $(TOPDIR)
VPATH = $(SRCDIR)

LIBNAME = _no-name_
MYFILENAME = _no-such-file_
MYPOSTFIX =

ifneq (,$(wildcard $(PWD)/Makefile.rules.mk))
   include Makefile.rules.mk
endif
ifneq (,$(wildcard $(PWD)/Makefile.dep.mk))
   include Makefile.dep.mk
endif
ifneq (,$(wildcard $(PWD)/mes_recettes))
   include $(PWD)/mes_recettes
endif
ifneq (,$(wildcard $(SRCDIR)/include/recettes))
   include $(SRCDIR)/include/recettes
endif
ifneq (,$(wildcard $(SRCDIR)/mes_recettes))
   include $(SRCDIR)/mes_recettes
endif

Makefile: | Makefile.rules.mk Makefile.dep.mk
	ln -s $(TOPDIR)/Makefile.subdir.mk $@ 2>/dev/null || true
Makefile.rules.mk:
	ln -s $(TOPDIR)/Makefile.rules.mk $@ 2>/dev/null || true
Makefile.dep.mk:
	here=$(PWD) ;\
	cd $(SRCDIR) ;\
	find . -type f | grep -v '/.*/' | s.dependencies.pl > $${here}/$@


.PHONY: objects
objects: $(OBJECTS)

.PHONY: libs
libs: lib$(LIBNAME).a
lib$(LIBNAME).a:
	$(AR) rv $@ *.o
	ln -s $@ lib$(LIBNAME)_$(VERSION).a

lib$(MYFILENAME)$(MYPOSTFIX).a:
	$(AR) rv $@ $(MYFILENAME).o
	ln -s $@ lib$(MYFILENAME)$(MYPOSTFIX)_$(VERSION).a

.PHONY: allabs
allabs: allbin_$(PKGNAME)
	ls -l $(PWD)/*