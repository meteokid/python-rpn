SHELL = /bin/bash

#----
ifeq (,$(modelutils))
   $(error FATAL ERROR, modelutils is not defined)
endif
ifeq (,$(PKGNAME))
   $(error FATAL ERROR, PKGNAME is not defined)
endif
ifeq (,$(VERSION))
   $(error FATAL ERROR, VERSION is not defined)
endif
ifeq (,$(EC_ARCH))
   $(error FATAL ERROR, EC_ARCH is not defined)
endif
ifeq ($(BASE_ARCH),$(EC_ARCH))
   $(error FATAL ERROR, EC_ARCH is not fully defined)
endif

ifneq (,$(wildcard $(modelutils)/include/recettes))
   include $(modelutils)/include/recettes
endif
ifneq (,$(wildcard $(PWD)/Makefile.rules.mk))
   include $(PWD)/Makefile.rules.mk
endif
ifneq (,$(wildcard $(PWD)/include/recettes))
ifneq ($(modelutils)/include/recettes,$(PWD)/include/recettes)
   include $(PWD)/include/recettes
endif
endif

#----
.DEFAULT: all
.PHONY: buildall distall all
buildall: libs allabs
distall: distdir 
all: allssmpkgs

VPATH = $(SRCDIR)

BUILDDIRARCH = $(BUILDDIRROOT)/$(EC_ARCH)

MODPATH = $(addprefix $(BUILDDIRARCH)/,$(SUBDIRS_LIB))
INCPATH = $(INCDIR0)/$(EC_ARCH) $(INCDIR0)/$(BASE_ARCH) $(INCDIR0) $(SRCDIR) $(addprefix $(SRCDIR)/,$(SUBDIRS_SRC))


## Target: dist (export full, not build,pkg)
.PHONY: dist
dist: | $(DISTDIR) $(DISTDIR)/$(PKGNAME)_$(VERSION).tgz
$(DISTDIR)/$(PKGNAME)_$(VERSION).tgz: | $(BUILDDIRROOT)/disttgz/$(PKGNAME)_$(VERSION)
	cd $(BUILDDIRROOT)/disttgz ;\
	$(TAR) czf $@ $(PKGNAME)_$(VERSION)/ ;\
	ls -l $@
$(BUILDDIRROOT)/disttgz/$(PKGNAME)_$(VERSION): | $(BUILDDIRROOT)
	mkdir -p $@ 2> /dev/null || true ;\
	rsync -a --delete $(TOPDIR)/ $@/


## Target: builddir (Create Build directory with basic files)
.PHONY: builddir
builddir: | $(BUILDDIRARCH)
	ls -ld $(BUILDDIRARCH)
$(BUILDDIRROOT):
	mkdir -p $@ 2>/dev/null || true
$(BUILDDIRARCH):
	mkdir -p $@ 2>/dev/null || true ;\
	for mysubdir in $(SUBDIRS_LIB) ; do \
		mkdir -p $@/$${mysubdir} 2>/dev/null || true ;\
		cd $@/$${mysubdir} ;\
		$(MAKE) -f $(TOPDIR)/Makefile.subdir.mk $(MFLAGS) $(NOPRINTDIR) \
			Makefile \
			TOPDIR=$(TOPDIR) SRCDIR=$(SRCDIR)/$${mysubdir} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
		ls -ld $@/$${mysubdir} ;\
	done

.PHONY: versionfiles
versionfiles: $(BUILDDIRARCH)/include/$(PKGNAME)_version.h $(BUILDDIRARCH)/include/$(PKGNAME)_version.inc $(BUILDDIRARCH)/include/$(PKGNAME)_version.dot $(BUILDDIRARCH)/.VERSION
$(BUILDDIRARCH)/include: | builddir
	mkdir -p $@ 2>/dev/null || true
$(BUILDDIRARCH)/include/$(PKGNAME)_version.h: | $(BUILDDIRARCH)/include
	mu.mk_version_file "$(PKGNAME)" "$(VERSION)" $(BUILDDIRARCH)/include c
$(BUILDDIRARCH)/include/$(PKGNAME)_version.inc: | $(BUILDDIRARCH)/include
	mu.mk_version_file "$(PKGNAME)" "$(VERSION)" $(BUILDDIRARCH)/include f
$(BUILDDIRARCH)/include/$(PKGNAME)_version.dot: | $(BUILDDIRARCH)/include
	mu.mk_version_file "$(PKGNAME)" "$(VERSION)" $(BUILDDIRARCH)/include sh
$(BUILDDIRARCH)/.VERSION: | $(BUILDDIRARCH)
	mu.mk_version_file "$(PKGNAME)" "$(VERSION)" $(BUILDDIRARCH) sh s


## Target: objects (compile everything)
.PHONY: objects
objects: | builddir versionfiles
	for mysubdir in $(SUBDIRS_LIB) ; do \
		cd $(BUILDDIRARCH)/$${mysubdir} ;\
		$(MAKE) $(NOPRINTDIR) $(MFLAGS) $@ \
			TOPDIR=$(TOPDIR) \
			SRCDIR=$(SRCDIR)/$${mysubdir} \
			EC_INCLUDE_PATH="$(INCPATH) $(BUILDDIRARCH)/include $(MODPATH) $(EC_INCLUDE_PATH)" ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
	done ;\


## Target: libs (creat all libs after compiling)
.PHONY: libs
libs: objects | $(BUILDDIRARCH)/lib$(PKGNAME).a
	for mysubdir in $(SUBDIRS_LIB_PRE) $(SUBDIRS_LIB_POST) ; do \
		cd $(BUILDDIRARCH)/$${mysubdir} ;\
		$(MAKE) $(NOPRINTDIR) $(MFLAGS) $@ \
			TOPDIR=$(TOPDIR) \
			LIBNAME=$(PKGNAME)_$${mysubdir} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
		ls -l $(BUILDDIRARCH)/$${mysubdir}/lib$(PKGNAME)_$${mysubdir}.a ;\
	done ;\
	for mysubdir in $(SUBDIRS_LIB_SPLIT) ; do \
		cd $(BUILDDIRARCH)/$${mysubdir} ;\
		for myfile in `ls *.o` ; do \
			$(MAKE) $(NOPRINTDIR) $(MFLAGS) lib$${myfile%.*}.a \
				TOPDIR=$(TOPDIR) \
				MYFILENAME=$${myfile%.*} ;\
			if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
			ls -l $(BUILDDIRARCH)/$${mysubdir}/lib$${myfile%.*}.a ;\
		done ;\
	done ;\
	ls -l $(BUILDDIRARCH)/lib$(PKGNAME).a
$(BUILDDIRARCH)/lib$(PKGNAME).a:
	for mysubdir in $(SUBDIRS_LIB_MERGE) ; do \
		$(AR) rv $@ $(BUILDDIRARCH)/$${mysubdir}/*.o ;\
	done ;\
	cd $(BUILDDIRARCH) ;\
	ln -s lib$(PKGNAME).a lib$(PKGNAME)_$(VERSION).a


## Target: allabs (Build all executable)
.PHONY: allabs
allabs: libs | $(BUILDDIRARCH)/bin $(BUILDDIRARCH)/binwrk/Makefile
	cd $(BUILDDIRARCH)/binwrk ;\
	$(MAKE) $(NOPRINTDIR) $(MFLAGS) allbin_$(PKGNAME) \
		TOPDIR=$(TOPDIR) \
		BINDIR=$(BUILDDIRARCH)/bin \
		LCLPO=. \
		EC_LD_LIBRARY_PATH="$(BUILDDIRARCH) $(addprefix $(BUILDDIRARCH)/,$(SUBDIRS_LIB)) $(EC_LD_LIBRARY_PATH)" \
		EC_INCLUDE_PATH="$(INCPATH) $(BUILDDIRARCH)/include $(MODPATH) $(EC_INCLUDE_PATH)" ;\
	if [[ $${?} != 0 ]] ; then exit 1 ; fi
$(BUILDDIRARCH)/binwrk/Makefile: | $(BUILDDIRARCH)/binwrk
	cd $(BUILDDIRARCH)/binwrk ;\
	$(MAKE) -f $(TOPDIR)/Makefile.subdir.mk $(NOPRINTDIR) $(MFLAGS) Makefile \
		TOPDIR=$(TOPDIR) SRCDIR=$(BUILDDIRARCH)/binwrk ;\
	if [[ $${?} != 0 ]] ; then exit 1 ; fi
$(BUILDDIRARCH)/binwrk:
	mkdir -p $@ 2>/dev/null || true
$(BUILDDIRARCH)/bin:
	mkdir -p $@ 2>/dev/null || true


## Target: distdir (copy all files into ssm organized dirs)
.PHONY: distdir distdirall distdirarch distdirmulti
distdir:
	for mytype in $(SSM_PKG); do \
		$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) distdir$${mytype} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
	done
distdirall: | $(DISTDIR)/$(PKGNAME)_$(VERSION)_all
	ls -ld $(DISTDIR)/$(PKGNAME)_$(VERSION)_all
distdirarch: | $(DISTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH)
	ls -ld $(DISTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH)
distdirmulti: | $(DISTDIR)/$(PKGNAME)_$(VERSION)_multi
	ls -ld $(DISTDIR)/$(PKGNAME)_$(VERSION)_multi
$(DISTDIR):
	mkdir -p $@ 2>/dev/null || true
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all: | $(DISTDIR) $(BUILDINFO)
	(cd $(DISTDIR) ;\
	$(TAR) xzf $(SSMTMPLDIR)/$(SSMTMPLNAMEARCH).ssm ;\
	mv -f $(SSMTMPLNAMEARCH) $@ 2>/dev/null || true) ;\
	echo $(shell mu.mk_ssm_control $(PKGNAME) $(VERSION) "all ; $(BASE_ARCH)" $(TOPDIR)/BUILDINFO $(TOPDIR)/DESCRIPTION > $(DISTDIR)/SSMCONTROLFILE_all) ;\
	mv -f $(DISTDIR)/SSMCONTROLFILE_all $@/.ssm.d/control ;\
	for mydir in $(SUBDIRS_SSMALL) ; do \
		$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $@/$${mydir} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
	done ;\
	mkdir -p $@/lib 2>/dev/null || true ;
	touch $@/lib/libdummy.a 2>/dev/null || true
	#touch is a patch to force smm to add $@/include to EC_INCLUDE_PATH
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all/src: | $(DISTDIR)/$(PKGNAME)_$(VERSION)_all/include
	mkdir -p $@ 2>/dev/null || true ;\
	for mydir in $(SUBDIRS_SRC) ; do \
		rsync -a --delete $(SRCDIR)/$${mydir} $@ ;\
	done ;\
	find $@ -type f -exec chmod a-wx+r {} \; ;\
	cd $@ && ln -s ../include include
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all/RCS: $(DISTDIR)/$(PKGNAME)_$(VERSION)_all/src
	mkdir -p $@ 2>/dev/null || true ;\
	(cd $(INCDIR0) ;\
	cp .[rc]* $@/ 2>/dev/null || true ; \
	for item in $(shell ls -1 recettes* cibles* 2>/dev/null); do \
		cp $${item} $@/.$${item}  2>/dev/null || true ;\
	done) ;\
	cd $(DISTDIR)/$(PKGNAME)_$(VERSION)_all/src ;\
	for mydir in $(SUBDIRS_SRC) ; do\
		MYDIR=$@_`echo $${mydir} | tr 'a-z' 'A-Z'` ;\
		rm -rf $${MYDIR} 2>/dev/null || true;\
		(cd $${mydir} && mu.src2rcs . && mv RCS $${MYDIR}) || true ;\
	done
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all/include:
	mkdir -p $@ 2>/dev/null || true ;\
	rsync -a --delete $(INCDIR0)/ $@/ ;\
	find $@ -type f -exec chmod a-wx+r {} \;
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all/lib:
	mkdir -p $@ 2>/dev/null || true ;\
	rsync -a --delete $(LIBDIR0)/ $@/ ;\
	find $@ -type f -exec chmod a-wx+r {} \;
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all/bin:
	mkdir -p $@ 2>/dev/null || true ;\
	rsync -a --delete $(BINDIR0)/ $@/ ;\
	find $@ -type f -exec chmod a-w+rx {} \;
$(DISTDIR)/$(PKGNAME)_$(VERSION)_all/share:
	mkdir -p $@ 2>/dev/null || true ;\
	rsync -a --delete $(SHAREDIR)/ $@/ ;\
	find $@ -type f -exec chmod a-wx+r {} \;

$(DISTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH): | $(DISTDIR) $(BUILDINFO)
	(cd $(DISTDIR) ;\
	$(TAR) xzf $(SSMTMPLDIR)/$(SSMTMPLNAMEARCH).ssm ;\
	mv -f $(SSMTMPLNAMEARCH) $@ 2>/dev/null || true) ;\
	echo $(shell mu.mk_ssm_control $(PKGNAME) $(VERSION) "$(SSMARCH) ; $(BASE_ARCH)" $(TOPDIR)/BUILDINFO $(TOPDIR)/DESCRIPTION > $(DISTDIR)/SSMCONTROLFILE_$(SSMARCH)) ;\
	mv -f $(DISTDIR)/SSMCONTROLFILE_$(SSMARCH) $@/.ssm.d/control ;\
	for mydir in $(SUBDIRS_SSMARCH) ; do \
		$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $@/$${mydir} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
	done 
$(DISTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH)/include/$(EC_ARCH): objects
	mkdir -p $@ 2>/dev/null || true ;\
	cp $(BUILDDIRARCH)/*.mod $(BUILDDIRARCH)/*/*.mod $@/ 2>/dev/null || true ;\
	cp $(BUILDDIRARCH)/include/$(PKGNAME)_version.* $@/ 2>/dev/null || true
#TODO: link to oldarch
$(DISTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH)/lib/$(EC_ARCH): libs
	mkdir -p $@ 2>/dev/null || true ;\
	cp $(BUILDDIRARCH)/lib*.a $(BUILDDIRARCH)/lib*.a.fl $(BUILDDIRARCH)/*/lib*.a $(BUILDDIRARCH)/*/lib*.a.fl $@/ 2>/dev/null || true ;\
	cp $(BUILDDIRARCH)/.VERSION $@/ 2>/dev/null || true
#TODO: link to oldarch
$(DISTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH)/bin/$(BASE_ARCH): allabs
	mkdir -p $@ 2>/dev/null || true ;\
	cp $(BUILDDIRARCH)/bin/* $@/ 2>/dev/null || true ;\
	rm -f $@/Makefile* $@/cibles* $@/recettes* 2>/dev/null || true 

#TODO: $(DISTDIR)/$(PKGNAME)_$(VERSION)_multi


## Target: allssmpkgs (tar pre-filled ssm dir)
.PHONY: force_allssmpkgs allssmpkgs ssmpkgall ssmpkgarch ssmpkgmulti
force_allssmpkgs:
	for mytype in $(SSM_PKG); do \
		if [[ x$${mytype} == xarch ]] ; then\
			rm -f $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH).ssm 2>/dev/null || true ;\
		else \
			rm -f $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_$${mytype}.ssm 2>/dev/null || true ;\
		fi ;\
		$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) ssmpkg$${mytype} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
	done
allssmpkgs:
	for mytype in $(SSM_PKG); do \
		$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) ssmpkg$${mytype} ;\
		if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
	done
ssmpkgall: | $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_all.ssm
	ls -ld $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_all.ssm
ssmpkgarch: | $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH).ssm
	ls -ld $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH).ssm
ssmpkgmulti: | $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_multi.ssm
	ls -ld $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_multi.ssm
$(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_all.ssm: | distdirall
	cd $(DISTDIR) ;\
	echo $(SSMDEPOTDIR) > .SsmDepot ;\
	$(PKGNAME)_$(VERSION)_all/maint/make-ssm
$(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_multi.ssm: | distdirmulti
	cd $(DISTDIR) ;\
	echo $(SSMDEPOTDIR) > .SsmDepot ;\
	$(PKGNAME)_$(VERSION)_multi/maint/make-ssm
$(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_$(SSMARCH).ssm: | distdirarch
	cd $(DISTDIR) ;\
	echo $(SSMDEPOTDIR) > .SsmDepot ;\
	$(PKGNAME)_$(VERSION)_$(SSMARCH)/maint/make-ssm
BUILDINFO:
	echo "Dependencies (s.ssmuse.dot): " > $@
	cat ssmuse_dependencies.bndl >> $@


## Target: install (install/publish ssm pkgs)
.PHONY: install install_domain install_pkgs install_pkgs_all install_pkgs_arch install_pkgs_multi install_bndl
#TODO: domain name should be: $(PKGNAME)_$(VERSION) or $(PKGNAME)/$(PKGNAME)_$(VERSION)
install: install_domain install_pkgs install_bndl
install_domain: $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)
install_pkgs: | install_domain
	for mytype in $(SSM_PKG); do \
		$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) \
			install_pkgs_$${mytype} ;\
		if [[ $${?} != 0 ]] ; then echo $${?} exit 1 ; fi ;\
	done
install_pkgs_all: ssmpkgall install_domain
	$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)/$(PKGNAME)_$(VERSION)_all SSMARCH=all ;\
	if [[ $${?} != 0 ]] ; then exit 1 ; fi
install_pkgs_arch: ssmpkgarch install_domain
	for myfile in $(wildcard $(SSMDEPOTDIR)/$(PKGNAME)_$(VERSION)_*.ssm); do \
		mytype=`echo $${myfile##*_}` ; mytype=`echo $${mytype%.*}` ;\
		if [[ x$${mytype} != xall ]] ; then \
			$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) \
				$(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)/$(PKGNAME)_$(VERSION)_$${mytype} SSMARCH=$${mytype} ;\
			if [[ $${?} != 0 ]] ; then exit 1 ; fi ;\
		fi ;\
	done
install_pkgs_multi: ssmpkgmulti install_domain 
	$(MAKE) -f Makefile.base.mk $(NOPRINTDIR) $(MFLAGS) $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)/$(PKGNAME)_$(VERSION)_multi SSMARCH=multi ;\
	if [[ $${?} != 0 ]] ; then exit 1 ; fi
install_bndl: install_domain $(SSMINSTALLDIRBNDL)/$(PKGNAME)/$(VERSION).bndl

$(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION):
	mu.mkdir_tree $(DESTDIR) $(SSM_RELDIRDOM)/$(PKGNAME)/$(VERSION) $(SSMPOSTCHMOD) ;\
	chmod u+w $@ ;\
	cd $@ ;\
	mydomain=`true_path .` ;\
	ssm created -d $${mydomain} --defaultRepositorySource $(SSMDEPOTDIR) --yes ;\
	if [[ ! -e $${mydomain}/.SsmDepot ]] ; then echo $(SSMDEPOTDIR) > $@/.SsmDepot ; fi ;\
	chmod $(SSMPOSTCHMOD) $(SSMINSTALLDIRDOM)
$(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)/$(PKGNAME)_$(VERSION)_$(SSMARCH): | $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)
	cd $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION) ;\
	mydomain=`true_path .` ;\
	mu.chmod_ssm_dom u+w $${mydomain} ;\
	ssm install -d $${mydomain} -p $(PKGNAME)_$(VERSION)_$(SSMARCH) --yes --skipOnInstalled ;\
	ssm publish -d $${mydomain} -p $(PKGNAME)_$(VERSION)_$(SSMARCH) --yes --skipOnPublished ;\
	mu.chmod_ssm_dom $(SSMPOSTCHMOD) $${mydomain}
$(SSMINSTALLDIRBNDL)/$(PKGNAME)/$(VERSION).bndl: | $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(VERSION)
	mu.mkdir_tree $(DESTDIR) $(SSM_RELDIRBNDL)/$(PKGNAME) $(SSMPOSTCHMOD) ;\
	chmod u+w $(SSMINSTALLDIRBNDL)/$(PKGNAME) ;\
	cat $(SSMDEPENDENCIES) > $@ ;\
	echo $(SSM_RELDIRDOM)$(PKGNAME)/$(VERSION) >> $@ ;\
	chmod $(SSMPOSTCHMOD) $(SSMINSTALLDIRBNDL)/$(PKGNAME) $@ ;\
	ls -l $@

