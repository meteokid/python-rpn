MAKE=make

VERSION    = 1.3.2
LASTUPDATE = 2015-03

include Makefile_base.mk
include $(EC_ARCH)/Makefile.inc.mk

BASEDIR = $(PWD)

COMPONENTS = utils

COMM     =
OTHERS   = $(RPNCOMM) $(LAPACK) $(BLAS) massv_p4 $(LLAPI) $(IBM_LD)
LIBS     = $(OTHERS)

INSTALLDIR = $(HOME)/ovbin/python/lib.linux-i686-$(PYVERSION)-dev
DOCTESTPYMODULES = rpn_helpers.py rpnstd.py 

PYVERSIONFILE = rpn_version.py
CVERSIONFILE  = rpn_version.h

SSM_DESTDIR = $(HOME)/SsmBundles/
SSM_RELDIR  = ENV/
#SSM_X       = x/
SSM_X       = py/$(PYVERSION)/
SSM_RELDIRDOM     = $(SSM_RELDIR)d/$(SSM_X)
SSM_RELDIRBNDL    = $(SSM_RELDIR)$(SSM_X)
SSMINSTALLDIR     = $(SSM_DESTDIR)$(SSM_RELDIR)
SSMINSTALLDIRDOM  = $(SSM_DESTDIR)$(SSM_RELDIRDOM)
SSMINSTALLDIRBNDL = $(SSM_DESTDIR)$(SSM_RELDIRBNDL)
SSMDEPENDENCIES   = $(PWD)/ssmuse_pre.bndl $(PWD)/ssmuse_dependencies.bndl
SSMDEPENDENCIESPOST = $(PWD)/ssmuse_post.bndl
SSMPOSTCHMOD        = a-w
SSMTMPLNAMEARCH     = PKG_V998V_ARCH
SSMTMPLDIR          = $(PWD)/share/ssm_tmpl
SSMDEPOTDIR         = $(HOME)/SsmDepot
SSMARCH             = linux26-x86-64
PKGNAME = rpnpy
SSMDOMAINNAME = $(PKGNAME)_$(VERSION)
SSMPKGNAME = $(PKGNAME)$(PYVERSION)_$(VERSION)_$(SSMARCH)

.PHONY: all install ssmpkg install_ssm install_ssm_domain install_ssm_pkgs install_ssm_bndl clean ctags doctests alltests

versionfile: $(PYVERSIONFILE)
$(PYVERSIONFILE): $(PWD)/Makefile
	echo "__VERSION__ = '$(VERSION)'" > $(PYVERSIONFILE)
	echo "__LASTUPDATE__ = '$(LASTUPDATE)'" >> $(PYVERSIONFILE)
	echo "#define VERSION \"$(VERSION)\"" > $(CVERSIONFILE)
	echo "#define LASTUPDATE \"$(LASTUPDATE)\"" >> $(CVERSIONFILE)

default: all

# slib: all
# 	r.build \
# 	  -obj $(FTNALLOBJ) \
# 	  -shared \
# 	  -librmn $(RMNLIBSHARED) \
# 	  -o jim.so

all: versionfile
	for i in $(COMPONENTS); \
	do cd $$i ; $(MAKE) all ; cd .. ;\
	done ;\
	RMNLIBSHARED=$(RMNLIBSHARED) LDFLAGS="$(LDFLAGS)" python setup.py build --compiler=$(CCNAME)
	#python setup.py build --compiler=intel
	#CC=$(CC) CFLAGS=-I${HOME}/include python setup.py build
	# other flags: LDFLAGS, INCLUDES, LIBS

clean:
	rm -f testfile.fst;\
	rm -rf build; \
	rm -rf run; \
	for i in $(COMPONENTS); \
	do \
	cd $$i ; $(MAKE) clean0 ; make clean; cd .. ;\
	done

ctags: clean
	rm -f tags TAGS
	for mydir in . $(COMPONENTS); do \
	echo $$mydir ;\
	list2="$$mydir/*.f $$mydir/*.ftn* $$mydir/*.hf"; \
	for myfile in $$list2; do \
		echo $$myfile ;\
		etags --language=fortran --defines --append $$myfile ; \
		ctags --language=fortran --defines --append $$myfile ; \
	done ; \
	list3="$$mydir/*.c $$mydir/*.h"; \
	for myfile in $$list3; do \
		echo $$myfile ;\
		etags --language=c --defines --append $$myfile ; \
		ctags --language=c --defines --append $$myfile ; \
	done ; \
	list4="$$mydir/*.py"; \
	for myfile in $$list4; do \
		echo $$myfile ;\
		etags --language=python --defines --append $$myfile ; \
		ctags --language=python --defines --append $$myfile ; \
	done ; \
	done

doctests: #all
	#export PYTHONPATH=$(PWD)/build/lib.$(PYARCH):$(PYTHONPATH)
	echo -e "\n======= PY-DocTest List ========\n" ; \
	for i in $(DOCTESTPYMODULES); \
	do echo -e "\n==== PY-DocTest: " $$i "====\n"; python $$i ;\
	done
	echo -e "\n======= PY-UnitTest List ========\n" ; \

alltests: doctests
	#export PYTHONPATH=$(PWD)/build/lib.$(PYARCH):$(PYTHONPATH)
	for i in $(COMPONENTS); \
	do echo -e "\n==== Make Test: " $$i "====\n PYTHONPATH="$(PWD)/build/lib.$(PYARCH):$(PYTHONPATH) "\n"; cd $$i ; $(MAKE) test PYTHONPATH=$(PWD)/build/lib.$(PYARCH):$(PYTHONPATH); cd .. ;\
	done; \
	echo -e "\n======= Other Tests ========\n" ; \
	cd test ; $(MAKE) test PYTHONPATH=$(PWD)/build/lib.$(PYARCH):$(PYTHONPATH); cd ..

install: all
	cp -R build/lib.* $(INSTALLDIR)

ssmpkg: $(SSMDEPOTDIR)/$(SSMPKGNAME).ssm
$(SSMDEPOTDIR)/$(SSMPKGNAME).ssm: all
	cd $(PWD)/build ;\
	tar xzf $(SSMTMPLDIR)/$(SSMTMPLNAMEARCH).ssm ;\
	rm -rf $(SSMPKGNAME) || true ;\
	mv	$(SSMTMPLNAMEARCH) $(SSMPKGNAME) ;\
	echo $(SSMDEPOTDIR) > $(SSMPKGNAME)/.SsmDepot ;\
	cp -R lib.$(PYARCH) $(SSMPKGNAME)/lib/ ;\
	cd $(SSMPKGNAME)/lib ;\
	rm -rf python || true ;\
	ln -s ./lib.$(PYARCH)/. python ;\
	cd ../.. ;\
	tar czf $@ $(SSMPKGNAME) ;\
	ls -l $@

install_ssm: install_ssm_domain install_ssm_pkgs install_ssm_bndl
install_ssm_domain: $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(SSMDOMAINNAME)
$(SSMINSTALLDIRDOM)/$(PKGNAME)/$(SSMDOMAINNAME):
	chmod u+w $(SSMINSTALLDIRDOM) || true ;\
	mkdir -p $(SSMINSTALLDIRDOM)/$(PKGNAME) || true ;\
	chmod u+w $(SSMINSTALLDIRDOM)/$(PKGNAME) || true ;\
	mkdir -p $@ ;\
	chmod u+w $@ ;\
	cd $@ ;\
	mydomain=`true_path .` ;\
	ssm created -d $${mydomain} --defaultRepositorySource $(SSMDEPOTDIR) --yes ;\
	if [[ ! -e $${mydomain}/.SsmDepot ]] ; then echo $(SSMDEPOTDIR) > $@/.SsmDepot ; fi ;\
	echo "systemRelease:3.*:ssmOsRelease:26 24:" >> $@/etc/ssm.d/platforms/Linux ;\
	chmod $(SSMPOSTCHMOD) $(SSMINSTALLDIRDOM)

install_ssm_pkgs: $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(SSMDOMAINNAME)/$(SSMPKGNAME)
#$(SSMINSTALLDIRDOM)/$(PKGNAME)/$(SSMDOMAINNAME)/$(SSMPKGNAME): install_ssm_domain $(SSMDEPOTDIR)/$(SSMPKGNAME).ssm
$(SSMINSTALLDIRDOM)/$(PKGNAME)/$(SSMDOMAINNAME)/$(SSMPKGNAME): install_ssm_domain
	set -x ; cd $(SSMINSTALLDIRDOM)/$(PKGNAME)/$(SSMDOMAINNAME) ;\
	mydomain=`true_path .` ;\
	chmod u+w -R $${mydomain} ;\
	ssm install -d $${mydomain} -p $(SSMPKGNAME) --yes --skipOnInstalled && \
	ssm publish -d $${mydomain} -p $(SSMPKGNAME) --yes --skipOnPublished && \
	chmod $(SSMPOSTCHMOD) -R $${mydomain} ;\
	ls -ld $@ ;\
	ls -ld $${mydomain}/etc/ssm.d/published/$(SSMPKGNAME)

install_ssm_bndl: $(SSMINSTALLDIRBNDL)/$(PKGNAME)/$(VERSION).bndl
$(SSMINSTALLDIRBNDL)/$(PKGNAME)/$(VERSION).bndl: install_ssm_pkgs
	chmod u+w $(SSMINSTALLDIRBNDL) || true ;\
	mkdir -p $(SSMINSTALLDIRBNDL)/$(PKGNAME) || true ;\
	chmod u+w $(SSMINSTALLDIRBNDL)/$(PKGNAME) || true ;\
	cat $(SSMDEPENDENCIES) > $@ ;\
	echo $(SSM_RELDIRDOM)$(PKGNAME)/$(SSMDOMAINNAME) >> $@ ;\
	if [[ x$(SSMDEPENDENCIESPOST) != x ]] ; then [[ -r `ls $(SSMDEPENDENCIESPOST)` ]] && cat $(SSMDEPENDENCIESPOST) >> $@ ;fi ;\
	chmod $(SSMPOSTCHMOD) $(SSMINSTALLDIRBNDL)/$(PKGNAME) $@ ;\
	ls -l $@
