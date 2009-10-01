MAKE=make

include Makefile_base
include Makefile_$(ARCH)

BASEDIR=$(PWD)

MYDYN = jim
COMPONENTS = utils $(MYDYN) scrip
FTNALLSRC = utils/*.ftn90 $(MYDYN)/*.ftn90 utils/*.hf $(MYDYN)/*.hf
FTNALLOBJ = utils/*.o $(MYDYN)/*.o

COMM     =
OTHERS   =  $(RPNCOMM) lapack blas massvp4 bindcpu_002 $(LLAPI) $(IBM_LD)
#LIBS     = $(MODEL) $(V4D) $(PHY) $(PATCH) $(CHM) $(CPL) $(OTHERS)
LIBS     = $(OTHERS)

INSTALLDIR= $(HOME)/ovbin/python/lib.linux-i686-2.4-dev

DOCTESTPYMODULES = rpnstd.py

PYVERSIONFILE = rpn_version.py
CVERSIONFILE = rpn_version.h
VERSION   = 1.2-dev
LASTUPDATE= 2009-09

versionfile:
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
	python setup.py build

install: all
	cp build/lib.linux-i686-2.4/* $(INSTALLDIR)

clean:
	rm -f testfile.fst;\
	rm -rf build; \
	for i in $(COMPONENTS); \
	do \
	cd $$i ; $(MAKE) clean0 ; make clean; cd .. ;\
	done
	/bin/rm -f mec_$(MYDYN)_$(EC_ARCH).Abs

tags: clean
	rm -f tags TAGS
	list='$(FTNALLSRC)'; \
	for myfile in $$list; do \
		etags --language=fortran --defines --append $$myfile ; \
		ctags --language=fortran --defines --append $$myfile ; \
	done

alltests: all
	for i in $(COMPONENTS); \
	do echo -e "\n==== Make Test: " $$i "====\n"; cd $$i ; $(MAKE) test ; cd .. ;\
	done
	for i in $(DOCTESTPYMODULES); \
	do echo -e "\n==== PY-DocTest: " $$i "====\n"; python $$i ;\
	done
	cd test ; $(MAKE) test ; cd ..
