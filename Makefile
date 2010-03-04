MAKE=make

include Makefile_base
include Makefile_$(ARCH)

BASEDIR=$(PWD)

COMPONENTS = utils test

COMM     =
OTHERS   =  $(RPNCOMM) lapack blas massvp4 bindcpu_002 $(LLAPI) $(IBM_LD)
#LIBS     = $(MODEL) $(V4D) $(PHY) $(PATCH) $(CHM) $(CPL) $(OTHERS)
LIBS     = $(OTHERS)

DOCTESTPYMODULES = rpn_helpers.py rpnstd.py 

PYVERSIONFILE = rpn_version.py
CVERSIONFILE = rpn_version.h
VERSION   = 1.2.0
LASTUPDATE= 2010-03

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

clean:
	rm -f testfile.fst;\
	rm -rf build; \
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

alltests: all
	for i in $(COMPONENTS); \
	do echo -e "\n==== Make Test: " $$i "====\n"; cd $$i ; $(MAKE) test ; cd .. ;\
	done
	for i in $(DOCTESTPYMODULES); \
	do echo -e "\n==== PY-DocTest: " $$i "====\n"; python $$i ;\
	done
	cd test ; $(MAKE) test ; cd ..
