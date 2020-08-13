ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnpy/include/Makefile.local.rpnpy.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

ifeq (,$(wildcard $(rpnpy)/VERSION))
   $(error Not found: $(rpnpy)/VERSION)
endif
RPNPY_VERSION0  = $(shell cat $(rpnpy)/VERSION)
RPNPY_VERSION   = $(notdir $(RPNPY_VERSION0))
RPNPY_VERSION_X = $(dir $(RPNPY_VERSION0))

RPNPY_SFX = $(RDE_LIBS_SFX)

RPNPY_LIB_MERGED_NAME_0 = 
RPNPY_LIBS_MERGED_0 = 
RPNPY_LIBS_OTHER_0  = 
RPNPY_LIBS_EXTRA_0  = 
RPNPY_LIBS_ALL_0    = 
RPNPY_LIBS_FL_0     = 
RPNPY_LIBS_0        = 

RPNPY_LIB_MERGED_NAME = $(foreach item,$(RPNPY_LIB_MERGED_NAME_0),$(item)$(RPNPY_SFX))
RPNPY_LIBS_MERGED = $(foreach item,$(RPNPY_LIBS_MERGED_0),$(item)$(RPNPY_SFX))
RPNPY_LIBS_OTHER  = $(foreach item,$(RPNPY_LIBS_OTHER_0),$(item)$(RPNPY_SFX))
RPNPY_LIBS_EXTRA  = $(foreach item,$(RPNPY_LIBS_EXTRA_0),$(item)$(RPNPY_SFX))
RPNPY_LIBS_ALL    = $(foreach item,$(RPNPY_LIBS_ALL_0),$(item)$(RPNPY_SFX))
RPNPY_LIBS_FL     = $(foreach item,$(RPNPY_LIBS_FL_0),$(item)$(RPNPY_SFX))
RPNPY_LIBS        = $(foreach item,$(RPNPY_LIBS_0),$(item)$(RPNPY_SFX))

RPNPY_LIB_MERGED_NAME_V = $(foreach item,$(RPNPY_LIB_MERGED_NAME),$(item)_$(RPNPY_VERSION))
RPNPY_LIBS_MERGED_V = $(foreach item,$(RPNPY_LIBS_MERGED),$(item)_$(RPNPY_VERSION))
RPNPY_LIBS_OTHER_V  = $(foreach item,$(RPNPY_LIBS_OTHER),$(item)_$(RPNPY_VERSION))
RPNPY_LIBS_EXTRA_V  = $(foreach item,$(RPNPY_LIBS_EXTRA),$(item)_$(RPNPY_VERSION))
RPNPY_LIBS_ALL_V    = $(foreach item,$(RPNPY_LIBS_ALL),$(item)_$(RPNPY_VERSION))
RPNPY_LIBS_FL_V     = $(foreach item,$(RPNPY_LIBS_FL),$(item)_$(RPNPY_VERSION))
RPNPY_LIBS_V        = $(foreach item,$(RPNPY_LIBS),$(item)_$(RPNPY_VERSION))

RPNPY_LIBS_FL_FILES       = $(foreach item,$(RPNPY_LIBS_FL),$(LIBDIR)/lib$(item).a.fl)
RPNPY_LIBS_ALL_FILES      = $(foreach item,$(RPNPY_LIBS_ALL) $(RPNPY_LIBS_EXTRA),$(LIBDIR)/lib$(item).a)
RPNPY_LIBS_ALL_FILES_PLUS =  ## $(LIBDIR)/lib$(RPNPY_LIB_MERGED_NAME).a $(RPNPY_LIBS_ALL_FILES) $(RPNPY_LIBS_FL_FILES)

OBJECTS_MERGED_rpnpy = $(foreach item,$(RPNPY_LIBS_MERGED_0),$(OBJECTS_$(item)))

RPNPY_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_rpnpy),$(item).[Mm][Oo][Dd])

RPNPY_ABS        = extlibdotfile
RPNPY_ABS_FILES  = $(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot


RPNPY_DOC_TESTS_FILES = $(wildcard $(rpnpy)/lib/*.py) $(wildcard $(rpnpy)/lib/rpnpy/*.py) $(wildcard $(rpnpy)/lib/rpnpy/[a-z]*/*.py)
RPNPY_UNIT_TESTS_FILES = $(wildcard $(rpnpy)/share/tests/test_*.py)
RPNPY_SH_TESTS_FILES  = $(wildcard $(rpnpy)/share/tests/test_*.sh)
RPNPY_TESTS = rpnpy_doctests rpnpy_unittests rpnpy_shtests

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(RPNPY_LIBS_V)

## System-wide definitions

RDE_LIBS_USER_EXTRA := $(RDE_LIBS_USER_EXTRA) $(RPNPY_LIBS_ALL_FILES_PLUS)
RDE_BINS_USER_EXTRA := $(RDE_BINS_USER_EXTRA) $(RPNPY_ABS_FILES)

ifeq (1,$(RDE_LOCAL_LIBS_ONLY))
   RPNPY_ABS_DEP = $(RPNPY_LIBS_ALL_FILES_PLUS) $(RDE_ABS_DEP)
endif

##
.PHONY: rpnpy_vfiles rpnpy_version.inc rpnpy_version.h rpnpy_version.py
#RPNPY_VFILES = rpnpy_version.inc rpnpy_version.h rpnpy_version.py
RPNPY_VFILES = $(rpnpy)/lib/rpnpy/version.py
rpnpy_vfiles: $(RPNPY_VFILES)
rpnpy_version.inc:
	.rdemkversionfile "rpnpy" "$(RPNPY_VERSION)" $(rpnpy)/include f
rpnpy_version.h:
	.rdemkversionfile "rpnpy" "$(RPNPY_VERSION)" $(rpnpy)/include c
LASTUPDATE = $(shell date '+%Y-%m-%d %H:%M %Z')
rpnpy_version.py: $(rpnpy)/lib/rpnpy/version.py
$(rpnpy)/lib/rpnpy/version.py:
	echo "__VERSION__ = '$(RPNPY_VERSION)'" > $(rpnpy)/lib/rpnpy/version.py
	echo "__LASTUPDATE__ = '$(LASTUPDATE)'" >> $(rpnpy)/lib/rpnpy/version.py

BUILDNAME = 

#---- ARCH Specific overrides -----------------------------------------

#---- Abs targets -----------------------------------------------------

## Rpnpy Targets
.PHONY: $(RPNPY_ABS)

forced_extlibdotfile: rpnpy_vfiles
	rm -f $(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot
	$(MAKE) extlibdotfile

extlibdotfile: $(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot

$(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot:
	$(rpnpy)/bin/.rpy.mk.setenv.__extlib__ $@
	cat $@

.PHONY: allbin_rpnpy allbincheck_rpnpy
allbin_rpnpy: | $(RPNPY_ABS)
allbincheck_rpnpy:
	for item in $(RPNPY_ABS_FILES) ; do \
		if [[ ! -x $${item} ]] ; then exit 1 ; fi ;\
	done ;\
	exit 0

#---- Tests targets ---------------------------------------------------
.PHONY: rpnpy_tests rpnpy_doctests rpnpy_unittests rpnpy_shtests

ifeq (,$(PYTHON))
   PYTHON = python
endif
PYTHONVERSION=py$(shell echo `$(PYTHON) -V 2>&1 | sed 's/python//i'`)
RPNPY_TESTLOGDIR := $(ROOT)/_testlog_/rpnpy/$(PYTHONVERSION)_v$(RPNPY_VERSION)
# export RPNPY_NOLONGTEST=1

rpnpy_tests: $(RPNPY_TESTS)

rpnpy_doctests:
	if [[ "x$(RPNPY_TESTLOGDIR)" != "x" ]] ; then \
		mkdir -p $(RPNPY_TESTLOGDIR) > /dev/null 2>&1 || true ; \
	fi ; \
	cd $(TMPDIR) ; \
	mkdir tmp 2>/dev/null || true ; \
	TestLogDir=$(RPNPY_TESTLOGDIR) ; \
	echo -e "\n======= PY-DocTest List ========\n" ; \
	for i in $(RPNPY_DOC_TESTS_FILES); do \
		logname=`echo $${i} | sed "s|$(rpnpy)||" | sed "s|/|_|g"` ; \
		if [[ x$${i##*/} != xall.py && x$${i##*/} != xproto.py  && x$${i##*/} != xproto_burp.py ]] ; then \
			echo -e "\n==== PY-DocTest: " $$i "==== " $(PYTHONVERSION) " ====\n"; \
			$(PYTHON) $$i > $${TestLogDir:-.}/$${logname}.log 2> $${TestLogDir:-.}/$${logname}.err ;\
			grep failures $${TestLogDir:-.}/$${logname}.log ;\
		fi ; \
	done

rpnpy_unittests:
	if [[ "x$(RPNPY_TESTLOGDIR)" != "x" ]] ; then \
		mkdir -p $(RPNPY_TESTLOGDIR) > /dev/null 2>&1 || true ; \
	fi ; \
	cd $(TMPDIR) ; \
	mkdir tmp 2>/dev/null || true ; \
	TestLogDir=$(RPNPY_TESTLOGDIR) ; \
	echo -e "\n======= PY-UnitTest List ========\n" ; \
	for i in $(RPNPY_UNIT_TESTS_FILES); do \
		logname=`echo $${i} | sed "s|$(ROOT)||" | sed "s|/|_|g"` ; \
		echo -e "\n==== PY-UnitTest: " $$i "==== " $(PYTHONVERSION) " ====\n"; \
		$(PYTHON) $$i > $${TestLogDir:-.}/$${logname}.log 2> $${TestLogDir:-.}/$${logname}.err ;\
		cat  $${TestLogDir:-.}/$${logname}.err ;\
	done

#TODO: find a way to force $(PYTHON) to be used by script
rpnpy_shtests:
	if [[ "x$(RPNPY_TESTLOGDIR)" != "x" ]] ; then \
		mkdir -p $(RPNPY_TESTLOGDIR) > /dev/null 2>&1 || true ; \
	fi ; \
	cd $(TMPDIR) ; \
	mkdir tmp 2>/dev/null || true ; \
	TestLogDir=$(RPNPY_TESTLOGDIR) ; \
	echo -e "\n======= SH-UnitTest List ========\n" ; \
	for i in $(RPNPY_SH_TESTS_FILES); do \
		logname=`echo $${i} | sed "s|$(ROOT)||" | sed "s|/|_|g"` ; \
		echo -e "\n==== SH-UnitTest: " $$i "==== " $(PYTHONVERSION) " ====\n"; \
		$$i > $${TestLogDir:-.}/$${logname}.log 2> $${TestLogDir:-.}/$${logname}.err ;\
		cat  $${TestLogDir:-.}/$${logname}.err ;\
	done


#---- Lib target - automated ------------------------------------------
rpnpy_LIB_template1 = \
$$(LIBDIR)/lib$(2)_$$($(3)_VERSION).a: $$(OBJECTS_$(1)) ; \
rm -f $$@ $$@_$$$$$$$$; \
ar r $$@_$$$$$$$$ $$(OBJECTS_$(1)); \
mv $$@_$$$$$$$$ $$@

LIB_template1fl = \
$$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a.fl: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a $$@

LIB_template2fl = \
$$(LIBDIR)/lib$(1).a.fl: $$(LIBDIR)/lib$(1)_$$($(2)_VERSION).a.fl ; \
cd $$(LIBDIR) ; \
rm -f $$@ ; \
ln -s lib$(1)_$$($(2)_VERSION).a.fl $$@

.PHONY: rpnpy_libs
rpnpy_libs: $(OBJECTS_rpnpy) $(RPNPY_LIBS_ALL_FILES_PLUS) | $(RPNPY_VFILES)
$(foreach item,$(RPNPY_LIBS_ALL_0) $(RPNPY_LIBS_EXTRA_0),$(eval $(call rpnpy_LIB_template1,$(item),$(item)$(RPNPY_SFX),RPNPY)))
$(foreach item,$(RPNPY_LIBS_ALL) $(RPNPY_LIBS_EXTRA),$(eval $(call LIB_template2,$(item),RPNPY)))
$(foreach item,$(RPNPY_LIBS_FL),$(eval $(call LIB_template1fl,$(item),RPNPY)))
$(foreach item,$(RPNPY_LIBS_FL),$(eval $(call LIB_template2fl,$(item),RPNPY)))

rpnpy_libs_fl: $(RPNPY_LIBS_FL_FILES)

$(LIBDIR)/lib$(RPNPY_LIB_MERGED_NAME)_$(RPNPY_VERSION).a: $(OBJECTS_MERGED_rpnpy) | $(RPNPY_VFILES)
# 	rm -f $@ $@_$$$$; ar r $@_$$$$ $(OBJECTS_MERGED_rpnpy); mv $@_$$$$ $@
# $(LIBDIR)/lib$(RPNPY_LIB_MERGED_NAME).a: $(LIBDIR)/lib$(RPNPY_LIB_MERGED_NAME)_$(RPNPY_VERSION).a
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(RPNPY_LIB_MERGED_NAME)_$(RPNPY_VERSION).a $@

# $(LIBDIR)/lib$(RPNPY_LIBS_SHARED_0)_$(RPNPY_VERSION).so: $(OBJECTS_rpnpy) $(RPNPY_LIBS_OTHER_FILES) | $(RPNPY_VFILES)
# 	export RBUILD_EXTRA_OBJ="$(OBJECTS_MERGED_rpnpy)" ;\
# 	export RBUILD_LIBAPPL="$(RPNPY_LIBS_OTHER) $(RPNPY_LIBS_DEP)" ;\
# 	$(RBUILD4MPI_SO)
# 	ls -l $@
# $(LIBDIR)/lib$(RPNPY_LIBS_SHARED_0).so: $(LIBDIR)/lib$(RPNPY_LIBS_SHARED_0)_$(RPNPY_VERSION).so
# 	cd $(LIBDIR) ; rm -f $@ ;\
# 	ln -s lib$(RPNPY_LIBS_SHARED_0)_$(RPNPY_VERSION).so $@
# 	ls -l $@ ; ls -lL $@

ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnpy/include/Makefile.local.rpnpy.mk [END] ==================)
endif
