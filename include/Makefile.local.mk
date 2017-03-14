ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: $$rpnpy/include/Makefile.local.mk)
$(info ## )
endif
## General/Common definitions for models (excluding Etagere added ones)

ifeq (,$(wildcard $(rpnpy)/VERSION))
   $(error Not found: $(rpnpy)/VERSION)
endif
RPNPY_VERSION0  = $(shell cat $(rpnpy)/VERSION)
RPNPY_VERSION   = $(notdir $(RPNPY_VERSION0))
RPNPY_VERSION_X = $(dir $(RPNPY_VERSION0))

## Some Shortcut/Alias to Lib Names

# ifeq (,$(RPNPY_RMN_VERSION))
#    $(error RPNPY_RMN_VERSION not defined; export RPNPY_RMN_VERSION=_015.2)
# endif

# RMN_VERSION    = rmn$(RPNPY_RMN_VERSION)# RPNPY_LIBS_MERGED = rpnpy_main rpnpy_driver rpnpy_utils rpnpy_tdpack rpnpy_base
# ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
# RPNPY_LIBS_OTHER  =  rpnpy_stubs rpnpy_massvp7_wrap
# else
# RPNPY_LIBS_OTHER  =  rpnpy_stubs
# endif
# RPNPY_LIBS_ALL    = $(RPNPY_LIBS_MERGED) $(RPNPY_LIBS_OTHER)
# RPNPY_LIBS        = rpnpy $(RPNPY_LIBS_OTHER) 
# RPNPY_LIBS_V      = rpnpy_$(RPNPY_VERSION) $(RPNPY_LIBS_OTHER) 

# RPNPY_LIBS_ALL_FILES = $(foreach item,$(RPNPY_LIBS_ALL),$(LIBDIR)/lib$(item).a)
# RPNPY_LIBS_ALL_FILES_PLUS = $(LIBDIR)/librpnpy.a $(RPNPY_LIBS_ALL_FILES) 

# OBJECTS_MERGED_rpnpy = $(foreach item,$(RPNPY_LIBS_MERGED),$(OBJECTS_$(item)))

# RPNPY_MOD_FILES  = $(foreach item,$(FORTRAN_MODULES_rpnpy),$(item).[Mm][Oo][Dd])

# RPNPY_ABS        = yyencode yydecode yy2global time2sec_main
# RPNPY_ABS_FILES  = $(foreach item,$(RPNPY_ABS),$(BINDIR)/$(item).Abs )

## Base Libpath and libs with placeholders for abs specific libs
##MODEL1_LIBAPPL = $(RPNPY_LIBS_V)

#STATIC_COMPILER = -static -static-intel
STATIC_COMPILER = -static
#STATIC_COMPILER = -static-intel

##
.PHONY: rpnpy_vfiles rpnpy_version.inc rpnpy_version.h rpnpy_version.py
RPNPY_VFILES = rpnpy_version.inc rpnpy_version.h rpnpy_version.py
rpnpy_vfiles: $(RPNPY_VFILES)
rpnpy_version.inc:
	.rdemkversionfile "rpnpy" "$(RPNPY_VERSION)" $(ROOT)/include f
rpnpy_version.h:
	.rdemkversionfile "rpnpy" "$(RPNPY_VERSION)" $(ROOT)/include c
LASTUPDATE = $(shell date '+%Y-%m-%d %H:%M %Z')
rpnpy_version.py:
	echo "__VERSION__ = '$(RPNPY_VERSION)'" > $(ROOT)/lib/rpnpy/version.py
	echo "__LASTUPDATE__ = '$(LASTUPDATE)'" >> $(ROOT)/lib/rpnpy/version.py


#---- ARCH Specific overrides -----------------------------------------
# ifeq (aix-7.1-ppc7-64,$(ORDENV_PLAT))
# LIBMASSWRAP = rpnpy_massvp7_wrap
# endif

RDE_MKL=

#---- Abs targets -----------------------------------------------------
.PHONY: sharedlibs_cp sharedlibs extlibdotfile
sharedlibs_cp: $(LIBDIR)/librmnshared_rpnpy_cp.so $(LIBDIR)/libdescripshared_rpnpy.so
sharedlibs: $(LIBDIR)/librmnshared_rpnpy.so $(LIBDIR)/libdescripshared_rpnpy.so

forced_extlibdotfile: 
	rm -f $(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot
	$(MAKE) extlibdotfile

extlibdotfile: $(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot

$(rpnpy)/.setenv.__extlib__.${ORDENV_PLAT}.dot:
	librmnpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name librmnshared_*.so | head -1`;\
	if [[ x$${librmnpath##*/} == x ]] ; then \
		librmnpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name librmnshared*.so | head -1`;\
	fi ;\
	librmnname=`echo $${librmnpath##*/} | cut -c13-`;\
	libvgdpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name libdescripshared_*.so | head -1`;\
	if [[ x$${libvgdpath##*/} == x ]] ; then \
		libvgdpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name libdescripshared*.so | head -1`;\
	fi ;\
	libvgdname=`echo $${libvgdpath##*/} | cut -c17-`;\
	libburpcpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name libburp_c_shared_*.so | head -1`;\
	if [[ x$${libburpcpath##*/} == x ]] ; then \
		libburpcpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name libburp_c_shared*.so | head -1`;\
	fi ;\
	libburpcname=*;\
	echo "export RPNPY_RMN_LIBPATH=\$${RPNPY_RMN_LIBPATH:-$${librmnpath%/*}}" >> $@ ;\
	echo "export RPNPY_RMN_VERSION=\$${RPNPY_RMN_VERSION:-$${librmnname%.so}}" >> $@ ;\
	echo "export RPNPY_VGD_LIBPATH=\$${RPNPY_VGD_LIBPATH:-$${libvgdpath%/*}}" >> $@ ;\
	echo "export RPNPY_VGD_VERSION=\$${RPNPY_VGD_VERSION:-$${libvgdname%.so}}" >> $@ ;\
	echo "export RPNPY_BURPC_LIBPATH=\$${RPNPY_BURPC_LIBPATH:-$${libburpcpath%/*}}" >> $@ ;\
	echo "export RPNPY_BURPC_VERSION=\$${RPNPY_BURPC_VERSION:-$${libburpcname%.so}}" >> $@ ;\
	echo "export LD_LIBRARY_PATH=\$${RPNPY_RMN_LIBPATH}:\$${LD_LIBRARY_PATH}" >> $@ ;\
	echo "export LIBPATH=\$${RPNPY_RMN_LIBPATH}:\$${LIBPATH}" >> $@
	cat $@

$(LIBDIR)/librmnshared_rpnpy_cp.so:
	libfullpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name librmnshared*.so | head -1`;\
	cp $$libfullpath $@ ;\
	cd $(LIBDIR) ;\
	rm -rf librmnshared_rpnpy.so;\
	ln -s $(notdir $@) librmnshared_rpnpy.so

$(LIBDIR)/librmnshared_rpnpy.so:
	libfullpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name librmn.a | head -1`;\
	mytmpdir=librmnshared_rpnpy.so_dir_$$  ;\
	mkdir $$mytmpdir 2> /dev/null ;\
	cd $$mytmpdir ;\
	rm -f *.o ;\
	ar x $$libfullpath ;\
	rm -rf vpow_ibm.o whiteboard_omp.o whiteboard_st.o *ccard*.o *ccard* \
		fmain2cmain.o resident_time.o non_preempt_clock.o ;\
	$(BUILDFC_NOMPI) -shared -openmp $(STATIC_COMPILER) -o $@ *.o ;\
	cd .. ; rm -rf $$mytmpdir
	#rde.f90_ld $(VERBOSEVL) -shared -L$(LIBDIR)  -o $@ *.o
	#rde.f90_ld $(VERBOSEVL) -shared -openmp -static -static-intel -l -o $@ *.o
	#$(BUILDFC_NOMPI) -shared -openmp -static -static-intel -o $@ *.o

$(LIBDIR)/libdescripshared_rpnpy.so: $(LIBDIR)/librmnshared_rpnpy.so
	libfullpath=`rdefind $(EC_LD_LIBRARY_PATH)  --maxdepth 0 --name libdescrip.a | head -1`;\
	mytmpdir=librmnshared_rpnpy.so_dir_$$  ;\
	mkdir $$mytmpdir 2> /dev/null ;\
	cd $$mytmpdir ;\
	rm -f *.o ;\
	ar x $$libfullpath ;\
	export EC_INCLUDE_PATH="" ;\
	$(RDEF90_LD) -shared $(RDEALL_LIBPATH) $(STATIC_COMPILER) -L$(LIBDIR) -lrmnshared_rpnpy $(foreach item,$(RDEALL_LIBPATH_NAMES),-Wl,-rpath,$(item)) -o $@ *.o ;\
	cd .. ; rm -rf $$mytmpdir
	#rde.f90_ld $(VERBOSEVL) -shared -L$(LIBDIR) -lrmnshared_rpnpy -o $@ *.o
	#$(BUILDFC_NOMPI) -shared -openmp -static -static-intel -lrmnshared_rpnpy -o $@ *.o
	#$(BUILDFC_NOMPI) -shared -openmp $(STATIC_COMPILER) -L$(LIBDIR) -lrmnshared_rpnpy -o $@ *.o ;\

#TODO: $(LIBDIR)/libburp_c_shared_rpnpy.so:


#---- Lib target - automated ------------------------------------------


ifneq (,$(DEBUGMAKE))
$(info ## ==== $$rpnpy/include/Makefile.local.mk [END] ==================)
endif
