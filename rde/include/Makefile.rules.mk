ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: Makefile.rules.mk)
$(info ## )
endif

ifeq (,$(CONST_BUILD))
   ifneq (,$(DEBUGMAKE))
      $(info include $(ROOT)/include $(MAKEFILE_CONST))
   endif
   include $(ROOT)/$(MAKEFILE_CONST)
endif

INCSUFFIXES = $(CONST_RDESUFFIXINC)
SRCSUFFIXES = $(CONST_RDESUFFIXSRC)

.SUFFIXES :
.SUFFIXES : $(INCSUFFIXES) $(SRCSUFFIXES) .o 

## ==== compilation / load macros

RDEFTN2F = rde.ftn2f $(VERBOSEVL) $(OMP) $(RDE_COMP_RULES_FILE_USER)
RDEFTN902F90 = rde.ftn2f -f90 $(VERBOSEVL) $(OMP) $(RDE_COMP_RULES_FILE_USER)
RDEF77 = rde.f77 $(VERBOSEVL) $(RDE_COMP_RULES_FILE_USER)
RDEF90 = rde.f90 $(VERBOSEVL) $(RDE_COMP_RULES_FILE_USER)
RDEF90_LD = rde.f90_ld $(VERBOSEVL) $(RDE_COMP_RULES_FILE_USER)
RDEFTN77 = rde.ftn77 $(VERBOSEVL) $(RDE_COMP_RULES_FILE_USER)
RDEFTN90 = rde.ftn90 $(VERBOSEVL) $(RDE_COMP_RULES_FILE_USER)
RDECC = rde.cc $(VERBOSEVL) $(RDE_COMP_RULES_FILE_USER)

FTNC77 = export EC_LD_LIBRARY_PATH="" ; $(RDEFTN2F) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) -src1
FTNC90 = export EC_LD_LIBRARY_PATH="" ; $(RDEFTN902F90) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) -src1
FC77 = export EC_LD_LIBRARY_PATH="" ; $(RDEF77) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) $(RDEALL_FFLAGS) -c -src1
FC90a = $(RDEF90) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) $(RDEALL_FFLAGS) -c -src1
FC90aNoInclude = $(RDEF90) $(RDEALL_DEFINES) $(RDEALL_FFLAGS) -c -src1
FC90 = export EC_LD_LIBRARY_PATH="" ; $(FC90a)
FC95 = $(FC90)
FC03 = $(FC90)

FTNC77FC77 = $(RDEFTN77) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) $(RDEALL_FFLAGS) -c -src1
FTNC90FC90 = $(RDEFTN90) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) $(RDEALL_FFLAGS) -c -src1

CC = $(RDECC) $(RDEALL_DEFINES) $(RDEALL_INCLUDES) $(RDEALL_CFLAGS) -c -src1

# PTNC = sed 's/^[[:blank:]].*PROGRAM /      SUBROUTINE /' | sed 's/^[[:blank:]].*program /      subroutine /'  > $*.f


BUILDFC = export EC_INCLUDE_PATH="" ; $(RDEF90_LD) $(RDEALL_LFLAGS) $(RDEALL_LIBPATH)
BUILDCC = export EC_INCLUDE_PATH="" ; $(RDECC)  $(RDEALL_LFLAGS) $(RDEALL_LIBPATH)
BUILDFC_NOMPI = export EC_INCLUDE_PATH="" ; $(RDEF90_LD) $(RDEALL_LFLAGS_NOMPI) $(RDEALL_LIBPATH)
BUILDCC_NOMPI = export EC_INCLUDE_PATH="" ; $(RDECC)  $(RDEALL_LFLAGS_NOMPI) $(RDEALL_LIBPATH)

DORBUILD4BIDONF = \
	mkdir .bidon 2>/dev/null ; cd .bidon ;\
	.rdemakemodelbidon $${MAINSUBNAME} > bidon_$${MAINSUBNAME}.f90 ;\
	$(FC90aNoInclude) bidon_$${MAINSUBNAME}.f90 >/dev/null || status=1 ;\
	rm -f bidon_$${MAINSUBNAME}.f90 ;\
	cd ..
DORBUILD4BIDONC = \
	mkdir .bidon 2>/dev/null ; cd .bidon ;\
	.rdemakemodelbidon -c $${MAINSUBNAME} > bidon_$${MAINSUBNAME}.c ;\
	$(CC) bidon_$${MAINSUBNAME}.c >/dev/null || status=1 ;\
	rm -f bidon_$${MAINSUBNAME}.c ;\
	cd ..
DORBUILD4EXTRAOBJ = \
	RBUILD_EXTRA_OBJ1="`ls $${RBUILD_EXTRA_OBJ:-_RDEBUILDNOOBJ_} 2>/dev/null`"
DORBUILD4COMMSTUBS = \
	lRBUILD_COMM_STUBS="" ;\
	if [[ x"$${RBUILD_COMM_STUBS}" != x"" ]] ; then \
	   lRBUILD_COMM_STUBS="-l$${RBUILD_COMM_STUBS}";\
	fi
DORBUILDEXTRALIBS = \
	lRBUILD_EXTRA_LIB="" ;\
	if [[ x"$${RBUILD_EXTRA_LIB}" != x"" ]] ; then \
		for mylib in $${RBUILD_EXTRA_LIB} ; do \
	   	lRBUILD_EXTRA_LIB="$${lRBUILD_EXTRA_LIB} -l$${mylib}";\
		done ;\
	fi
DORBUILDLIBSAPPL = \
	lRBUILD_LIBAPPL="" ;\
	if [[ x"$${RBUILD_LIBAPPL}" != x"" ]] ; then \
		for mylib in $${RBUILD_LIBAPPL} ; do \
	   	lRBUILD_LIBAPPL="$${lRBUILD_LIBAPPL} -l$${mylib}";\
		done ;\
	fi
DORBUILDRPATH = \
	LRBUILD_RPATH="" ;\
	for mypath in $(RDEALL_LIBPATH) ; do \
		if [[ x$${LRBUILD_RPATH} == x ]] ; then \
			LRBUILD_RPATH=" -Wl,-rpath,$${mypath}";\
		else \
	   	LRBUILD_RPATH="$${LRBUILD_RPATH},$${mypath}";\
		fi ;\
	done ;\
DORBUILD4FINALIZE = \
	rm -f .bidon/bidon_$${MAINSUBNAME}.o 2>/dev/null || true ;\
	if [[ x$${status} == x1 ]] ; then exit 1 ; fi

RBUILD_EXTRA_OBJ0 = *.o

RBUILD4objMPI = RBUILD_EXTRA_OBJ=$(RBUILD_EXTRA_OBJ0) ; $(RBUILD4MPI)
RBUILD4MPI = \
	status=0 ;\
	$(DORBUILD4BIDONF) ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(BUILDFC) -mpi -o $${RBUILD_ABS_NAME:-$@} $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) \
	   .bidon/bidon_$${MAINSUBNAME}.o $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	$(DORBUILD4FINALIZE)

RBUILD4objNOMPI = RBUILD_EXTRA_OBJ=$(RBUILD_EXTRA_OBJ0) ; $(RBUILD4NOMPI)
RBUILD4NOMPI = \
	status=0 ;\
	$(DORBUILD4BIDONF) ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILD4COMMSTUBS) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(BUILDFC_NOMPI) -o $${RBUILD_ABS_NAME:-$@} $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) $${lRBUILD_COMM_STUBS}\
	   .bidon/bidon_$${MAINSUBNAME}.o $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	$(DORBUILD4FINALIZE)

RBUILD4objMPI_C = RBUILD_EXTRA_OBJ=$(RBUILD_EXTRA_OBJ0) ; $(RBUILD4MPI_C)
RBUILD4MPI_C = \
	status=0 ;\
	$(DORBUILD4BIDONC) ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(BUILDCC)  -mpi -o $${RBUILD_ABS_NAME:-$@} $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) \
	   .bidon/bidon_$${MAINSUBNAME}.o $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	$(DORBUILD4FINALIZE)

RBUILD4objNOMPI_C = RBUILD_EXTRA_OBJ=$(RBUILD_EXTRA_OBJ0) ; $(RBUILD4NOMPI_C)
RBUILD4NOMPI_C = \
	status=0 ;\
	$(DORBUILD4BIDONC) ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILD4COMMSTUBS) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(BUILDCC_NOMPI) -o $${RBUILD_ABS_NAME:-$@} $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) $${lRBUILD_COMM_STUBS}\
	   .bidon/bidon_$${MAINSUBNAME}.o $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	$(DORBUILD4FINALIZE)


ifneq (,$(filter aix-%,$(RDE_BASE_ARCH))$(filter AIX-%,$(RDE_BASE_ARCH)))

#	$${LRBUILD_RPATH} : -Wl,-rpath not understood by xlf13 !?!

RBUILD4MPI_SO = \
	status=0 ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(BUILDFC) -mpi -o $${RBUILD_LIBSO_NAME:-$@}.o $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) \
	    -G -qmkshrobj $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	ar rcv $${RBUILD_LIBSO_NAME:-$@} $${RBUILD_LIBSO_NAME:-$@}.o ;\
	$(DORBUILD4FINALIZE)

RBUILD4NOMPI_SO = \
	status=0 ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILD4COMMSTUBS) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(BUILDFC_NOMPI) -shared -mpi -o $${RBUILD_LIBSO_NAME:-$@}.o $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) $${lRBUILD_COMM_STUBS}\
	   -G -qmkshrobj $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	ar rcv $${RBUILD_LIBSO_NAME:-$@} $${RBUILD_LIBSO_NAME:-$@}.o ;\
	$(DORBUILD4FINALIZE)

else

RBUILD4MPI_SO = \
	status=0 ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILD4COMMSTUBS) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(DORBUILDRPATH) ;\
	$(BUILDFC) -shared -mpi -o $${RBUILD_LIBSO_NAME:-$@} $${LRBUILD_RPATH} $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) $${lRBUILD_COMM_STUBS} \
	   $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	$(DORBUILD4FINALIZE)

RBUILD4NOMPI_SO = \
	status=0 ;\
	$(DORBUILD4EXTRAOBJ) ;\
	$(DORBUILDLIBSAPPL) ; $(DORBUILDEXTRALIBS) ;\
	$(DORBUILDRPATH) ;\
	$(BUILDFC_NOMPI) -shared -o $${RBUILD_LIBSO_NAME:-$@} $${LRBUILD_RPATH} $${lRBUILD_EXTRA_LIB} $${lRBUILD_LIBAPPL} $(RDEALL_LIBS) \
	   $${RBUILD_EXTRA_OBJ1} $(CODEBETA) || status=1 ;\
	$(DORBUILD4FINALIZE)

endif


## ==== Implicit Rules

# .ptn.o:
# 	#.ptn.o:
# 	rm -f $*.f
# 	$(FC77) $< && touch $*.f

# .ptn.f:
# 	#.ptn.f:
# 	rm -f $*.f
# 	$(FTNC77) $<

#TODO: .FOR.o
#TODO: .tmpl90.o

.ftn.o:
	#.ftn.o:
	rm -f $*.f
	$(FTNC77FC77) $< && touch $*.f
.ftn.f:
	#.ftn.f:
	rm -f $*.f
	$(FTNC77) $<
.f90.o:
	#.f90.o:
	$(FC90) $<
.F90.o:
	#.F90.o:
	$(FC90) $<
.F95.o:
	#.F95.o:
	$(FC95) $<
.F03.o:
	#.F03.o:
	$(FC03) $<
.f.o:
	#.f.o:
	$(FC77) $<
.ftn90.o:
	#.ftn90.o:
	rm -f $*.f90
	$(FTNC90FC90) $< && touch $*.f90
.cdk90.o:
	#.cdk90.o:
	$(FTNC90FC90) $< && touch $*.f90
.cdk90.f90:
	#.cdk90.f90:
	rm -f $*.f90
	$(FTNC90) $<
.ftn90.f90:
	#.ftn90.f90:
	rm -f $*.f90
	$(FTNC90) $<

.c.o:
	#.c.o:
	$(CC) $<

# .s.o:
# 	#.s.o:
# 	$(AS) -c $(CPPFLAGS) $(ASFLAGS) $<

ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.rules.mk [END] =======================================)
endif
