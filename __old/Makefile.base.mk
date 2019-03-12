default: all

BASEDIR=$(PWD)

INCLUDES = $(BASEDIR)/utils $(BASEDIR)/jim

#OMP    = -openmp
OMP    = 
OPTIL  = 2
MPI    = 
RMNLIBSHARED = rmnshared_015.2
RMNLIB  = rmn_015.2
#RMNLIB  = rmnbeta_6_oct_2008
#LIBPATH = .
#RMNLIB  = rmn_rc010
#RPNCOMM = rpn_comm222

#DEBUG = "-debug"
DEBUG =

RCOMPIL = s.compile $(DEBUG)
RBUILD  = s.compile $(DEBUG)
CCOMPF  =
CCOMPF =
CC = $(RCOMPIL) -arch $(EC_ARCH) -abi $(ABI)  -defines "=$(DEFINE)" -includes "$(INCLUDES)" -O $(OPTIL) -optc="$(CFLAGS) -mkl -fp-model precise" $(COMPF) $(CCOMPF) -src
#CC = s.cc
FC = $(RCOMPIL) -arch $(EC_ARCH) -abi $(ABI)  -defines "=$(DEFINE)" -includes "$(INCLUDES)" -O $(OPTIL) -optf="$(FFLAGS) -mkl -fp-model precise" $(COMPF) $(FCOMPF) -src
FTNC = $(RCOMPIL) -arch $(EC_ARCH) -abi $(ABI) -defines "=$(DEFINE)" -optf="$(FFLAGS) $(CPPFLAGS)" -P $(COMPF) $(FCOMPF) -src
PTNC = sed 's/^[[:blank:]].*PROGRAM /      SUBROUTINE /' | sed 's/^[[:blank:]].*program /      subroutine /'  > $*.f

.SUFFIXES: .o .f .f90 .c .ptn .ftn .ftn90 .cdk .cdk90 .h .hf

.ptn.o:
	rm -f $*.f
	$(FC) $<
.ptn.f:
	rm -f $*.f
	$(FTNC) $<

.ftn.o:
	rm -f $*.f
	$(FC) $<
.ftn.f:
	rm -f $*.f
	$(FTNC) $<
.f90.o:
	$(FC) $<
.f.o:
	$(FC) $<
.ftn90.o:
	rm -f $*.f90
	$(FC) $<
.cdk90.o:
	$(FC) $<
.cdk90.f90:
	rm -f $*.f90
	$(FTNC) $<
.ftn90.f90:
	rm -f $*.f90
	$(FTNC) $<
.c.o:
	$(CC) $<
%.o : %.mod 

#==========================================
clean0:
#Faire le grand menage. On enleve tous les fichiers sources inutiles et les .o
	-if [ "*.[chfs]" != "`echo *.[chfs]`" ] ; \
	then 	for i in *.[chfs]; \
	do \
	if (r.ucanrm $$i)  ; \
	then rm -f $$i; \
	fi; \
	done \
	fi; \
	if [ "*.cdk" != "`echo *.cdk`" ] ; \
	then 	for i in *.cdk; \
	do \
	if (r.ucanrm  $$i) ; \
	then rm -f $$i; \
	fi; \
	done \
	fi; \
	if [ "*.txt" != "`echo *.txt`" ] ; \
	then 	for i in *.txt; \
	do \
	if (r.ucanrm  $$i) ; \
	then rm -f $$i; \
	fi; \
	done \
	fi; \
	if [ "*.*_sh" != "`echo *.*_sh`" ] ; \
	then \
	for i in *.*_sh; \
	do \
	if (r.ucanrm  $$i) ; \
	then rm -f $$i; \
	fi; \
	done \
	fi; \
	if [ "*.[fp]tn" != "`echo *.[fp]tn`" ] ; \
	then \
	for i in *.[fp]tn; \
	do \
	if (r.ucanrm $$i) ; \
	then rm -f $$i; \
	fi; \
	done \
	fi ; \
	rm -f *.o
	-if [ "*.f90" != "`echo *.f90`" ] ; \
	then \
	for i in *.f90; \
	do \
	if (r.ucanrm $$i)  ; \
	then rm -f $$i; \
	fi; \
	done \
	fi; \
	if [ "*.cdk90" != "`echo *.cdk90`" ] ; \
	then \
	for i in *.cdk90; \
	do \
	if (r.ucanrm  $$i) ; \
	then rm -f $$i; \
	fi; \
	done \
	fi ; \
	if [ "*.ftn90" != "`echo *.ftn90`" ] ; \
	then \
	for i in *.ftn90; \
	do \
	if (r.ucanrm $$i) ; \
	then rm -f $$i; \
	fi; \
	done \
	fi
	rm -f *.o *.mod ;\
	rm -rf .fo/ ;\
	echo

qclean:
#Faire un petit menage. On enleve tous les .o et les .f produits a partir de .ftn/.ptn
	-if [ "*.[fp]tn" != "`echo *.[fp]tn`" ] ; \
	then \
	for i in *.[fp]tn ; \
	do \
	fn=$${i%.[fp]tn}; \
	rm -f $$fn.f; \
	done \
	fi ; \
	if [ "*.ftn90" != "`echo *.ftn90`" ] ; \
	then \
	for i in *.ftn90 ; \
	do \
	fn=$${i%.ftn90};\
	rm -f $$fn.f90; \
	done \
	fi
	-rm -f *.o *.mod ;\
	echo

