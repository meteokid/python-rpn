
.SUFFIXES :
.SUFFIXES : .c .o .f90 .f .ftn .ftn90 .cdk90 .tmpl90 .F .FOR .F90 .itf90 .inc

RCOMPIL = s.compile $(MPI) $(OMP) -includes ./ $(INCLUDE_PATH) $(INCLUDE_MOD)
RBUILD  = s.compile
FCOMPF = 
CCOMPF =
COMPF = 
FC   = $(RCOMPIL) -defines "=$(DEFINE)" -O $(OPTIL) -optf="$(FFLAGS)" $(COMPF) $(FCOMPF) -src
FTNC = $(RCOMPIL) -defines "=$(DEFINE)"             -optf="$(FFLAGS) $(CPPFLAGS)" -P $(COMPF) $(FCOMPF) -src
CC   = $(RCOMPIL) -defines "=$(DEFINE)" -O $(OPTIL) -optc="$(CFLAGS)" $(COMPF) $(CCOMPF) -src

.c.o:
	$(CC) $<
.f.o:
	$(FC) $<
.f90.o:
	$(FC) $<
.ftn.o:
	rm -f $*.f90
	$(FC) $<
.ftn90.o:
	rm -f $*.f90
	$(FC) $<
.cdk90.o:
	$(FC) $<
.tmpl90.o:
	s.tmpl90.ftn90 < $<  > $*.ftn90
	s.ftn90 -c -o $@ -src $(EC_ARCH)/$*.ftn90 $(COMPILE_FLAGS) $(FFLAGS)

.ftn.f:
	rm -f $*.f
	$(FTNC) $<
.ftn90.f90:
	rm -f $*.f90
	$(FTNC) $<
.cdk90.f90:
	rm -f $*.f90
	$(FTNC) $<
.tmpl90.ftn90:
	s.tmpl90.ftn90 < $<  > $@

# .ftn90.itf90:
# 	$(FTNC) $< -defines =-DAPI_ONLY ; mv -f $*.f90 $*.itf90
# 	#mu.ftn2f -f90 -defines "=$(DEFINE)" -optf="$(FFLAGS) $(CPPFLAGS)" -P $(COMPF) $(FCOMPF) -src $<  > $@
# 	#r.gppf -lang-f90+ -chop_bang -gpp -F -D__FILE__=\"#file\" -D__LINE__=#line $
#{vincludes[@]} -DAPI_ONLY $< > $@

# .c.o:
# 	s.cc -c -o $@ -src $< $(COMPILE_FLAGS) $(CFLAGS) 
# .f.o:
# 	s.f77 -c -o $@ -src $< $(COMPILE_FLAGS) $(FFLAGS)
# .f90.o:
# 	s.f90 -c -o $@ -src $< $(COMPILE_FLAGS) $(FFLAGS)
# .ftn.o:
# 	s.ftn -c -o $@ -src $< $(COMPILE_FLAGS) $(FFLAGS)
# .ftn90.o:
# 	s.ftn90 -c -o $@ -src $<  $(COMPILE_FLAGS) $(FFLAGS)
# .cdk90.o:
# 	s.ftn90 -c -o $@ -src $<  $(COMPILE_FLAGS) $(FFLAGS)
.for.o:
	s.f77 -c -o $@ -src $< $(COMPILE_FLAGS) $(FFLAGS)
.F.o:
	s.f77 -c -o $@ -src $<  $(COMPILE_FLAGS) $(FFLAGS)
.FOR.o:
	s.f77 -c -o $@ -src $<  $(COMPILE_FLAGS) $(FFLAGS)
.F90.o:
	s.f90 -c -o $@ -src $<  $(COMPILE_FLAGS) $(FFLAGS)

#%_interface.cdk90 : %.tmpl90
#	FileName=$@ ; cat $< | r.tmpl90.ftn90 - $${FileName%.ftn90}
