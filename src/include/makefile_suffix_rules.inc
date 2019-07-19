include $(PROJECT_ROOT)/include/compiler.mk

INC = -I. -I$(PROJECT_ROOT)/include \
      -I$(PROJECT_ROOT)/librmn/PUBLIC_INCLUDES

.c.o:
	$(CC) -fPIC -O$(OPTIMIZ) $(CFLAGS) $(INC) -c $<

.o.a:
	ar rv $@ $<
	rm -f $*.o

.ftn.f .ftn90.f90:
	cpp $(INC) $< $@

.f.o .F.o .f90.o .F90.o:
	$(FC) -fPIC -O$(OPTIMIZ) $(FFLAGS) $(INC) -c $<


