include ../Makefile_base
include ../Makefile_$(ARCH)

MPI    = 
RMNLIB = rmnbeta

BASEDIR = $(PWD)/..

PIMSRC = $(shell ls *.ftn90)
PIMOBJ = $(patsubst %.ftn90, %.o, $(PIMSRC))

default: all
all: $(PIMOBJ)

#
# ALL dependencies ...
#

jim_grid_mod.o: jim_grid_mod.ftn90 jim_const.hf $(BASEDIR)/utils/rmnlib_basics.hf $(BASEDIR)/utils/vect_mod.o 

jim_grid_test.o: jim_grid_test.ftn90

test: jim_grid_test.o jim_grid_mod.o
	r.build -obj jim_grid_test.o jim_grid_mod.o $(BASEDIR)/utils/vect_mod.o -arch $(EC_ARCH) -libpath $(LIBPATH) -librmn $(RMNLIB) -libappl "$(LIBS)" $(MPI) -o jim_grid_test_$(EC_ARCH).Abs;\
	/bin/rm -f bidon.o
	./jim_grid_test_$(EC_ARCH).Abs

OTHERS   =  $(RPNCOMM) lapack blas massvp4 bindcpu_002 $(LLAPI) $(IBM_LD)
LIBS     = $(OTHERS)
