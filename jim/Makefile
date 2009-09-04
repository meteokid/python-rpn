include ../Makefile_base
include ../Makefile_$(ARCH)

BASEDIR = $(PWD)/..

PIMSRC = $(shell ls *.ftn90)
PIMOBJ = $(patsubst %.ftn90, %.o, $(PIMSRC))

default: all
all: $(PIMOBJ)

#
# ALL dependencies ...
#

jim_grid_mod.o: jim_grid_mod.ftn90 jim_const.hf $(BASEDIR)/utils/rmnlib_basics.hf $(BASEDIR)/utils/vect_mod.o 

OTHERS   =  $(RPNCOMM) lapack blas massvp4 bindcpu_002 $(LLAPI) $(IBM_LD)
LIBS     = $(OTHERS)

