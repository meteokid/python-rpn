include ../Makefile_base
include ../Makefile_$(ARCH)

BASEDIR = $(PWD)/..

UTILSSRC = $(shell ls *.ftn90)
UTILSOBJ = $(patsubst %.ftn90, %.o, $(UTILSSRC))

all: $(UTILSOBJ)

#
# ALL dependencies ...
#
vect_mod.o: vect_mod.ftn90 rmnlib_basics.hf
get_corners_xy.o: get_corners_xy.ftn90

