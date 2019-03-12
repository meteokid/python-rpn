include ../Makefile_base.mk
include ../$(EC_ARCH)/Makefile.inc.mk

BASEDIR = $(PWD)/..

UTILSSRC = $(shell ls *.ftn90)
#UTILSSRCC = $(shell ls *.c)
UTILSOBJ = $(patsubst %.ftn90, %.o, $(UTILSSRC))
#UTILSOBJC = $(patsubst %.c, %.o, $(UTILSSRCC))

all: $(UTILSOBJ) $(UTILSOBJC)

#
# ALL dependencies ...
#
vect_mod.o: vect_mod.ftn90 rmnlib_basics.hf
get_corners_xy.o: get_corners_xy.ftn90 int_macros.hf
#py_capi_ftn_utils.o: py_capi_ftn_utils.c py_capi_ftn_utils.h

test: get_corners_xy.o
	s.compile -src tests/test_get_corners_xy.ftn90 -obj get_corners_xy.o -o test_get_corners_xy.Abs
	./test_get_corners_xy.Abs

clean:
	rm -f *.o
