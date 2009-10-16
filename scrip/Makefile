include ../Makefile_base
include ../Makefile_$(ARCH)

BASEDIR = $(PWD)/..

#PIMSRC = $(shell ls *.ftn90)
#PIMOBJ = $(patsubst %.ftn90, %.o, $(PIMSRC))

SCRIPOBJ= \
	kinds_mod.o \
	constants.o \
	grids.o \
	remap_vars.o \
	remap_distwgt.o \
	remap_conserv.o \
	remap_bilinear.o \
	remap_bicubic.o \
	timers.o \
	remap_write.o \
	scrip.o \
	remap.o \
	scrip_interface.o

default: all
all: $(SCRIPOBJ)

test:  $(SCRIPOBJ) scrip_interp_test.o
	r.build -obj scrip_interp_test.o $(SCRIPOBJ) $(BASEDIR)/utils/get_corners_xy.o -arch $(EC_ARCH) -libpath $(LIBPATH) -librmn $(RMNLIB) -libappl "$(LIBS)" $(MPI) -o scrip_interp_test_$(EC_ARCH).Abs;\
	/bin/rm -f bidon.o
	./scrip_interp_test_$(EC_ARCH).Abs

#
# ALL dependencies ...
#

#OTHERS   =  $(RPNCOMM) lapack blas massvp4 bindcpu_002 $(LLAPI) $(IBM_LD)
#LIBS     = $(OTHERS)

kinds_mod.o: kinds_mod.f

constants.o: constants.f kinds_mod.o

iounits.o: iounits.f kinds_mod.o

grids.o: grids.f kinds_mod.o constants.o iounits.o

remap_vars.o: remap_vars.f kinds_mod.o constants.o grids.o

remap_conserv.o: remap_conserv.f kinds_mod.o constants.o \
        timers.o remap_vars.o grids.o

remap_distwgt.o: remap_distwgt.f kinds_mod.o constants.o \
        remap_vars.o grids.o

remap_bilinear.o: remap_bilinear.f kinds_mod.o constants.o \
        remap_vars.o grids.o timers.o

remap_bicubic.o: remap_bicubic.f kinds_mod.o constants.o \
        remap_vars.o grids.o

timers.o: timers.f kinds_mod.o constants.o

remap_write.o: remap_write.f kinds_mod.o constants.o \
        remap_vars.o grids.o

remap.o: remap.f kinds_mod.o constants.o

scrip.o: scrip.f kinds_mod.o constants.o iounits.o timers.o \
        remap_vars.o grids.o remap_conserv.o remap_distwgt.o \
        remap_bilinear.o remap_bicubic.o remap_write.o

scrip_interp_test.o: scrip_interp_test.ftn90

#scrip_interface.f90: scrip_interface.ftn90
scrip_interface.o: scrip_interface.ftn90 grids.o remap_vars.o kinds_mod.o
#scrip_interface.o: scrip_interface.ftn90

clean:
	rm -f *.o *.mod scrip_interp_test.f90 scrip_interface.f90
