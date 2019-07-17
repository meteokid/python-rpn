
ifeq ($(BASE_ARCH),$(EC_ARCH))
   $(info ERROR: please setup the compilation environment by typing the folowing:)
   $(info . setup.dot)
   $(error )
endif

DESCRIP_PATH = ../src

DIST = levels_with_ref lowLevel_levels topLevel_levels use_dpidpis use_dpidpis_withref_prof use_new_gen use_new_read c_simple_pressure c_use_new_read c_use_build

default: distclean all

all: $(DIST)

pressure_for_all_vcode: pressure_for_all_vcode.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

levels_with_ref: levels_with_ref.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

lowLevel_levels: lowLevel_levels.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

topLevel_levels: topLevel_levels.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

use_dpidpis: use_dpidpis.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

use_dpidpis_withref_prof: use_dpidpis_withref_prof.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

use_new_gen: use_new_gen.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

use_new_read: use_new_read.F90 $(DESCRIP_PATH)/libdescrip.a
	s.compile -optf ="$(MY_FFLAGS) $(FFLAGS)" -includes $(DESCRIP_PATH) -src $@.F90 -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

c_simple_pressure: c_simple_pressure.c $(DESCRIP_PATH)/libdescrip.a
	s.compile -bidon c -main $@ -optc ="$(MY_CFLAGS) $(CFLAGS)" -includes $(DESCRIP_PATH) -src $@.c -o $@ -libpath $(DESCRIP_PATH) -libappl rmn descrip

c_use_new_read: c_use_new_read.c $(DESCRIP_PATH)/libdescrip.a
	s.compile -bidon c -main $@ -optc ="$(MY_CFLAGS) $(CFLAGS)" -includes $(DESCRIP_PATH) -src $@.c -o $@ -libpath $(DESCRIP_PATH) -libappl rmn descrip
c_pressure_for_all_vcode: c_pressure_for_all_vcode.c $(DESCRIP_PATH)/libdescrip.a
	s.compile -bidon c -main $@ -optc ="$(MY_CFLAGS) $(CFLAGS)" -includes $(DESCRIP_PATH) -src $@.c -o $@ -libpath $(DESCRIP_PATH) -libappl rmn descrip

c_use_build: c_use_build.c $(DESCRIP_PATH)/libdescrip.a
	s.compile -bidon c -main $@ -optc ="$(MY_CFLAGS) $(CFLAGS)" -includes $(DESCRIP_PATH) -src $@.c -o $@ -libpath $(DESCRIP_PATH) -libappl descrip -librmn

$(DESCRIP_PATH)/libdescrip.a: $(DESCRIP_PATH)/*.c $(DESCRIP_PATH)/*.F90 $(DESCRIP_PATH)/*.h $(DESCRIP_PATH)/*.hc
	cd $(DESCRIP_PATH) ;\
	make;\

clean:
	rm -f *.f90 *.o *.mod test_output.fst

distclean: clean
	rm -f $(DIST)

tild:
	rm -f *~
