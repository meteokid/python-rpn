ifneq (,$(DEBUGMAKE))
$(info ## ====================================================================)
$(info ## File: Makefile.user.mk)
$(info ## )
endif

#VERBOSE = 1
#OPTIL   = 2
#OMP     = -openmp
#MPI     = -mpi
#LFLAGS = 
#FFLAGS  = '-g -C -traceback -warn all'
#CFLAGS  =
#LIBAPPL = 
#LIBPATH_USER = 


maintest_vgrid_wb=test_vgrid_wb.Abs
test_vgrid_wb_abs: | test_vgrid_wb_rm $(BINDIR)/$(maintest_vgrid_wb)
	ls -l $(BINDIR)/$(maintest_vgrid_wb)
test_vgrid_wb_rm:
	rm -f $(BINDIR)/$(maintest_vgrid_wb)
$(BINDIR)/$(maintest_vgrid_wb): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="test_vgrid_wb" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)

maintest_timestr=test_timestr.Abs
test_timestr: | test_timestr_rm $(BINDIR)/$(maintest_timestr)
	ls -l $(BINDIR)/$(maintest_timestr)
test_timestr_rm:
	rm -f $(BINDIR)/$(maintest_timestr)
$(BINDIR)/$(maintest_timestr): | $(MODELUTILS_VFILES)
	export MAINSUBNAME="test_timestr" ;\
	export ATM_MODEL_NAME="$${MAINSUBNAME}" ;\
	export ATM_MODEL_VERSION="$(MODELUTILS_VERSION)" ;\
	export RBUILD_LIBAPPL="$(MODELUTILS_LIBS_V) $(MODELUTILS_LIBS_DEP)" ;\
	export RBUILD_COMM_STUBS=$(LIBCOMM_STUBS) ;\
	$(RBUILD4objNOMPI)


ifneq (,$(DEBUGMAKE))
$(info ## ==== Makefile.user.mk [END] ========================================)
endif
