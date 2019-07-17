
# Set default fortran compiler to gfortran.
ifeq ($(origin FC),default)
FC = gfortran
endif

# Extra flags required for gfortran.
ifneq (,$(findstring gfortran,$(FC)))
FFLAGS := $(FFLAGS) -fcray-pointer -ffree-line-length-none
endif

