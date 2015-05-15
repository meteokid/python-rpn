      subroutine checkdmpart
      implicit none

      integer, external ::  grid_nml2 , gem_nml , gemdm_config

#include <clib_interface_mu.hf>
#include "glb_ld.cdk"
#include "grd.cdk"
#include "lun.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "hzd.cdk"

      character*16 ndomains_S
      integer err,Pelocal,Petotal,npex,npey

      call RPN_COMM_init('',Pelocal,Petotal,1,1)

      Step_runstrt_S='2011020300'
      Step_dt=5

      Grd_yinyang_L = .false.
      Grd_yinyang_S = ''
      if (clib_getenv ('GEM_YINYANG',ndomains_S).ge.0) &
      Grd_yinyang_L = .true.
      Lun_out=6

      err = grid_nml2 ('./gem_settings.nml',G_lam)
      if (err .lt. 0) goto 987
      err = grid_nml2 ('print',G_lam)

      err = gem_nml   ('./gem_settings.nml')
      if (err .lt. 0) goto 987
      err = gemdm_config ()
      if (err .lt. 0) goto 987

      write (6,'(/a)') '  ====> READING npex, npey from STDIN:'
      read(5,*) npex, npey

      call domain_decomp2 ( npex, npey, .true. )

      call sol_transpose ( npex, npey, .true. )

      if (.not. G_lam) then
         if ((Hzd_lnr.gt.0.).or.(Hzd_lnr_theta.gt.0.) &
                            .or.(Hzd_lnr_tr   .gt.0.))& 
         call hzd_imp_transpose   ( npex, npey, .true. )
         call vspng_imp_transpose ( npex, npey, .true. )
      endif

      write (6,'(/a/)') '  ====> CHECKDMPART IS OK'

 987  call rpn_comm_FINALIZE(err)

      return
      end

! ex: echo "4 5" | r.mpirun -pgm ./checkdmpart_${BASE_ARCH}.Abs
! or echo "4 5" | ./checkdmpart_${BASE_ARCH}.Abs
