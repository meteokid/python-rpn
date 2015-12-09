      subroutine checkdmpart
      use iso_c_binding
      implicit none

      integer, external ::  grid_nml2 , gem_nml , gemdm_config

#include <clib_interface_mu.hf>
#include "glb_ld.cdk"
#include "grd.cdk"
#include "lun.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "hzd.cdk"
      include "rpn_comm.inc"

      external dummy_checkdm
      integer, external :: domain_decomp3,sol_transpose2, &
                 hzd_imp_transpose2,vspng_imp_transpose2

      integer pe_xcoord(1000), pe_ycoord(1000)
      character*16 ndomains_S,npex_S,npey_S
      integer err,ierr(4),Pelocal,Petotal,npex,npey,i,max_io_pes

      npex=1 ; npey=1
      call RPN_COMM_init(dummy_checkdm,Pelocal,Petotal,npex,npey)

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

      if (clib_getenv ('Ptopo_npex',npex_S).lt.0) then
         write (6,1001) 'Ptopo_npex'
         goto 987
      else
         read (npex_S,*,end=33,err=33) npex
         goto 201
  33     write (6,1002) 'Ptopo_npex',npex_S
         goto 987
      endif
 201  if (clib_getenv ('Ptopo_npey',npey_S).lt.0) then
         write (6,1001) 'Ptopo_npey'
         goto 987
      else
         read (npey_S,*,end=43,err=43) npey
         goto 301
  43     write (6,1002) 'Ptopo_npey',npey_S
         goto 987
      endif

 301  ierr=0
      ierr(1)= domain_decomp3 ( npex, npey, .true. )
      ierr(2)= sol_transpose2 ( npex, npey, .true. )

      if (.not. G_lam) then
         if ((Hzd_lnr.gt.0.).or.(Hzd_lnr_theta.gt.0.) &
                            .or.(Hzd_lnr_tr   .gt.0.))& 
         ierr(3)= hzd_imp_transpose2   ( npex, npey, .true. )
         ierr(4)= vspng_imp_transpose2 ( npex, npey, .true. )
      endif

      if (minval(ierr) .lt. 0 ) goto 987

      write (6,'(/a//a,i6,a)') '  ====> CHECKDMPART IS OK',&
      ' Looping RPN_COMM_io_pe_valid_set over ',npex*npey,' Pes'

 987  max_io_pes=npex*npey
      do i= 1, npex*npey
         err= RPN_COMM_io_pe_valid_set (pe_xcoord,pe_ycoord,i,&
                                        npex,npey,.false.,0)
         if (err.ne.0) then
            max_io_pes= i-1
            exit
         endif
      end do

      write (6,'(/a,i9/)') '  ====> MAX_PES_IO= ',max_io_pes

      call rpn_comm_FINALIZE(err)

 1001 format (/' =====> Error: Env variable ',a,' is undefined'/)
 1002 format (/' =====> Error: Env variable ',a,' is incorrectly defined ',a/)

      return
      end

subroutine dummy_checkdm
return
end

! export Ptopo_npex=4 ; export Ptopo_npey=3
! r.run_in_parallel -pgm checkdmpart_${BASE_ARCH}.Abs -npex 1

