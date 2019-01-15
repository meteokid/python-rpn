!---------------------------------- LICENCE BEGIN -------------------------------
! GEM - Library of kernel routines for the GEM numerical atmospheric model
! Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!                       Environnement Canada
! This library is free software; you can redistribute it and/or modify it 
! under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, version 2.1 of the License. This library is
! distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
! without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
! PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
! You should have received a copy of the GNU Lesser General Public License
! along with this library; if not, write to the Free Software Foundation, Inc.,
! 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
!---------------------------------- LICENCE END ---------------------------------

!s/r itf_phy_restart
!
      subroutine itf_phy_restart (F_WorR_S, F_spin_L)
      implicit none

      character*(*) F_WorR_S
      logical F_spin_L
!author
!     M. Desgagne - Mars 2008
!
!revision
! v3_31 - Desgagne M.       - initial MPI version
! v4_14 - Dugas B.          - account for increment forcing data
!

#include <arch_specific.hf>
#include <clib_interface_mu.hf>
#include <gmm.hf>
#include "lun.cdk"
#include "lctl.cdk"
#include "grd.cdk"
#include "cstv.cdk"
#include "init.cdk"
#include "schm.cdk"
#include "rstr.cdk"
#include "path.cdk"
#include "ptopo.cdk"
#include "cfld.cdk"
#include "step.cdk"

      integer,external :: fnom,fclos

      character*512  fn
      character*16   datev,datev_infile,startindx
      logical spin_L,have_userbus_L,read_userbus_L
      integer dim,unf,ier,ibuf(4),datstp,gmmstat,errcode,yela,tousla,dim_infile
      real   rbuf(6)
      real*8 dayfrac, sec_in_day
      parameter (sec_in_day=86400.0d0)
      type(gmm_metadata) :: meta_busper
      real, pointer, dimension(:,:) :: BUSPER_3d
!
!     ---------------------------------------------------------------
!
! This code should all be transfered to the physics and
! be replaced by a call phy_restart(F_WorR_S)
! Within phy_restart we should also see the 
! call itf_cpl_restart(F_WorR_S) that we see below.
! That will bring the cpl interface to the proper level.

      if (F_WorR_S == 'W') then

      if ( .not. Schm_phyms_L ) return
!
      unf = 0
!
      dayfrac = dble(Step_kount) * Cstv_dt_8 / sec_in_day
      call incdatsd (datev,Step_runstrt_S,dayfrac)

      fn='restart_BUSPER'
      if (F_spin_L) fn= 'BUSPER4spinphy_'//trim(datev)

      ier = fnom (unf,fn,'SEQ+UNF',0)

      if (Lun_out.gt.0) write(Lun_out,3000) Lctl_step,trim(fn)

      write(unf) F_spin_L 

      if (F_spin_L) then
         call datp2f (datstp,datev)
         gmmstat = gmm_get ('BUSPER_3d',BUSPER_3d,meta_busper)
         dim = (meta_busper%l(1)%high-meta_busper%l(1)%low+1)*&
               (meta_busper%l(2)%high-meta_busper%l(2)%low+1)

         ibuf(1) = datstp
         ibuf(2) = dim
         ibuf(3) = Grd_ni
         ibuf(4) = Grd_nj

         rbuf(1) = Grd_dx
         rbuf(2) = Grd_dy
         rbuf(3) = Grd_xlon1
         rbuf(4) = Grd_xlat1
         rbuf(5) = Grd_xlon2
         rbuf(6) = Grd_xlat2

         write(unf) ibuf, rbuf
         write(unf) BUSPER_3d
      endif

      ier = fclos(unf)

      endif

      if (F_WorR_S == 'R') then
!     No busper recycling required if the physics is inactive
      if ( .not. Schm_phyms_L ) return

!     Basic setup for busper reading

      Rstri_user_busper_L = .false.
      have_userbus_L = (clib_fileexist(trim(Path_input_S)//'/BUSPER.tar') == CLIB_OK .and. Step_kount == 0)
      dayfrac = dble(Step_kount) * Cstv_dt_8 / sec_in_day
      call incdatsd (datev,Step_runstrt_S,dayfrac)

!     Check for possible busper flavours and completeness

      errcode= 0
      if (have_userbus_L) then
!        User has provided an input busper - make sure that it is complete
         write (startindx,'((i3.3),a1,(i3.3))') Ptopo_mycol,'-',Ptopo_myrow
         if (Grd_yinyang_L) then
            fn = '../../busper/'//trim(Grd_yinyang_S)//'/'//trim(startindx)//'/BUSPER4spinphy_'//trim(datev)
         else
            fn = '../busper/'//trim(startindx)//'/BUSPER4spinphy_'//trim(datev)
         endif
         if (Lun_out > 0) write(Lun_out,1000) trim(fn)
         if (clib_fileexist (trim(fn)).lt.1) errcode= -1
      endif

      call gem_error (errcode,'itf_phy_restart','Incomplete set of USER specified BUSPER4spinphy files')

      if (.not. have_userbus_L) then
!        Check for a valid restart busper, and return if none are found
         fn  = 'restart_BUSPER'
         yela= clib_fileexist (trim(fn))
         call rpn_comm_ALLREDUCE (yela,tousla,1,"MPI_INTEGER","MPI_SUM","grid",ier)
         if (tousla .eq. -Ptopo_numproc) return
         if (yela.lt.1) errcode= -1
      endif

      call gem_error (errcode,'itf_phy_restart','Incomplete set of restart_BUSPER files')

!     Open the busper file and check the spinup switch for reading
      unf = 0
      ier = fnom ( unf,fn,'SEQ+UNF+OLD',0 )
      if (Lun_out.gt.0) write(Lun_out,2000) Lctl_step,trim(fn)

      errcode= -1
      read (unf,err=999) spin_L

!     Read busper metadata and continue to read busper only if it matches current configuration
      if (spin_L) then
         read (unf,err=999) ibuf,rbuf
         call datf2p (datev_infile,ibuf(1))
         dim_infile = ibuf(2)
         gmmstat = gmm_get ('BUSPER_3d',BUSPER_3d,meta_busper)
         dim = (meta_busper%l(1)%high-meta_busper%l(1)%low+1)*&
               (meta_busper%l(2)%high-meta_busper%l(2)%low+1)
         if ( (datev_infile .ne.  datev      )   .or.  &
              (dim_infile   .ne.  dim        )   .or. &
              (ibuf(3)      .ne.  Grd_ni     )   .or. &
              (ibuf(4)      .ne.  Grd_nj     )   .or. &
              (rbuf(1)      .ne.  Grd_dx     )   .or. &
              (rbuf(2)      .ne.  Grd_dy     )   .or. &
              (rbuf(3)      .ne.  Grd_xlon1  )   .or. &
              (rbuf(4)      .ne.  Grd_xlat1  )   .or. &
              (rbuf(5)      .ne.  Grd_xlon2  )   .or. &
              (rbuf(6)      .ne.  Grd_xlat2  )  ) then
            if (Lun_out.gt.0) write(Lun_out,2006) 'BUSPER4spinphy_'//trim(datev)
            goto 999
         else
            read (unf,err=999) BUSPER_3d
         endif
!        Successful read of busper
         errcode= 0
         Rstri_user_busper_L = .true.
      else
         errcode= 0
      endif

!     Shutdown and error handling
999   ier = fclos(unf) 

      call gem_error (errcode,'itf_phy_restart','cannot read physics restart correctly')
      endif
      
 1000 format(/,'FOUND USER BUSPER INPUT - NOW LOOKING FOR PE-SPECIFIC RESTART AT',x,a)
 2000 format(/,'READING A PHYSICS RESTART FILE AT TIMESTEP #',I8,x,a, &
             /,'============================================')
 2006 format(/,'INCONSISTENT SET OF PHYSICS RESTART FILE: ',a/ &
               'WILL ABORT'/)
 3000 format(/,'WRITING A PHYSICS RESTART FILE AT TIMESTEP #',I8,x,a, &
             /,'============================================')
!
!     ---------------------------------------------------------------
!      
      return
      end
