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

!**s/r dynstep -  Control of the dynamical timestep of the model

      subroutine dynstep
      implicit none
#include <arch_specific.hf>

!author 
!     Michel Roch - rpn - nov 1993

!revision
! v2_00 - Desgagne M.       - initial MPI version
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_60 - Lee V.            - call yyg_nest_tracers before t02t1

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "orh.cdk"
#include "init.cdk"
#include "lctl.cdk"
#include "step.cdk"
#include "schm.cdk"
#include "vtopo.cdk"
#include "grd.cdk"
#include "tr3d.cdk"
#include "vt1.cdk"

      character(len=GMM_MAXNAMELENGTH) :: tr_name
      logical first_L, yyblend
      integer itraj, n, istat, keep_itcn
      real, pointer, dimension(:,:,:) :: tr1
!
!     ---------------------------------------------------------------
!
      if (Lun_debug_L) write(Lun_out,1000)
      call timing_start2 ( 10, 'DYNSTEP', 1 )

!     first_L is TRUE  for the first timestep
!           or the first timestep after digital filter initialisation

      first_L = (Step_kount.eq.1).or.(.not.Init_mode_L .and.  &
                 Step_kount.eq.(Init_dfnp+1)/2)

      keep_itcn = Schm_itcn

      itraj = Schm_itraj

      if (Schm_bitpattern_L) then
         call pospers ()
      else
         if ( first_L) then
            call pospers ()
            itraj = max( 5, Schm_itraj )
         endif
      endif
    
      if (Lun_debug_L) write(Lun_out,1005) Schm_itcn-1

      call psadj_init

      do Orh_icn = 1,Schm_itcn-1
    
         call tstpdyn (itraj)
         itraj = Schm_itraj
         
         call hzd_momentum
         
      end do
      
      if (Lun_debug_L) write(Lun_out,1006)
  
      Orh_icn=Schm_itcn
 
      call tstpdyn ( Schm_itraj )

      call tracers_step (.true. )

      call psadj

      call tracers_step (.false.)

!     ------------------------------------------------------------
!     C	  When the timestep is completed, rename all the
!     C        variables at time level t1 -> t0 and rename all the
!     C        variables at time level t0 -> t1 for the next timestep
!     ------------------------------------------------------------

      call t02t1 ()

      if (Grd_yinyang_L) then
         do n= 1, Tr3d_ntr
            tr_name = 'TR/'//trim(Tr3d_name_S(n))//':P'
            istat = gmm_get(tr_name, tr1)
            call yyg_xchng (tr1 , l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                            .true., 'CUBIC')
         end do
         istat = gmm_get(gmmk_wt1_s , wt1)
         call yyg_xchng (wt1 , l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                         .true., 'CUBIC')
         yyblend= (Schm_nblendyy .gt. 0)
         if (yyblend) &
         call yyg_blend (mod(Step_kount,Schm_nblendyy).eq.0)
      else
         call nest_gwa
         call spn_main
      endif

      call hzd_main_stag

      call pw_update_GPW
      call pw_update_UV
      call pw_update_T

      if ( Lctl_step-Vtopo_start .eq. Vtopo_ndt) Vtopo_L = .false.

      Schm_itcn = keep_itcn

      call timing_stop ( 10 )

 1000 format( &
      /,'CONTROL OF DYNAMICAL STEP: (S/R DYNSTEP)', &
      /,'========================================'/)
 1005 format( &
      /3X,'##### Crank-Nicholson iterations: ===> PERFORMING',I3, &
          ' timestep(s) #####'/)
 1006 format( &
      /3X,'##### Crank-Nicholson iterations: ===> DONE... #####'/)
!
!     ---------------------------------------------------------------
!
      return
      end
