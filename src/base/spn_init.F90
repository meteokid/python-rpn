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
!---------------------------------- LICENCE END ----------------------

!*s/r spn_init - initialize spectral nudging profile, filter

      subroutine spn_init
      use spn_work_mod
#include <arch_specific.hf>
      implicit none

!author
!     Minwei Qian (CCRD) & Bernard Dugas, Syed Husain  (MRB)  - summer 2015
!
!revision
! v4_80 - Qian, Dugas, Hussain            - initial version

#include "lun.cdk"
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "grd.cdk"
#include "lam.cdk"
#include "dcst.cdk"
#include "ptopo.cdk"
#include "cstv.cdk"
#include "ldnh.cdk"
#include "geomg.cdk"
#include "ver.cdk"
#include "spn.cdk"

      integer i,j,k,err1
      real t_turn, b_turn, pi2, nudging_tau
!
!----------------------------------------------------------------------
!
      if (.not.G_lam) return

      if(Spn_nudging_S.eq.'') return

      i = G_ni-Lam_pil_w-Lam_pil_e
      call itf_fft_nextfactor ( i )
      if ( i.ne.G_ni-Lam_pil_w-Lam_pil_e ) then
             call stop_mpi(-1,'spn_init','Error with grid N in X')
      endif

      j = G_nj-Lam_pil_s-Lam_pil_n
      call itf_fft_nextfactor ( j )
      if ( j.ne.G_nj-Lam_pil_s-Lam_pil_n ) then
        call stop_mpi(-1,'spn_init','Error with grid N in Y')
      endif
      pi2 = atan( 1.0_8 )*2.


      call up2low( Spn_nudging_S, Spn_nudging_S )
      call low2up( Spn_trans_shape_S, Spn_trans_shape_S )

      if (Lun_out > 0) write(Lun_out,1000) Spn_nudging_S


      ! Allocate once and for all

      allocate( prof(G_nk), &
                fxy(G_ni+2,G_nj+2), &
                Ldiff3D(ldnh_minx:ldnh_maxx,ldnh_miny:ldnh_maxy,G_nk-1), stat=err1 )

      if (err1 > 0) call stop_mpi(-1,'spn_init','Error in spn_init')

      call spn_calfiltre (G_ni-Lam_pil_w-Lam_pil_e, &
                          G_nj-Lam_pil_s-Lam_pil_n)

      prof=0.

      ! nudging_tau is nudging time scale in hours
      ! t_turn and b_turn are top and bottom turnning points

      t_turn= max( Spn_up_const_lev,Ver_hyb%m(   1  ) )
      b_turn= min( Spn_start_lev   ,Ver_hyb%m(G_nk-1) )
      nudging_tau = Spn_relax_hours

      if (Spn_trans_shape_S == 'COS2' ) then

         do k=1,G_nk
            if (Ver_hyb%m(k) <= b_turn .and. Ver_hyb%m(k) >= t_turn) then
               prof(k) = cos(pi2-pi2*(b_turn-Ver_hyb%m(k))/(b_turn-t_turn))
            elseif (Ver_hyb%m(k) < t_turn) then
               prof(k)=1.
            else
               prof(k)=0.
            endif
            prof(k) = prof(k)*prof(k)
         enddo


      elseif (Spn_trans_shape_S == 'LINEAR' ) then

         do k=1,G_nk
            if (Ver_hyb%m(k) <= b_turn .and. Ver_hyb%m(k) >= t_turn) then
               prof(k) =  (b_turn-Ver_hyb%m(k))/(b_turn-t_turn)
            elseif (Ver_hyb%m(k) < t_turn) then
               prof(k)=1.
            else
               prof(k)=0.
            endif
         enddo

      else

         if (Lun_out > 0) write(Lun_out,1001) Spn_trans_shape_S
         call stop_mpi(-1,'spn_init','Error in spn_init')


      endif

      do k=1,G_nk
         prof(k) = prof(k) * Cstv_dt_8/3600./nudging_tau
      enddo

 1000 format(/' In SPN_INIT, Spn_nudging_S = ',A8/)
 1001 format(/' In SPN_INIT, unknown Spn_trans_shape_S ',A8/)
 1002 format(/' In SPN_INIT, Cstv_dt_8 =  ',F12.4/)
 1003 format(/' In SPN_INIT, Spn_trans_shape_S ',A8/)
!
!----------------------------------------------------------------------
!
      return
      end subroutine spn_init
