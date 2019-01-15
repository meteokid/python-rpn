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
!**s/r set_fft - determine if fast fourier transforms is needed
!

!
      subroutine set_fft
      implicit none
#include <arch_specific.hf>
!
!author 
!     michel roch - rpn - june 1993
!
!revision
! v2_00 - Lee V.            - initial MPI version (from setfft v1_03)
! v2_40 - Qaddouri A.       - adjust for LAM version
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_30 - Tanguay M.        - Abort if LAM adjoint not FFT 
! v4_40 - Qaddouri A.       _ Adjust for Yin-Yang FFT (sine, not cosine)
!
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "fft.cdk"
#include "hzd.cdk"
#include "sol.cdk"
#include "grd.cdk"
#include "lun.cdk"

      integer npts,onept,next_down
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(Lun_out,1000)

      Fft_fast_L = .false.

      if ((Sol_type_S.ne.'DIRECT').and.(Hzd_type_S.ne.'HO_IMP')) then
         if (Lun_out.gt.0) write (Lun_out,*) &
         'FFT not usefull in this model configuration'
         return
      endif

      if ( .not. sol_fft_L ) then
         if (Lun_out.gt.0) write (Lun_out,*) &
         'FFT not requested: sol_fft_L= .false.'
         return
      endif

      onept= 0
      if (Grd_yinyang_L) onept= 1
      npts= G_ni-Lam_pil_w-Lam_pil_e+onept

      call itf_fft_nextfactor2 ( npts, next_down )

      if ( npts .ne. G_ni-Lam_pil_w-Lam_pil_e+onept ) then
         if (Lun_out.gt.0) write (Lun_out,3001) &
         G_ni-Lam_pil_w-Lam_pil_e+onept,npts,next_down
         return
      endif
!
      Fft_fast_L= .true.
      if (Lun_out.gt.0) write(Lun_out,*) 'Fft_fast_L = ',Fft_fast_L
!
 1000 format( &
      /,'COMMON INITIALIZATION AND PREPARATION FOR FFT (S/R SET_FFT)', &
      /,'===========================================================')
 3001 format ('Fft_fast_L = .false. ====> NI = ',i6,' NOT FACTORIZABLE' &
              /'Neighboring factorizable G_NIs are: ',i6,' and',i6)
!
!     ---------------------------------------------------------------
!
      return
      end
