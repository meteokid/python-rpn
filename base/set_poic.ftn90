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
!**s/r set_poic - preparation of projection matrix in the east-west, 
!                   C grid model
!

!
      subroutine set_poic ( F_eval_8, F_evec_8, F_xp0_8, F_xp2_8, &
                                                   F_npts, NSTOR )
      implicit none
#include <arch_specific.hf>
!
      integer F_npts, NSTOR
      real*8  F_eval_8(NSTOR)  , F_evec_8(NSTOR,NSTOR)
      real*8  F_xp0_8 (NSTOR,3), F_xp2_8 (NSTOR,3)
!
!author
!     jean cote - sept 1995 - from setpois
!
!revision
! v2_00 - Desgagne/Lee      - initial MPI version (from setpoic v1_03)
! v3_00 - Desgagne & Lee    - Lam configuration
! v4_40 - Qaddouri A.       - call prepoic indicating "Dirichlet" for Yin-Yang
!                             instead of "Neumann" (boundary conditions)
!
!object
!     See above id
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_eval_8     O    - eigenvalue vector
! F_evec_8     O    - eigenvector matrix
! F_xp0_8      I    - alongx projector diagonals
! F_xp2_8      I    - alongx second derivatives diagonals
! F_npts       I    - number of points to operate on
!

#include "dcst.cdk"
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "grd.cdk"
!
      integer i, j
      real*8 a_8(F_npts,F_npts), b_8(F_npts,F_npts), r_8(F_npts)
      real*8 zero,one,two,pdfaz
      parameter( zero = 0.0 , one = 1.0, two = 2.0 )
!
! --------------------------------------------------------------------
!
!     put input arguments in real*8 arrays
!
      F_evec_8= zero ; F_eval_8= zero ; a_8= zero ; b_8= zero

      do i = 1, F_npts-1
         a_8(i,i+1) = F_xp2_8(i+Lam_pil_w,3)
         a_8(i,i  ) = F_xp2_8(i+Lam_pil_w,2)
         a_8(i+1,i) = a_8(i,i+1)
         b_8(i,i+1) = F_xp0_8(i+Lam_pil_w,3)
         b_8(i,i  ) = F_xp0_8(i+Lam_pil_w,2)
         b_8(i+1,i) = b_8(i,i+1)
      end do

      a_8(F_npts,F_npts)= F_xp2_8(NSTOR-Lam_pil_e,2)
      b_8(F_npts,F_npts)= F_xp0_8(NSTOR-Lam_pil_e,2)

      if (G_lam) then
          call geneigl3 (r_8, a_8, b_8, F_npts, F_npts, 3*F_npts-1)
          if (.not.Grd_yinyang_L) then
             pdfaz= ONE/sqrt((G_xg_8(G_ni-glb_pil_e)-G_xg_8(glb_pil_w)))
             r_8(1)          = 0.0d0
             a_8(1:F_npts,1) = pdfaz
          endif
      else
          a_8(F_npts,1  ) = F_xp2_8(F_npts,3)
          a_8(1,F_npts  ) =  a_8(F_npts,1)
          b_8(F_npts,1  ) = F_xp0_8(F_npts,3)
          b_8(1,F_npts  ) =  b_8(F_npts,1)
          call geneigl3 (r_8, a_8, b_8, F_npts, F_npts, 3*F_npts-1)
          pdfaz = ONE / sqrt( TWO*Dcst_pi_8 )
          r_8(1)          = 0.0d0
          a_8(1:F_npts,1) = pdfaz
      endif
!
!     put real*8 results in output arguments arrays
!     NOTE: for non-LAM case, the loops are F_npts=NSTOR
!
      do j = 1,F_npts
      do i = 1,F_npts
         F_evec_8(i+Lam_pil_w,j+Lam_pil_s) = a_8(i,j)
      enddo
      enddo

      do i = 1, F_npts
         F_eval_8(i+Lam_pil_w) = r_8(i)
      end do
!
! --------------------------------------------------------------------
!
      return
      end
