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

!**s/r nest_bcs

      subroutine nest_bcs
      implicit none
#include <arch_specific.hf>

!author 
!     Michel Desgagne   - Summer 2010
!
!revision
! v4_14 - Desgagne M.          - initial version

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lam.cdk"
#include "schm.cdk"
#include "theo.cdk"
#include "nest.cdk"
#include "rhsc.cdk"
#include "cstv.cdk"

      integer i,j,k,gmmstat
!
!----------------------------------------------------------------------
!
      if (.not. Lam_ctebcs_L) call nest_intt

      if (Schm_theoc_L) then
         if (north+south+west+east.ge.1) then
            if ( Theo_case_S .eq. 'BUBBLE'   .or.    &
                 Theo_case_S .eq. 'BUBBLE_G' .or.    &
                 Theo_case_S .eq. '2_BUBBLES' ) then
               call mirror ()
            else
               call nest_bcs_t0 ()
            endif
            call slabsym ()
         endif
      else

         call nest_bcs_t0 ()

      endif

!**************************************
! Apply HORIZONTAL BOUNDARY CONDITIONS 
!**************************************

      gmmstat = gmm_get (gmmk_rhsu_s  , rhsu  )
      gmmstat = gmm_get (gmmk_rhsv_s  , rhsv  )
      gmmstat = gmm_get (gmmk_nest_u_s, nest_u)
      gmmstat = gmm_get (gmmk_nest_v_s, nest_v)

      if (l_west) then
!$omp do
         do k=1,l_nk
         do j= 1+pil_s, l_nj-pil_n
               rhsu (pil_w,j,k) = Cstv_invT_8 * nest_u(pil_w,j,k)
         enddo
         enddo
!$omp enddo
      endif

      if (l_east) then
!$omp do
         do k=1,l_nk
         do j= 1+pil_s, l_nj-pil_n
            rhsu (l_ni-pil_e,j,k) = Cstv_invT_8 * nest_u(l_ni-pil_e,j,k)
         enddo
         enddo
!$omp enddo
      endif

      if (l_south) then
!$omp do
         do k=1,l_nk
         do i= 1+pil_w, l_ni-pil_e
            rhsv (i,pil_s,k) = Cstv_invT_8 * nest_v(i,pil_s,k)
         enddo
         enddo
!$omp enddo
      endif

      if (l_north) then
!$omp do
         do k=1,l_nk
         do i= 1+pil_w, l_ni-pil_e
            rhsv (i,l_nj-pil_n,k) = Cstv_invT_8 * nest_v(i,l_nj-pil_n,k)
         enddo
         enddo
!$omp enddo
      endif
!
!----------------------------------------------------------------------
!
      return
      end
