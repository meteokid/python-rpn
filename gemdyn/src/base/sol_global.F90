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
!*s/r sol_global - Solution of the elliptic problem for global grids 'G'
!
      subroutine sol_global (F_sol_8, F_rhs_8, F_dg1, F_dg2, F_dwfft, F_ni, F_nj, F_nk)
      implicit none
#include <arch_specific.hf>

      integer F_ni, F_nj, F_nk
      real*8 F_rhs_8(F_ni,F_nj,F_nk), F_sol_8 (F_ni,F_nj,F_nk), &
             F_dg1(*), F_dg2(*), F_dwfft(*)

!
!author 
!     Michel Desgagne / Abdessamad Qaddouri -- January 2014
!
!revision
! v4_70 - Desgagne/Qaddouri  - initial version

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "ldnh.cdk"
#include "sol.cdk"
#include "opr.cdk"
#include "ptopo.cdk"
#include "fft.cdk"
#include "schm.cdk"
#include "trp.cdk"

      integer Gni, i, j, nev, NSTOR, dim
      real*8, dimension(:), allocatable :: wk_evec_8, abpt
!
!     ---------------------------------------------------------------
!
      if (Fft_fast_L) then

         call sol_fft_glb ( F_sol_8, F_rhs_8                       ,&
                      ldnh_maxx, ldnh_maxy, ldnh_nj                ,&
                      trp_12smax, trp_12sn, trp_22max, trp_22n     ,&
                      Schm_nith, G_ni, G_nj, Ptopo_npex, Ptopo_npey,&
                      Sol_ai_8, Sol_bi_8, Sol_ci_8, F_dg2, F_dwfft )
         
      else
         
         Gni= G_ni-Lam_pil_w-Lam_pil_e
         dim= Gni*Gni
         allocate ( wk_evec_8(Gni*Gni) )
         do j=1,Gni
         do i=1,Gni
            wk_evec_8((j-1)*Gni+i)= &
            Opr_xevec_8((j+Lam_pil_w-1)*G_ni+i+Lam_pil_w)
         enddo
         enddo
         
         call sol_mxma ( F_sol_8, F_rhs_8, wk_evec_8         ,&
              ldnh_maxx, ldnh_maxy, ldnh_nj, dim             ,&
              trp_12smax, trp_12sn, trp_22max, trp_22n       ,&
              G_ni, G_nj, G_nk, trp_12sn                     ,& 
              Ptopo_npex, Ptopo_npey                         ,&
              Sol_ai_8,Sol_bi_8,Sol_ci_8,F_dg1,F_dg2,F_dwfft )
         
         deallocate (wk_evec_8)

      endif
!
!     ---------------------------------------------------------------
! 
      return
      end
