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

!**s/r hzd_solfft - parallel direct solution of high-order diffusion 
!                   equation with fffts


       subroutine hzd_solfft2(F_sol, F_Rhs_8, F_a_8, F_c_8, F_deltai_8, &
                   minx1, maxx1, minx2, maxx2, nx1, nx2, nx3, F_pwr   , &
                   minx,maxx,miny,maxy,gnk,gni,nil,njl,nkl            , &
                   F_opsxp0_8, F_opsyp0_8,F_cdiff,F_npex,F_npey)
      implicit none
#include <arch_specific.hf>
!
      integer  minx1, maxx1, minx2, maxx2 , nx1, nx2, nx3, F_pwr, &
               minx , maxx , miny , maxy  , gnk, gni, &
               njl  , nkl  , nil  , F_npex, F_npey
      real*8  F_opsxp0_8(*), F_opsyp0_8(*)              , &
                  F_a_8(1:F_pwr,1:F_pwr,minx2:maxx2,nx3), &
                  F_c_8(1:F_pwr,1:F_pwr,minx2:maxx2,nx3), &
             F_deltai_8(1:F_pwr,1:F_pwr,minx2:maxx2,nx3), &
             F_Rhs_8(minx:maxx,miny:maxy,gnk)
      real   F_cdiff, F_sol(minx:maxx,miny:maxy,gnk)
!
!author
!     Abdessamad Qaddouri
!
!revision
! v2_10 - Qaddouri A.        - initial version
! v3_02 - J. P. Toviessi     - remove data overflow bug
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_11 - Corbeil L.        - new RPNCOMM transpose
!
!object
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
!  F_sol        I/O      r.h.s. and result of horizontal diffusion
!  F_Rhs_8      I        work vector
!
!----------------------------------------------------------------
!
#include "ptopo.cdk"

      real*8  fdg1_8 ( miny:maxy , minx1:maxx1, gni+F_npex  ), &
              fdg2_8 (minx1:maxx1, minx2:maxx2, nx3+F_npey  ), &
              dn3_8  (minx1:maxx1, minx2:maxx2, F_pwr,nx3   ), &
              sol_8  (minx1:maxx1, minx2:maxx2, F_pwr,nx3   ), &
              fwft_8 ( miny:maxy , minx1:maxx1, gni+2+F_npex)
      real*8, parameter :: ZERO_8=0.0
      real*8  fact_8, pri
      integer o1,o2,i,j,k,jw,pi0,pin
      integer ki, kkii, ki0, kin, kilon, kitotal
!     __________________________________________________________________
!
      call hzd_solfft_lam2(F_sol, F_Rhs_8          , &
                              F_a_8, F_c_8, F_deltai_8, &
                   minx1, maxx1, minx2, maxx2, nx1, nx2, nx3, F_pwr, &
                   minx,maxx,miny,maxy,gnk,gni,nil,njl,nkl         , &
                   F_opsxp0_8, F_opsyp0_8,F_cdiff,F_npex,F_npey)
!
!     __________________________________________________________________
!
      return
      end

