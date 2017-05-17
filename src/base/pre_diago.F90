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

!**s/r  pre_diago -  Diagonal preconditioning  
!

      subroutine pre_diago( Sol, Rhs, Minx, Maxx, Miny, Maxy, &
                            nil,njl, minx1, maxx1, minx2, maxx2, Nk )
      use gem_options
      implicit none
#include <arch_specific.hf>
!
      integer Minx, Maxx, Miny, Maxy, nil, njl, &
              minx1, maxx1, minx2, maxx2,Nk
      real*8  Rhs(Minx:Maxx,Miny:Maxy,Nk), &
              Sol(Minx:Maxx,Miny:Maxy,Nk)
!
! author    Abdessamad Qaddouri -  December 2006
!
!revision
! v3_30 - Qaddouri A.       - initial version
!
#include "glb_ld.cdk"
#include "opr.cdk"
#include "cstv.cdk"
#include "sol.cdk"
!
      integer j,i,k,ii,jj
      real*8  stencil1,cst,di_8,wwk(nk)
!
!     ---------------------------------------------------------------
!
      do k=1, NK
         wwk(k)= (Cstv_hco1_8+Cstv_hco0_8*Opr_zeval_8(k))
      enddo                                                 
!
      do k = 1,Nk 
         do j=1+sol_pil_s, njl-sol_pil_n
            jj=j+l_j0-1
            di_8= Opr_opsyp0_8(G_nj+jj) / cos( G_yg_8 (jj) )**2
            do i=1+sol_pil_w, nil-sol_pil_e
               ii=i+l_i0-1
               cst= wwk(k)
               stencil1=cst*Opr_opsxp0_8(G_ni+ii)* &
                 Opr_opsyp0_8(G_nj+jj) +Opr_opsxp2_8(G_ni+ii)*di_8+ &
                 Opr_opsxp0_8(G_ni+ii)*Opr_opsyp2_8(G_nj+jj)
!
               Rhs(i,j,k) =sol(i,j,k)/stencil1
            enddo
         enddo
      enddo
!
!     ---------------------------------------------------------------
!
      return
      end

