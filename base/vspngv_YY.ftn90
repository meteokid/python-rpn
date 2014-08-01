!----------------------------------LICENCE BEGIN -------------------------------
!     GEM - Library of kernel routines for the GEM numerical atmospheric model
!     Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!     Environnement Canada
!     This library is free software; you can redistribute it and/or modify it 
!     under the terms of the GNU Lesser General Public License as published by
!     the Free Software Foundation, version 2.1 of the License. This library is
!     distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!     without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
!     PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
!     You should have received a copy of the GNU Lesser General Public License
!     along with this library; if not, write to the Free Software Foundation, Inc.,
!     59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
!----------------------------------LICENCE END ---------------------------------

!**   s/r vspngv_YY
!     

!     
      subroutine vspngv_YY ( F_champ, T_champ,Minx,Maxx,Miny,Maxy, pni, pnj )

      implicit none
#include <arch_specific.hf>
!     
      integer Minx,Maxx,Miny,Maxy,pni,pnj
      real F_champ(Minx:Maxx,Miny:Maxy,*)
      real T_champ(Minx:Maxx,Miny:Maxy,*)

!     
!     author   Abdessamad Qaddouri                        2012
!     
!     revision
!     v4-50 - Qaddouri&Girard          - Vectorial correction
!             A. Plante                  Optimization
!     
!     object
!     This routine is  5 points explicit  horizontal diffusion operator
!     the diffusion operator is an  approximation of The vector Laplacian
!     the diffusion using this operator conserves the angular momentum.
!     *
!     *
!     implicit
#include "glb_ld.cdk"
#include "vspng.cdk"
#include "opr.cdk"
#include "hzd.cdk"
#include "geomg.cdk"
#include "inuvl.cdk"
!     
      integer i,j,k,i0,in,j0,jn,istat
      real*8 two
      parameter(two=2.d0)
      real wk(l_ni,l_nj,Vspng_nk)
      integer ii,jj
      real*8 ruw(Minx:Maxx,Miny:Maxy,Vspng_nk)
      real*8 cst_8,cappa_8,aaa,bbb,ccc
      real*8, save, dimension(:,:,:), pointer :: stencils => null()
      logical, save :: done_L=.false.
!     
!---------------------------------------------------------------------
!     
      j0=1     + pil_s
      jn=l_njv - pil_n
      i0=1     + pil_w
      in=l_ni  - pil_e
!     
      call rpn_comm_xch_halo (F_champ,l_minx,l_maxx,l_miny,l_maxy,pni,pnj, &
      G_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!     interpole U on phi_grid
      call rpn_comm_xch_halo (T_champ,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj, &
      G_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!
      if(.not.done_L)then
         allocate(stencils(Minx:Maxx,Miny:Maxy,2:5),stat=istat)
         if(istat<0)call handle_error(-1,'vspngv_YY','Allocating stencils')
      endif
!
!     $omp parallel private (i,j,jj,ii,cst_8,stencil1,cappa_8, &
!     $opm                   aaa,bbb,ccc)
      if(.not.done_L)then
         allocate(stencils(Minx:Maxx,Miny:Maxy,5))
!$omp do
         do j=j0,jn
            jj=j+l_j0-1
            aaa=Hzd_yp0_8(G_nj+jj) / cos( G_yg_8 (jj) )**2
            bbb=cos(G_yg_8(jj))**2/( &
                 sin((G_yg_8(jj+1)+G_yg_8(jj  ))* 0.5)- &
                 sin((G_yg_8(jj  )+G_yg_8(jj-1))* 0.5))
            ccc=cos(G_yg_8(jj+1))**2/( &
                 sin((G_yg_8(jj+2)+G_yg_8(jj+1))* 0.5)- &
                 sin((G_yg_8(jj+1)+G_yg_8(jj  ))* 0.5))
            do i=i0,in
               ii=i+l_i0-1               
               stencils(i,j,2)=aaa/((G_xg_8(ii)   - G_xg_8(ii-1)))
               stencils(i,j,3)=aaa/((G_xg_8(ii+1) - G_xg_8(ii)  ))
               stencils(i,j,4)=Opr_opsxp0_8(G_ni+ii)*bbb
               stencils(i,j,5)=Opr_opsxp0_8(G_ni+ii)*ccc
               stencils(i,j,1)=-(stencils(i,j,2)+stencils(i,j,3)+ &
                                 stencils(i,j,4)+stencils(i,j,5))
            enddo
         enddo
!$omp end do
      endif
!     $omp do
      do j=j0,jn
         jj=j+l_j0-1
         cappa_8=  G_yg_8(jj+1)-(G_yg_8(jj+1)+G_yg_8(jj))*0.5
         do k=1,Vspng_nk
            do i=i0-1,in
               ii=i+l_i0-1
               ruw(i,j,k) =T_champ(i,j-1,k)*cappa_8+(1.-cappa_8)*T_champ(i,j  ,k)
               ! code (don't remove it) can be used if cubic interpolatin is needed
!               ruw(i,j,k) = inuvl_wyyv3_8(j,1) * T_champ(i,j-2,k) &
!                      + inuvl_wyyv3_8(j,2) * T_champ(i,j-1,k) &
!                      + inuvl_wyyv3_8(j,3) * T_champ(i,j  ,k) &
!                      + inuvl_wyyv3_8(j,4) * T_champ(i,j+1,k)
            enddo
         enddo
      enddo
!     $omp enddo
!
!     Aplly diffusion operator
!     $omp do
      do j=j0,jn
         jj=j+l_j0-1
         cst_8=-Geomg_invcyv2_8(j)+two
         do k=1,Vspng_nk
            do i=i0,in
               ii=i+l_i0-1
!     del2 explicit
               wk(i,j,k) =( &
                    stencils(i,j,1)*F_champ(i  ,j  ,k) + &
                    stencils(i,j,2)*F_champ(i-1,j  ,k) + &
                    stencils(i,j,5)*F_champ(i  ,j+1,k) + &
                    stencils(i,j,4)*F_champ(i  ,j-1,k) + &
                    stencils(i,j,3)*F_champ(i+1,j  ,k))/ &
                    (Opr_opsxp0_8(G_ni+ii)*Hzd_yp0_8(G_nj+jj))
!     vectorial correction
               wk(i,j,k)=wk(i,j,k)+F_champ(i  ,j  ,k)*cst_8&
                    +2*sin((G_yg_8(jj+1)+G_yg_8(jj))* 0.5)*Geomg_invcyv2_8(j)* &
                    (ruw(i,j,k)-ruw(i-1,j,k))/Opr_opsxp0_8(G_ni+ii)
            enddo
         enddo
      enddo
!     $omp end do
!     $omp do
      do j=j0,jn
         jj=j+l_j0-1 
         do k=1,Vspng_nk
            do i=i0,in
               ii=i+l_i0-1
               F_champ(i,j,k)=  F_champ(i,j,k)+Vspng_coef_8(k)*wk(i,j,k)               
            end do
         end do
      enddo
!     $omp enddo
!     $omp end parallel
!     
      done_L=.true.
!     
!----------------------------------------------------------------------
      return
      end

