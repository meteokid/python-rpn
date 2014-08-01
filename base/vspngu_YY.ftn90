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

!**   s/r vspngu_YY
!     

!     
      subroutine vspngu_YY ( F_champ, T_champ,Minx,Maxx,Miny,Maxy, pni, pnj )

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
!     
!     object
!     This routine is  5 points horizontal diffusion operator 
!     the diffusion operator is an  approximation the vector laplacian         
!     the diffusion using this operator conserves the angular momentum. 
!     
!     implicit
#include "glb_ld.cdk"
#include "vspng.cdk"
#include "opr.cdk"
#include "hzd.cdk"
#include "geomg.cdk"
#include "inuvl.cdk"
!     
      integer i,j,k,i0,j0,in,jn
      real*8 wk(Minx:Maxx,Miny:Maxy,Vspng_nk)
      integer ii,jj
      real*8 rvw(Minx:Maxx,Miny:Maxy,Vspng_nk)
      real*8 cappa_8,aaa,bbb,ccc
      real*8, save, dimension(:,:,:), pointer :: stencils => null()
      logical, save :: done_L=.false.
!     
!---------------------------------------------------------------------
!     
!     
      i0=1     + pil_w
      in=l_niu - pil_e
      j0=1     + pil_s
      jn=l_nj  - pil_n
!
      call rpn_comm_xch_halo (F_champ,l_minx,l_maxx,l_miny,l_maxy,pni,pnj, &
      G_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!
!     interpole V on phi_grid
      call rpn_comm_xch_halo (T_champ,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_njv, &
      G_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!
      if(.not.done_L)allocate(stencils(Minx:Maxx,Miny:Maxy,5))
!
!$omp parallel private (i,j,k,jj,ii,aaa,bbb,ccc,cappa_8)
      if(.not.done_L)then
!$omp do
         do j=j0,jn
            jj=j+l_j0-1            
            aaa=Opr_opsyp0_8(G_nj+jj) / cos( G_yg_8 (jj) )**2
            bbb=(sin  (G_yg_8(jj)) -sin(G_yg_8(jj-1))) /  &
                 (cos ((G_yg_8(jj)+G_yg_8(jj-1))*0.5)**2)
            ccc=(sin (G_yg_8(jj+1))-sin(G_yg_8(jj)))/ &
                 (cos((G_yg_8(jj+1)+G_yg_8(jj))*0.5)**2)
            do i=i0,in
               ii=i+l_i0-1
               stencils(i,j,2)=aaa/((G_xg_8(ii+1) - G_xg_8(ii-1))*0.5)
               stencils(i,j,3)=aaa/((G_xg_8(ii+2) - G_xg_8(ii))*0.5)
               stencils(i,j,4)= Hzd_xp0_8(G_ni+ii)/bbb
               stencils(i,j,5)=Hzd_xp0_8(G_ni+ii)/ccc
               stencils(i,j,1)=-(stencils(i,j,2)+stencils(i,j,3)+&
                    stencils(i,j,4)+stencils(i,j,5))
            enddo
         enddo
!$omp enddo
      endif
!
!$omp do
      do j=j0,jn
         jj=j+l_j0-1
         cappa_8= (G_yg_8(jj+1)+G_yg_8(jj))*0.5-G_yg_8(jj)
         do k=1,Vspng_nk
            do i=i0,in+1
               ii=i+l_i0-1
               rvw(i,j,k) =T_champ(i,j-1,k)*cappa_8+(1.-cappa_8)*T_champ(i,j  ,k)
! code (don't remove it) can be used if cubic interpolatin is needed
!               rvw(i,j,k) = inuvl_wyvy3_8(j,1) * T_champ(i,j-2,k) &
!                      + inuvl_wyvy3_8(j,2) * T_champ(i,j-1,k) &
!                      + inuvl_wyvy3_8(j,3) * T_champ(i,j  ,k) &
!                      + inuvl_wyvy3_8(j,4) * T_champ(i,j+1,k)
            enddo
         enddo
      enddo
!$omp enddo
!
!     Apply diffusion operator
!$omp do
      do j=j0,jn
         jj=j+l_j0-1
         aaa=-Geomg_invcy2_8(j)+2.d0
         bbb=2*geomg_sy_8(j)*Geomg_invcy2_8(j)
         do k=1,Vspng_nk
            do i=i0,in
               ii=i+l_i0-1
!     del2 explicit
               wk(i,j,k) =( &
                    stencils(i,j,1)*F_champ(i  ,j  ,k) + &
                    stencils(i,j,2)*F_champ(i-1,j  ,k) + &
                    stencils(i,j,3)*F_champ(i+1,j  ,k) + &
                    stencils(i,j,5)*F_champ(i  ,j+1,k) + &
                    stencils(i,j,4)*F_champ(i  ,j-1,k)) / &
                    (Opr_opsyp0_8(G_nj+jj)*Hzd_xp0_8(G_ni+ii))
               
!      vectorial correction and conservation AM constraint 
              wk(i,j,k)= wk(i,j,k)+F_champ(i ,j ,k)*aaa &
            -bbb*(rvw(i+1,j,k)-rvw(i,j,k))/Hzd_xp0_8(G_ni+ii)
            enddo
         enddo
      enddo
!$omp enddo
!$omp do         
      do j=j0,jn
         jj=j+l_j0-1
         do k=1,Vspng_nk
            do i=i0,in
               ii=i+l_i0-1
               F_champ(i,j,k)=F_champ(i,j,k)+ Vspng_coef_8(k)*wk(i,j,k)
            end do
         end do
      enddo
!$omp enddo
!$omp end parallel
!
      done_L=.true.
!
!----------------------------------------------------------------------
      return
      end

