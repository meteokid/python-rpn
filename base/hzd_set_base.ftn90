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

!**s/r hzd_set_base - Compute common diffusion operator
!
      subroutine hzd_set_base
      implicit none
#include <arch_specific.hf>

!author    
!     Michel Desgagne  -- fall 2013
!
!revision
! v4_70 - Desgagne M.       - initial version

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "hzd.cdk"

      integer i,j,dimx,dimy,Gni,Gnj
      real*8, dimension(:) , allocatable :: wk1_8,wk2_8
      real*8, parameter :: ZERO_8= 0.d0, ONE_8=1.d0, HALF_8=0.5d0
!
!     ---------------------------------------------------------------

      Gni = G_ni-Lam_pil_w-Lam_pil_e
      Gnj = G_nj-Lam_pil_s-Lam_pil_n

      dimx = 3*G_ni*2
      dimy = 3*G_nj*2

      allocate ( Hzd_xp0_8(dimx), Hzd_yp0_8  (dimy), Hzd_xp2_8(dimx), &
                 Hzd_yp2_8(dimy), Hzd_yp2su_8(dimy), Hzd_h0_8(G_nj*2))

      Hzd_xp0_8= ZERO_8
      Hzd_xp2_8= ZERO_8
      Hzd_yp0_8= ZERO_8
      Hzd_yp2_8= ZERO_8
      Hzd_h0_8 = ONE_8
      if (Hzd_difva_L) then
         do i = 1, G_nj
            Hzd_h0_8(i)  = ONE_8 / (ONE_8 + 2*((cos(G_yg_8(i)))**2))
         enddo
      endif

      allocate ( wk1_8(Gni), wk2_8(Gni*3) )

      do i = 1+Lam_pil_w, G_ni-Lam_pil_e
         Hzd_xp0_8(G_ni+i)  = G_xg_8(i+1) - G_xg_8(i)
         wk1_8(i-Lam_pil_w) = (G_xg_8(i+2)-G_xg_8(i)) * HALF_8
      end do

      call set_ops8 (wk2_8,wk1_8,ONE_8,G_periodx,Gni,Gni,1)
 
      do i=1,Gni
         Hzd_xp2_8(       i+Lam_pil_w)= wk2_8(      i)
         Hzd_xp2_8(G_ni  +i+Lam_pil_w)= wk2_8(Gni  +i)
         Hzd_xp2_8(G_ni*2+i+Lam_pil_w)= wk2_8(Gni*2+i)
      enddo

      deallocate (wk1_8,wk2_8)

      do j = 1+Lam_pil_s, G_nj-Lam_pil_n
         Hzd_yp0_8(G_nj+j) = sin(G_yg_8(j+1))-sin(G_yg_8(j))
      end do

      j= 1+Lam_pil_s
      Hzd_yp2_8(2*G_nj+j)= ((cos(G_yg_8(j+1))**2)*Hzd_h0_8(j+1))/( &
           sin((G_yg_8(j+2)+G_yg_8(j+1  ))* HALF_8)- &
           sin((G_yg_8(j  )+G_yg_8(j+1))* HALF_8))
      Hzd_yp2_8(G_nj+j) =-Hzd_yp2_8(2*G_nj+j)

      do j = 2+Lam_pil_s, G_njv-1-Lam_pil_n
         Hzd_yp2_8(2*G_nj+j)= ((cos(G_yg_8(j+1))**2)*Hzd_h0_8(j+1))/( &
              sin((G_yg_8(j+2)+G_yg_8(j+1))* HALF_8)- &
              sin((G_yg_8(j+1)+G_yg_8(j  ))* HALF_8))
         Hzd_yp2_8(j) = ((cos(G_yg_8(j))**2)*Hzd_h0_8(j-1))/( &
              sin((G_yg_8(j+1)+G_yg_8(j  ))* HALF_8)- &
              sin((G_yg_8(j  )+G_yg_8(j-1))* HALF_8))
         Hzd_yp2_8(G_nj+j) = - (Hzd_yp2_8(j) + Hzd_yp2_8(2*G_nj+j))
      enddo

      j= G_njv-Lam_pil_n
      Hzd_yp2_8(j) = Hzd_h0_8(j-1)*Hzd_yp2_8(2*G_nj+j-1)
      Hzd_yp2_8(G_nj+j) = - (Hzd_yp2_8(j) + Hzd_yp2_8(2*G_nj+j))

      if (Hzd_difva_L) then
         Hzd_yp2su_8= ZERO_8
         allocate ( wk1_8(Gnj), wk2_8(Gnj*3) )
         do j = 1+Lam_pil_s, G_nj-1-Lam_pil_n
            wk1_8(j-Lam_pil_s) = (sin (G_yg_8(j+1))-sin(G_yg_8(j))) / &
                 (cos ((G_yg_8(j+1)+G_yg_8(j))*HALF_8)**2)
         end do
         call hzd_set_ops8(wk2_8,wk1_8,ZERO_8,G_periody,Gnj,Gnj,1,Hzd_h0_8)
         do j=1,Gnj
            Hzd_yp2su_8(j+Lam_pil_s)=wk2_8(j)
            Hzd_yp2su_8(G_nj+j+Lam_pil_s)=wk2_8(Gnj+j)
            Hzd_yp2su_8(G_nj*2+j+Lam_pil_s)=wk2_8(Gnj*2+j)
         enddo
         deallocate (wk1_8, wk2_8 )
      endif
!
!     ---------------------------------------------------------------
      return
      end
