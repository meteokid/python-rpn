!Environment Canada - Atmospheric Science and Technology License/Disclaimer, 
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms 
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer 
!version 3 or (at your option) any later version that should be found at: 
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html 
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software; 
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec), 
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------

      subroutine mapping2drivergrid
      implicit none

#include <arch_specific.hf>
#include "phygrd.cdk"

      interface
         function mapphy2mod(ix,jx,off_i,off_j,lcl_ni,lcl_nj,dim_ni)
         integer :: ix,jx,off_i,off_j,lcl_ni,lcl_nj,dim_ni
         integer, dimension(2) :: mapphy2mod
         end function mapphy2mod

         function mapmod2phy(ix,jx,lcl_ni,dim_ni)
         integer :: ix,jx,lcl_ni,dim_ni
         integer, dimension(2) :: mapmod2phy
         end function mapmod2phy
      end interface

      integer i,j,p_offi,p_offj,ij(2)
!
!     ---------------------------------------------------------------
!
      allocate ( ijdrv_mod(2,phydim_ni,phydim_nj), &
                 ijdrv_phy(2,phydim_ni,phydim_nj), &
                 ijphy(2,phy_lcl_ni,phy_lcl_nj) )

      p_offi = phy_lcl_i0 - 1
      p_offj = phy_lcl_j0 - 1

      do j= 1, phydim_nj
         do i= 1, phydim_ni
            ij = mapphy2mod(i,j,p_offi,p_offj,phy_lcl_ni,phy_lcl_nj,phydim_ni)
            ijdrv_mod(1:2,i,j) = ij(1:2)
         end do
      end do

      p_offi = 0
      p_offj = 0

      do j= 1, phydim_nj
         do i= 1, phydim_ni
            ij = mapphy2mod(i,j,p_offi,p_offj,phy_lcl_ni,phy_lcl_nj,phydim_ni)
            ijdrv_phy(1:2,i,j) = ij(1:2)
         end do
      end do

      do j= 1, phy_lcl_nj
         do i= 1, phy_lcl_ni
            ij = mapmod2phy(i,j,phy_lcl_ni,phydim_ni)
            ijphy(1:2,i,j) = ij(1:2)
         end do
      end do
!
!     ---------------------------------------------------------------
!
      return
      end subroutine mapping2drivergrid
