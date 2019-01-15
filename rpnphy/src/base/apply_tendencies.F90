!-------------------------------------- LICENCE BEGIN ------------------------------------
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

subroutine apply_tendencies1(d,dsiz,v,vsiz,f,fsiz,ivar,iten,ni,nk)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object Linear combination of two arrays
   !@Arguments
   !          - input -
   ! ivar     variable  index
   ! iten     tendency  index
   ! ni       horizonal index
   ! nk       vertical  index
   ! busvol   volatile bus
   ! busper   permanent bus
   !
   !          - input/output -
   ! busdyn   dynamics bus

   integer,        intent(in)    :: dsiz,vsiz,fsiz,ivar,iten,ni,nk
   real   ,target, intent(in)    :: v(vsiz),f(fsiz)
   real   ,target, intent(inout) :: d(dsiz)

   !@Author L. Spacek (Oct 2011)
   !*

   integer :: i,k

   real, pointer, dimension(:)   :: ztdmask
   real, pointer, dimension(:,:) :: ziten, zivar

   ztdmask(1:ni)    => f(tdmask:)
   ziten(1:ni,1:nk) => v(iten:)
   zivar(1:ni,1:nk) => d(ivar:)
   do k=1,nk
      do i=1,ni
         zivar(i,k) = zivar(i,k) + ztdmask(i)*ziten(i,k)*delt
      enddo
   enddo

   return
end subroutine apply_tendencies1
