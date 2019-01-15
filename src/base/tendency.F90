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
!**s/p tendency
!
subroutine tendency4 ( uplus0,vplus0,wplus0,tplus0,huplus0,qcplus0, &
     vbus,dbus,rcdt1,vsiz,dsiz,kount,ni,nk )
  use phy_options
  use phybus
  implicit none
#include <arch_specific.hf>
  integer                :: vsiz,dsiz,kount,ni,nk
  real, dimension(ni,nk) :: uplus0,vplus0,wplus0,tplus0,huplus0,qcplus0
  real, target           :: vbus(vsiz), dbus(dsiz)
  real                   :: rcdt1
  !
  !Author
  !          L. Spacek (Oct 2011)
  !
  !Revision
  !
  !Object
  !          Calculates tendencies in physics
  !
  !Arguments
  !
  !          - Input -
  ! dsiz     dimension of dbus
  ! vsiz     dimension of vbus
  ! ni       horizontal running length
  ! nk       vertical dimension
  ! rcdt1    1/cdt1
  !
  !          - Output -
  ! uplus0   initial value of dbus(uplus)
  ! vplus0   initial value of dbus(vplus)
  ! tplus0   initial value of dbus(tplus)
  ! huplus0  initial value of dbus(huplus)
  ! qcplus0  initial value of dbus(qcplus)
  !
  !          - input/output -
  ! dbus     dynamics input field
  ! vbus     physics tendencies and other output fields from the physics
  !
  !Implicities
  !
#include "ens.cdk"
  include "thermoconsts.inc"
  !
  integer                :: i,j,k,nik,ierget

  real, pointer, dimension(:,:) :: zhuphytd, zhuplus, zqcphytd, zqcplus, zqdifv, ztdifv, ztphytd, ztplus, zuphytd, zudifv, zuplus, zvphytd, zvdifv, zvplus, zwphytd, zwplus
  !-------------------------------------------------------------

  zhuphytd(1:ni,1:nk) => vbus( huphytd:)
  zhuplus (1:ni,1:nk) => dbus( huplus:)
  zqcphytd(1:ni,1:nk) => vbus( qcphytd:)
  zqcplus (1:ni,1:nk) => dbus( qcplus:)
  zqdifv  (1:ni,1:nk) => vbus( qdifv:)
  ztdifv  (1:ni,1:nk) => vbus( tdifv:)
  ztphytd (1:ni,1:nk) => vbus( tphytd:)
  ztplus  (1:ni,1:nk) => dbus( tplus:)
  zuphytd (1:ni,1:nk) => vbus( uphytd:)
  zudifv  (1:ni,1:nk) => vbus( udifv:)
  zuplus  (1:ni,1:nk) => dbus( uplus:)
  zvphytd (1:ni,1:nk) => vbus( vphytd:)
  zvdifv  (1:ni,1:nk) => vbus( vdifv:)
  zvplus  (1:ni,1:nk) => dbus( vplus:)

  do k=1,nk
  do i=1,ni
     zuphytd (i,k) = (zuplus (i,k) - uplus0 (i,k)) * rcdt1
     zvphytd (i,k) = (zvplus (i,k) - vplus0 (i,k)) * rcdt1
     ztphytd (i,k) = (ztplus (i,k) - tplus0 (i,k)) * rcdt1
     zhuphytd(i,k) = (zhuplus(i,k) - huplus0(i,k)) * rcdt1
     zqcphytd(i,k) = (zqcplus(i,k) - qcplus0(i,k)) * rcdt1
  enddo
  enddo
  if(diffuw)then
     zwphytd (1:ni,1:nk) => vbus( wphytd:)
     zwplus  (1:ni,1:nk) => dbus( wplus:)
     do k=1,nk
     do i=1,ni
        zwphytd(i,k)  = (zwplus(i,k) - wplus0(i,k)) * rcdt1
     enddo
     enddo
  endif

  do i=1,ni
     zuphytd (i,nk) = zudifv(i,nk)
     zvphytd (i,nk) = zvdifv(i,nk)
     ztphytd (i,nk) = ztdifv(i,nk)
     zhuphytd(i,nk) = zqdifv(i,nk)
  end do

  if (stochphy.and.kount.ge.1) then
     do k=1,nk
     do i=1,ni
        zuplus (i,k) = uplus0 (i,k)
        zvplus (i,k) = vplus0 (i,k)
        ztplus (i,k) = tplus0 (i,k)
!       zhuplus(i,k) = huplus0(i,k)
!       zqcplus(i,k) = qcplus0(i,k)
     enddo
     enddo
  endif
  !
  !-------------------------------------------------------------
  !
end subroutine tendency4
