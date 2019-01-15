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

!**S/P  METOX - METHANE OXIDATION
Subroutine metox2 (d,v,f,dsiz,vsiz,fsiz,ni,nk)
   use phy_options
   use phybus
   Implicit None
#include <arch_specific.hf>
      integer,intent(in)        :: ni,nk,dsiz,vsiz,fsiz
      real,target, intent(in)   :: v(vsiz),f(fsiz)
      real,target,intent(inout) :: d(dsiz)
!
!author
!         M. Charron (RPN): November 2005
!
!revisions
!
!  001      L. Spacek (oct 2011) - Complete remake
!
!object
!
!         Produces the specific humidity tendency due to methane
!         oxidation (based on ECMWF scheme)
!
!
!
!arguments
!
!          - input -
! ivar     variable  index
! iten     tendency  index
! ni       horizonal index
! nk       vertical  index
! v        volatile bus
! f        permanent bus
!
!          - input/output -
! d        dynamics bus

Include "thermoconsts.inc"

!  Real, Parameter          :: qq=4.25e-6
!  Real, Parameter          :: alpha1=4.25e-6
  real, pointer, dimension(:) :: psp(:)
  real, pointer               :: sigma(:,:)
  real, pointer               :: oxme(:,:)
  real, pointer               :: qqp(:,:)
  real, Dimension(ni,nk)      :: press,kmetox
  integer :: i,k
  real    :: qq,alpha1
  qq=4.25e-6
  alpha1=0.5431969

  if (.not.lmetox)Return

  psp  (1:ni)       => f( pmoins:)
  sigma(1:ni,1:nk)  => d( sigw:)
  qqp  (1:ni,1:nk)  => d( huplus:)
  oxme (1:ni,1:nk)  => v( qmetox:)

  press(:,:) = Spread( psp(:), dim=2, ncopies=nk )
  press(:,:) = sigma(:,1:nk)*press(:,:)

  where(press>=10000.)
      kmetox=0.
  elsewhere (press>50..and.press<10000.)
     kmetox=1.157407e-7/(1.+alpha1*(alog(press/50.))**4/(alog(10000./press)))
  elsewhere
     kmetox=1.157407e-7
  endwhere

  oxme=kmetox*(qq-qqp)

!
  Call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,huplus,qmetox,ni,nk-1)

End Subroutine metox2
