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
!** S/P CALCZ0
      subroutine calcz0(mg,z0,z1,z2,z3,z4,uu,vv,ni)
!
      implicit none
#include <arch_specific.hf>
      integer ni,i,j
      real z1(ni),z2(ni),z3(ni),z4(ni),z0(ni),mg(ni)
      real uu(ni),vv(ni),zz,newz0
      real theta, otheta
!
!Author
!       V. Lee RPN (September 1995)
!
!Revision
!
!Object
!       CALCZ0 is a routine which recalculates the
!       roughness length Z0 using the wind components and
!       the directional roughess for each grid point.
!
!Arguments
!
!       - Output -
! Z0      calculated roughness using UU and VV
!
!       - Input -
! Z1     +x direction roughness (for westerly  winds)
! Z2     -x direction roughness (for easterly  winds)
! Z3     +y direction roughness (for southerly winds)
! Z4     -y direction roughness (for northerly winds)
! UU      zonal      component of the wind
! VV      meridional component of the wind
! NI      dimension of the arrays
!*
!

include "thermoconsts.inc"
!
      do 1 i=1,ni
!
      if (mg(i).ge.0.5) then

      newz0 = 0.0
      zz = sqrt( uu(i)**2 + vv(i)**2)
      if (zz .gt. 0.0) then
      theta = acos(abs(uu(i)/zz))
      otheta = PI/2.0 - theta

!     wind is flowing from west and/or south
        if (uu(i).ge.0.0.and.vv(i).ge.0.0) then
           newz0 = (otheta*z1(i) + theta*z3(i))*2.0/PI
!          newz0 = z1(i)*(uu(i)/zz)**2 + z3(i)*(vv(i)/zz)**2
        endif
!     wind is flowing from east and/or south
        if (uu(i).le.0.0.and.vv(i).ge.0.0) then
           newz0 = (otheta*z2(i) + theta*z3(i))*2.0/PI
!          newz0 = z2(i)*(uu(i)/zz)**2 + z3(i)*(vv(i)/zz)**2
        endif
!     wind is flowing from west and/or north
        if (uu(i).ge.0.0.and.vv(i).le.0.0) then
           newz0 = (otheta*z1(i) + theta*z4(i))*2.0/PI
!          newz0 = z1(i)*(uu(i)/zz)**2 + z4(i)*(vv(i)/zz)**2
        endif
!     wind is flowing from east and/or north
        if (uu(i).le.0.0.and.vv(i).le.0.0) then
           newz0 = (otheta*z2(i) + theta*z4(i))*2.0/PI
!          newz0 = z2(i)*(uu(i)/zz)**2 + z4(i)*(vv(i)/zz)**2
        endif

      endif

            if (newz0 .gt. 0.0) then
            z0(i) = newz0
            else
            z0(i) = (z1(i)+z2(i)+z3(i)+z4(i))/4.0
            endif

            z0(i) = MAX(z0(i),exp(-6.908))
      endif
!
1     continue

!
      return
      end
