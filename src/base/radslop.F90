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
!*** S/P radslop
!*

      subroutine radslop (f, fsiz, v, vsiz, n,&
                          hz,julien,idatim,trnch,kount)
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>

      integer fsiz,vsiz, n,nk,nkp, kount,trnch
      integer idatim(14)
      
      real f(fsiz), v(vsiz), dire,difu,albe
      real julien

      real hz
      integer i

!Authors
!        J. Mailhot, A. Erfani, J. Toviessi
!
!
!Revisions
!
!
!Object
!      To add the effects of the sloping terrain to the radiation
!      computation.
! 
!
!Arguments
!
!          - Input/Output -
! f        field of permanent physics variables
! fsiz     dimension of f
!
!          - Input
! v        field of volatile physics variables
! vsiz     dimension of v
! n        horizontal dimension
! hz       Greenwich hour (0 to 24)
! julien   Julien days
! idatim   time coded in standard RPN format
! trnch    number of the slice
! kount    number of the timestep

      real bcos(n),bsin(n),stan(n),ssin(n),scos(n)
      real heurser
      integer ierget
      integer it
      
	   if (.not.radslope) then
         do i=0 , n-1 
            f(fluslop+i) = 0.0
         enddo
         return
      endif

      call suncos1(scos,ssin,stan,bsin,bcos,n,                              &
                   f(dlat),f(dlon),hz,julien,idatim,radslope)

      do 100 i=0 , n-1 

          dire= f(fsd0+i)*( f(c1slop+i)+ stan(i+1)*( f(c2slop+i)*bcos(i+1) + &
                f(c3slop+i)*bsin(i+1))) * f(vv1+i)
          dire= max(0.0,dire)

          difu= (f(fsf0+i)* f(c4slop+i)) * f(vv1+i)

          albe = ((f(fsf0+i) + f(fsd0+i))*f(c5slop+i) * v(ap+i) ) * f(vv1+i)

        f(fluslop+i) = dire + difu + albe

 100   continue      

       call serxst2(f(fluslop), 'fusl', trnch, n, 1, 0.0, 1.0, -1)

	   return
       end	
