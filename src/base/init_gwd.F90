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
!**s/p init_gwd
!
      subroutine init_gwd (e, esiz, f, fsiz, &
                                 kount, trnch, ni, nk)
!
      use phybus
      implicit none
#include <arch_specific.hf>
!
      integer esiz, fsiz, kount, ni, nk, trnch
      real, target :: e(esiz), f(fsiz)
!
!Author
!          L. Spacek (May 2013
!
!Revision
! 001      
!
!Object
!          To initialize arrays for gravity wave drag
!
! Arguments
!
!          - Input -
! e        entry bus
! esiz     dimension of v
! f        field for permanent physics variables
! fsiz     dimension of f
! kount    timestep number
! trnch    row number
! ni       horizontal dimension
! nk       vertical dimension
!
!*
!
#include "phyinput.cdk"

      integer i
      real land(ni)
      real, pointer, dimension(:) :: zdhdxen, zdhdyen, zdhdxdyen, zlhtgen, &
                                     zdhdx, zdhdy, zdhdxdy, zlhtg, zmg
!
      if (kount/=0) return
!
      zdhdxen  (1:ni) => e( dhdxen:)
      zdhdyen  (1:ni) => e( dhdyen:)
      zdhdxdyen(1:ni) => e( dhdxdyen:)
      zlhtgen  (1:ni) => e( lhtgen:)
      zdhdx    (1:ni) => f( dhdx:)
      zdhdy    (1:ni) => f( dhdy:)
      zdhdxdy  (1:ni) => f( dhdxdy:)
      zlhtg    (1:ni) => f( lhtg:)
      zmg      (1:ni) => f( mg:)

      if (any('dhdxen'==phyinread_list_s(1:phyinread_n)))   zdhdx   = zdhdxen
      if (any('dhdyen'==phyinread_list_s(1:phyinread_n)))   zdhdy   = zdhdyen
      if (any('dhdxdyen'==phyinread_list_s(1:phyinread_n))) zdhdxdy = zdhdxdyen
      if (any('lhtgen'==phyinread_list_s(1:phyinread_n)))   zlhtg   = zlhtgen
      land  = - abs( nint( zmg ) )
!
     call equivmount (land, f(lhtg), f(dhdx), f(dhdy), f(dhdxdy), &
                      ni, 1, ni, f(slope), f(xcent), f(mtdir))
!
    end subroutine init_gwd
