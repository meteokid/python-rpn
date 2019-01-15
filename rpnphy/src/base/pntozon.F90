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
!**S/P PNTOZON
!
      SUBROUTINE PNTOZON
      implicit none
#include <arch_specific.hf>
!
!
!Author
!          J.Mailhot RPN(September 1989)
!
!Revision
! 001      B. Bilodeau (April 1994) - New common block ozopnt
! 002      B. Bilodeau (Sept  1997) - Remove CDZPOT
!
!Object
!          to determine the position of pointers to the ozone table
!
!Arguments
!
!*
!
!
#include "ozopnt.cdk"
!
!  CES DIMENSIONS CORRESPONDENT AUX CHAMPS SUIVANTS...
!
!     REAL FOZON(NLACL,NPCL),
!    %     CLAT(NLACL),PREF(NPCL)
!
!
!  ...D'OU LES POINTEURS
!
      FOZON=1
      CLAT=FOZON+NLACL*NPCL
      PREF=CLAT+NLACL
!
      RETURN
      END
