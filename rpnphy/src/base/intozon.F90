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
!**s/p intozon
!
      subroutine intozon (jour,mois,prout)
!
      implicit none
#include <arch_specific.hf>
!
      logical prout
      integer jour,mois
!
!Author
!          B. Dugas (Winter 2001) - From litozon2
!
!Revision
!
!Object
!          produces a new ozone field valid at jour/mois from
!          the previously read climatology table gozon12.
!          Allows bit-reproducibility for time integrations
!          running a multiple of 1 day per "clone".
!
!Arguments
!
!          - Input -
! mois     month of ozone record
! jour     day of the month
!
!Notes
!          Monthly climatological values are supposed to be valid
!          on the 15th. No interpolation is needed (done) when
!          the input jour is 15
!
#include "radiation.cdk"
#include "ozopnt.cdk"
!
      real      total,ecoule
      integer   basem,destm,courm
      integer   i,J,K,NLP,annee(12)
!
      DATA annee / 31,28,31,30,31,30,31,31,30,31,30,31 /
!
      NLP=NLACL*NPCL
!
!     doit-on interpoler ?
      if (jour.lt.15) then
         if (mois.eq.1) then
            destm = 1
            basem = 12
         else
            destm = mois
            basem = destm-1
         endif
         ecoule   = jour+annee(basem)-15
      else if (jour.gt.15) then
         if (mois.eq.12) then
            basem = 12
            destm = 1
         else
            basem = mois
            destm = basem+1
         endif
         ecoule   = jour-15
      else
         basem    = mois
         destm    = basem
      endif
!
      if (destm.ne.basem) then
!
         total = 1./annee(basem)

!        interpoler pour le jour courant.
         DO J=1,NLP
            goz(J) =  gozon12(J,basem) + &
                     (gozon12(J,destm)-gozon12(J,basem))*total*ecoule
         ENDDO
!
      else
!
         DO J=1,NLP
            goz(J)= gozon12(J,destm)
         ENDDO
!
      endif
!
      if (prout) write(6,1120) jour,mois
!
      RETURN
 1120 FORMAT(/' INTOZON: OZONE INTERPOLATED TO DAY ',I2,', MONTH ',I2)
      END
!
