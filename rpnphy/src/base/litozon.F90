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

      subroutine litozon (F_file_S, F_myproc)
      use phy_options
      implicit none
#include <arch_specific.hf>
!
      character *(*) F_file_S
      integer F_myproc

!Author
!          B. Bilodeau (April 1994) - From lirozon
!
!Revision
!
! 001      M. Desgagne (Oct 98) - call back to rdradf_d (from dynamics)
! 002      M. Desgagne (Mar 08) - optional ozone file content
!
!Object
!          to read in the ozone climatology table
!
!Arguments
!
!          - Input -
! F_file_S  full path of the radiation table file
!
!Notes
!     1 -  The ozone climatology file provided by J.P.Blanchet
!          is based on Kita and Sumi 1986.
!     2 -  Interpolation to the day of the month is done and all
!          monthly climatological values are supposed to be valid
!          on the 15th. No interpolation is needed (done) when
!          the input jour is 15

#include "radiation.cdk"
#include "ozopnt.cdk"

      external rd_ozone
      real     total,ecoule
      integer  j,basem,destm,mois,jour,NLP,annee(12)
      DATA annee / 31,28,31,30,31,30,31,31,30,31,30,31 /
!
!-----------------------------------------------------------------
!
      jour = 15
      mois = date(2)

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

      if (basem.ne.destm) total = 1./annee(basem)
      
      call phyrdfile (F_file_S, rd_ozone, 'OZONE', F_myproc)

      NLP=NLACL*NPCL
      call pntozon()

      if (destm.ne.basem) then

!        interpoler pour le jour courant.
         DO J=1,NLP
            goz(J) =  gozon12(J,basem) + &
                     (gozon12(J,destm)-gozon12(J,basem))*total*ecoule
         ENDDO

      else

         DO J=1,NLP
            goz(J)= gozon12(J,destm)
         ENDDO

      endif
!
!-----------------------------------------------------------------
!
      RETURN
      END

