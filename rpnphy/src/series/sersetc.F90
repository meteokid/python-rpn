!---------------------------------- LICENCE BEGIN -------------------------------
! GEM - Library of kernel routines for the GEM numerical atmospheric model
! Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!                       Environnement Canada
! This library is free software; you can redistribute it and/or modify it 
! under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, version 2.1 of the License. This library is
! distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
! without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
! PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
! You should have received a copy of the GNU Lesser General Public License
! along with this library; if not, write to the Free Software Foundation, Inc.,
! 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
!---------------------------------- LICENCE END ---------------------------------

!**S/P SERSETC  -  INITIALISER LES CHAINES DE CARACTERE POUR LES SERIES TEMP.
      SUBROUTINE SERSETC (NOM,VALEUR,N,IER)
      implicit none
#include <arch_specific.hf>
      include "series.cdk"
      INTEGER N,IER,N2
      CHARACTER *(*) NOM
      CHARACTER *(*) VALEUR(N)
      CHARACTER*4 NOM_MAJUSCULE
!
!Author
!          B. Bilodeau  -  Adaptation to UNIX  (August 1992)
!
!Revision
! 001      B. Bilodeau and G. Pellerin (Feb 94) -
!          No more reference to the zonal diagnostics package
! 002      B. Bilodeau (Jan 06) - 4-character names and conversion
!                                 to upper case
!Object
!          to initialize the character strings for time-series
!
!Arguments
!
!          - Input -
! NOM      variable name to be initialized
! VALEUR   array containing the value to initialize the variable
! N        number of values to initialize
!
!          - Output -
! IER      error code:
!          IER > 0, no error and code returns N
!          IER < 0, N is larger than the dimension of the variable
!          and code returns the maximum dimension of the variable
!
!
!IMPLICITES
!
!
!*
      INTEGER I
!
      IF (NOM .EQ. 'SURFACE') THEN
        NSURF = MIN(N,MXSRF)
        IF (NSURF.GT.NVAR) THEN
           WRITE(6,'(A)') 'TOO MANY SURFACE VARIABLES FOR TIME-SERIES'
           WRITE(6,1000) 'MAXIMUM : ',NVAR, ' REQUESTED : ',NSURF
1000       FORMAT(1X,A12,I4,A12,I4)
           CALL QQEXIT(1)
        ENDIF
        DO 10 I=1,NSURF
           CALL LOW2UP (VALEUR(I) (1:4), NOM_MAJUSCULE)
           SURFACE(I,1) (1:4) = NOM_MAJUSCULE
10      CONTINUE
        IER = SIGN(MIN(N,MXSRF),MXSRF-N)
!
      ELSE IF (NOM .EQ. 'PROFILS') THEN
        NPROF = MIN(N,MXPRF)
        IF (NPROF.GT.NVAR) THEN
           WRITE(6,'(A)') 'TOO MANY PROFILE VARIABLES FOR TIME-SERIES'
           WRITE(6,1000) 'MAXIMUM : ',NVAR, ' REQUESTED : ',NPROF
           CALL QQEXIT(1)
        ENDIF
        DO 20 I=1,NPROF
           CALL LOW2UP (VALEUR(I) (1:4), NOM_MAJUSCULE)
           PROFILS(I,1) (1:4) = NOM_MAJUSCULE
20      CONTINUE
        IER = SIGN(MIN(N,MXPRF),MXPRF-N)
!
      ELSE IF (NOM .EQ. 'NAME') THEN
        NSTAT_G = MIN(N,MXSTT)
        DO I=1,NSTAT_G
           NAME(I) = VALEUR(I)
        ENDDO
        IER = SIGN(MIN(N,MXSTT),MXSTT-N)
!
      ENDIF
!
      RETURN
!
!**S/P SERGETC  -  OBTENIR LES VALEURS D'UNE VARIABLE DES SERIES TEMP.
!
      ENTRY SERGETC(NOM,VALEUR,N,IER)
!
!Author
!          B. Bilodeau  -  Adaptation to UNIX  (August 1992)
!
!Object
!          ENTRY SERGETC of SERSETC
!          to get the values for a time-series variable
!
!Arguments
!
!          - Input -
! NOM      variable name to be initialized
! VALEUR   array containing the value to initialize the variable
! N        number of values to initialize
!
!          - Output -
! IER      error code:
!          IER > 0, no error and code returns N
!          IER < 0, N is larger than the dimension of the variable
!          and code returns the maximum dimension of the variable
!
!*
!
!
!     INITIALISER IER ET VALEUR POUR DETECTER OPTION
!     INEXISTANTE QUI SERAIT DEMANDEE
      IER = 0
      VALEUR (1) = '       '
!
      IF (NOM .EQ. 'SURFACE') THEN
        DO 50 I=1,NSURF
           VALEUR(I) (1:4) = SURFACE(I,1) (1:4)
50      CONTINUE
        IER = SIGN(NSURF,N-NSURF)
!
      ELSE IF (NOM .EQ. 'PROFILS') THEN
        DO 60 I=1,NPROF
           VALEUR(I) (1:4) = PROFILS(I,1) (1:4)
60      CONTINUE
        IER = SIGN(NPROF,N-NPROF)

      ELSE IF (NOM .EQ. 'ALLVARS') THEN
        N2 = 0
        DO I=1,NSURF
           N2 = N2 + 1
           VALEUR(N2)(1:4) = SURFACE(I,1)(1:4)
        enddo
        DO I=1,NPROF
           if (.not.any(PROFILS(I,1)(1:4) == VALEUR(1:N2))) then
              N2 = N2 + 1
              VALEUR(N2)(1:4) = PROFILS(I,1)(1:4)
           endif
        enddo
        IER = N2

      ELSE IF (NOM .EQ. 'NAME') THEN
        DO I=1,NSTAT_G
           VALEUR(I) = NAME(I)
        ENDDO
        IER = SIGN(NSTAT_G,N-NSTAT_G)
!
      ENDIF
!
      RETURN
      END
