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

!**S/P SERSET  -  INITIALISER UNE DES VARIABLES DES SERIES TEMPORELLES
!
      SUBROUTINE SERSET (NOM,VALEUR,N,IER)
      implicit none
#include <arch_specific.hf>
!
      CHARACTER *(*) NOM
      INTEGER N,IER
      INTEGER VALEUR(N)
!
!Author
!          M. Lepine  -  RFE model code revision project (Feb 87)
!
!Revision
! 001      B. Reid  (June 89)        -Zonal diagnostics
! 002      B. Bilodeau (December 89) -Update KOUNT
!                                     -Initialization of NPTRNCH
! 003      B. Bilodeau  (July 1991)- Adaptation to UNIX
! 004      B. Bilodeau  (August 1992)   - Add  S/R SERSETC
! 005      B. Bilodeau and G. Pellerin (Feb 94) -
!          No more reference to the zonal diagnostics package
! 006      N. Ek (Mar 95) - arbitrary interval of output of time series.
! 007      B. Bilodeau (Jan 96) - remove KA and create s/r SERSETM for KAM
! 008      B. Dugas (Apr 96) - Add option NSTAT
! 009      K. Winger (May 06) - Add option TSVER, TSMOYHR, SRWRI
!
!Object
!          to initialize a time-series variable
!
!Arguments
!
!          - Input -
! NOM      name of the variable to initialize
! VALEUR   table containing the values for initializing the variable
! N        number of values of initialize
!
!          - Output -
! IER      >0, no error, returned code is N
!          <0, error because N is greater than the dimension of the
!          variable. Returned code is maximum dimension for variable
!
!Notes
!          This routine contains ENTRY SERGET routine. It gets the
!          values for the variable.
!
!
!IMPLICITES
!
      include "series.cdk"
!
!MODULES
      EXTERNAL MOVLEV
!
!*
      REAL R
      INTEGER I
!
      IF (NOM .EQ. 'ISTAT') THEN
        CALL MOVLEV(VALEUR,IJSTAT,MIN(N,MXSTT))
        NSTAT = MIN(N,MXSTT)
        DO 10 I = 1,NSTAT
          IJSTAT(I,2) = IJSTAT(I,1) + (JSTAT(I) - 1) * NINJNK(1)
  10    CONTINUE
        IER = SIGN(MIN(N,MXSTT),MXSTT-N)
!
      ELSE IF (NOM .EQ. 'JSTAT') THEN
        CALL MOVLEV(VALEUR,JSTAT,MIN(N,MXSTT))
        NSTAT = MIN(N,MXSTT)
        DO 20 I = 1,NSTAT
          IJSTAT(I,2) = IJSTAT(I,1) + (JSTAT(I) - 1) * NINJNK(1)
  20    CONTINUE
        IER = SIGN(MIN(N,MXSTT),MXSTT-N)

      ELSE IF (NOM .EQ. 'STATNUM') THEN
        CALL MOVLEV(VALEUR,STATNUM,MIN(N,MXSTT))
        NSTAT = MIN(N,MXSTT)
        IER = SIGN(MIN(N,MXSTT),MXSTT-N)
!
      ELSE IF (NOM .EQ. 'ISTAT_G') THEN
        CALL MOVLEV(VALEUR,ISTAT_G,MIN(N,MXSTT))
        NSTAT_G = MIN(N,MXSTT)
        IER = SIGN(MIN(N,MXSTT),MXSTT-N)
!
      ELSE IF (NOM .EQ. 'JSTAT_G') THEN
        CALL MOVLEV(VALEUR,JSTAT_G,MIN(N,MXSTT))
        NSTAT_G = MIN(N,MXSTT)
        IER = SIGN(MIN(N,MXSTT),MXSTT-N)
!
      ELSE IF (NOM .EQ. 'HEURE') THEN
        CALL MOVLEV(VALEUR,R,1)
        HEURE = R
        IER = SIGN(MIN(N,1),1-N)
!
      ELSE IF (NOM .EQ. 'NOUTSER') THEN
        NOUTSER = VALEUR(1)
        IER = SIGN(MIN(N,1),1-N)
!
      ELSE IF (NOM .EQ. 'SERINT') THEN
        CALL MOVLEV(VALEUR,SERINT,1)
        IER = SIGN(MIN(N,1),1-N)
!
      ELSE IF (NOM .EQ. 'KOUNT') THEN
        CALL MOVLEV(VALEUR,KOUNT,1)
        IER = SIGN(MIN(N,1),1-N)
!
      ELSE IF (NOM .EQ. 'NSTAT')  THEN
        IF (NSTAT     .GT.  0     .AND. &
            VALEUR(1) .GT.  0     .AND. &
            VALEUR(1) .NE. NSTAT) THEN
          PRINT *,' NSTAT deja defini =',NSTAT
          PRINT *,' Nouvelle  valeur  =',VALEUR(1),' non utilise...'
          CALL QQEXIT( 1 )
        ELSE
          CALL MOVLEV(VALEUR,NSTAT,1)
          IER = SIGN(MIN(N,1),1-N)
        ENDIF
!
      ELSE IF (NOM .EQ. 'TSVER') THEN
        TSVER = VALEUR(1)
        IER = SIGN(MIN(N,1),1-N)
!
      ELSE IF (NOM .EQ. 'TSMOYHR') THEN
        TSMOYHR = VALEUR(1)
        IER = SIGN(MIN(N,1),1-N)
!
      ELSE IF (NOM .EQ. 'SRWRI') THEN
        SRWRI = VALEUR(1)
        IER = SIGN(MIN(N,1),1-N)

      ELSE IF (NOM .EQ. 'PAUSE') THEN
         if (VALEUR(1) == 0) then
            series_paused = .false.
         else
            series_paused = .true.
         endif
!
      ENDIF
!
      RETURN
!
!**S/P SERGET  -  OBTENIR LES VALEURS D'UNE VARIABLE DES SERIES TEMPOREL
!
      ENTRY SERGET(NOM,VALEUR,N,IER)
!
!Author
!          M. Lepine  -  RFE model code revision project (Feb 87)
!
!Object(SERSET)
!          to get values for the time-series variable
!
!Arguments
!
!          - Input -
! NOM      name of the variable to initialize
! VALEUR   table containing the values for initializing the variable
! N        number of values of initialize
!
!          - Output -
! IER      >0, no error, returned code is N
!          <0, error because N is greater than the dimension of the
!          variable. Returned code is maximum dimension for variable
!
!
!*
!
!
!     METTRE IER ET VALEUR A ZERO PAR DEFAUT
!     (POUR DETECTER OPTION INEXISTANTE
!      QUI SERAIT DEMANDEE)
      IER = 0
      VALEUR (1) = 0
!
      IF (NOM .EQ. 'ISTAT') THEN
        CALL MOVLEV(IJSTAT,VALEUR,MIN(N,MXSTT))
        IER = SIGN(NSTAT,N-NSTAT)
!
      ELSE IF (NOM .EQ. 'JSTAT') THEN
        CALL MOVLEV(JSTAT,VALEUR,MIN(N,MXSTT))
        IER = SIGN(NSTAT,N-NSTAT)
!
      ELSE IF (NOM .EQ. 'NINJNK') THEN
        CALL MOVLEV(NINJNK,VALEUR,MIN(N,3))
        IER = SIGN(3,N-3)
!
      ELSE IF (NOM .EQ. 'HEURE') THEN
        CALL MOVLEV(HEURE,VALEUR,1)
        IER = SIGN(1,N-1)
!
      ELSE IF (NOM .EQ. 'NOUTSER') THEN
        VALEUR(1) = NOUTSER
        IER = SIGN(1,N-1)
!
      ELSE IF (NOM .EQ. 'NSTAT') THEN
        VALEUR(1) = NSTAT
        IER = SIGN(1,N-1)
!
      ELSE IF (NOM .EQ. 'SERINT') THEN
        VALEUR(1) = SERINT
        IER = SIGN(1,N-1)
!
      ENDIF
!
      RETURN
      END
