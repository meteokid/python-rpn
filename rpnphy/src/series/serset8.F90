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

!**S/P SERSET8  -  INITIALISER UNE VARIABLE A 64 BITS
!
      SUBROUTINE SERSET8 (NOM,VALEUR,N,IER)
      implicit none
#include <arch_specific.hf>
!
      CHARACTER *(*) NOM
      INTEGER N,IER
      REAL*8 VALEUR(N)
!
!Author
!          B. Bilodeau- From subroutine serset
!
!Revision
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
!
!*
      INTEGER I
!
      IF (NOM .EQ. 'HEURE') THEN
        HEURE = VALEUR(1)
        IER = SIGN(MIN(N,1),1-N)
      ENDIF
!
      RETURN
!
!**S/P SERGET8 -  OBTENIR LES VALEURS D'UNE VARIABLE DES SERIES TEMPORELLES
!
      ENTRY SERGET8(NOM,VALEUR,N,IER)
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
      IF (NOM .EQ. 'HEURE') THEN
        VALEUR(1) = HEURE
        IER = SIGN(1,N-1)
      ENDIF
!
      RETURN
      END

