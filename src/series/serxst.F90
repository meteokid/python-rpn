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
!**S/P  SERXST

!/@*
function series_isstep(F_varname_S) result(F_istat_L)
   implicit none
   !@objective 
   !@arguments
   character(len=*), intent(in) :: F_varname_S
   !@return
   logical :: F_istat_L
   !*@/
#include "series.cdk"
   logical, external :: series_isvar
   character(len=32) :: varname_S
   !---------------------------------------------------------------
   F_istat_L = .false.
   if (series_paused .or. .not.(initok .and. P_serg_srsus_L)) return
   if (kount /= 1 .and. mod(kount, serint) /= 0)  return
   F_istat_L = .true.
   if (F_varname_S /= ' ') F_istat_L = series_isvar(F_varname_S)
   !---------------------------------------------------------------
   return
end function series_isstep


!/@*
function series_isvar(F_varname_S) result(F_istat_L)
   implicit none
   !@objective 
   !@arguments
   character(len=*), intent(in) :: F_varname_S
   !@return
   logical :: F_istat_L
   !*@/
#include <clib_interface_mu.hf>
#include "series.cdk"
   character(len=32) :: varname_S
   integer :: istat
   !---------------------------------------------------------------
   varname_S = F_varname_S
   istat = clib_toupper(varname_S)
   F_istat_L = .false.
   if (any(surface(1:nsurf,1) == varname_S) .or. &
        any(profils(1:nprof,1) == varname_S)) &
        F_istat_L = .true.
   !---------------------------------------------------------------
   return
end function series_isvar


      SUBROUTINE SERXST2( F , NOM , J , N , nk0 , FACS , FACF , ORD )
!
      implicit none
#include <arch_specific.hf>
      CHARACTER *(*) NOM
      INTEGER J, N, nk0, ORD
      REAL F(N,*),FACS,FACF
      CHARACTER*4 NOM_MAJUS
!
!Author
!          R. Benoit (RPN 1984)
!
!Revision
! 001      J. Cote RPN(January 1985)
!                - Recoding compatible SEF/RFE version
!                - Documentation
! 002      M. Lepine  -  RFE model code revision project (Feb 87)
! 002      M. Lepine  -  Ensuring that the code is re-entrant (Oct 87)
! 003      R. Benoit  -  Extraction by levels for the PROFILS
! 004      B. Reid  (June 89) - Zonal diagnostics
! 005      B. Bilodeau (Mar 91) - Eliminate the entrance point
!                VSERXST and the call to ZONXST.
! 006      B. Bilodeau  (July 1991)- Adaptation to UNIX
! 007      N. Ek (Mar 1995) - output only every SERINT time-steps
! 008      B. Bilodeau (Nov 1995) - KAM
! 009      B. Bilodeau and M. Desgagne (March 2001) - Build lists
!            surface(m,2) and profils(m,2) even if nstat=0 because in
!            MPI mode, processor 0 needs full lists in call to serwrit2
! 010      B. Bilodeau (Jan 2006) - Variable NOM converted in upper case
!
!Object
!          to extract variables and perform calculations for time-series
!
!Arguments
!
!          - Input -
! F        field containing the variable to extract
! NOM      name of variable to extract
! J        latitude of extraction, all stations if J=0
! N        horizontal dimension of extracted fields
! FACS     the multiplying factor on the time-series
! FACF     the F multiplying factor before extraction
! ORD      =0 if F is scalar
!          =1 if F is an independent horizontal vector
!          >1 if F is a dependent horizontal vector
!          <0 if F is a horizontal vector containing the K level of a
!          profile
!          if (ORD=0 or 1, and the name is a surface variable,
!          FACF is not used and F(1,1) is used.
!
!Notes
!          See SERDBU for more information. SERDBU must have
!          been previously called.
!
!IMPLICITES
!
#include <msg.h>
#include "series.cdk"
!
!MODULE
!
!*
      integer, parameter :: ORD_SKIP = -9
      integer, parameter :: ORD_ALL  = -1
      INTEGER K,L,M,NK,I,IJ,LPREM,LDERN,ord1
!

      IF (ORD.EQ.99   ) RETURN
      IF (.NOT. INITOK) RETURN
      IF (series_paused) RETURN
!
      IF ( (KOUNT.NE.1) .AND. (MOD(KOUNT,SERINT) .NE. 0) )  RETURN
!
      CALL LOW2UP (NOM, NOM_MAJUS)
!
      IF (J.EQ.0) THEN
!
!        TOUTES LES STATIONS
!
         LPREM = 1
         LDERN = NSTAT
!
      ELSE
!
!        LES STATIONS A LA "LATITUDE" J
!
!        PREMIERE STATION
!
         LPREM = 0
         DO L=1,NSTAT
            IF (J.EQ.JSTAT(L)) GO TO 2
         ENDDO
!
!        PAS DE STATION
!
         LDERN = -1
         GO TO 5
!
!        DERNIERE STATION
!
    2    LPREM = L
         LDERN = LPREM
         DO L=LPREM+1,NSTAT
            IF (J.NE.JSTAT(L)) GO TO 4
            LDERN = L
         ENDDO
    4    CONTINUE
      ENDIF
!
!
                  I = 1
      IF (J.EQ.0) I = 2
!
!     CHERCHE "NOM_MAJUS" DANS LES VARIABLES DE SURFACE
!
      ord1 = ord
      if (ord == ORD_SKIP) ord1 = ORD_ALL

    5 DO M=1,NSURF
         IF (NOM_MAJUS.EQ.SURFACE(M,1)) GO TO 7
      ENDDO
!
!     CHERCHE "NOM_MAJUS" DANS LES VARIABLES DE PROFILS
!
      DO M=1,NPROF
         IF (NOM_MAJUS.EQ.PROFILS(M,1)) then
!!$            if (nk0 < NINJNK(3)) then
!!$               call msg(MSG_INFOPLUS,'(serxst) skipped -- not enough levels, profile request for var: '//trim(NOM_MAJUS))
!!$               return
!!$            else
               GO TO 12
!!$            endif
         endif
      ENDDO
!
!     "NOM_MAJUS" N EST PAS REQUIS SUR LE FICHIER DE SERIES
!
      RETURN
!
!
!     OPERATIONS SUR UNE VARIABLE DE SURFACE
!
    7 continue
!!$      print *,'(serxst s3) '//trim(NOM_MAJUS),kount,M,ord
      DO L=LPREM,LDERN
         if (lastout_surf(statnum(L),M) >= kount) then
!!$            print *,'(serxst s2) '//trim(NOM_MAJUS),kount,lastout_prof(statnum(L),M),statnum(L),M,ord
            if (ord == ORD_SKIP) then
               call msg(MSG_INFOPLUS,'(serxst) skipped series, calles twice in same step for var: '//trim(NOM_MAJUS))
               return
            else
               call msg(MSG_INFOPLUS,'(serxst) series called twice in same step for var: '//trim(NOM_MAJUS))
            endif
         else
            lastout_surf(statnum(L),M) = kount
         endif
      enddo

      SURFACE(M,2) = NOM_MAJUS
!
      IF (LPREM.EQ.0) RETURN
!
      IF (ORD1.EQ.0 .OR. ORD1.EQ.1 ) THEN
!
         DO L=LPREM,LDERN
            SERS(statnum(L),M) = FACS * SERS(statnum(L),M) + F(1,1)
         ENDDO
!
      ELSE
!
         DO L=LPREM,LDERN
!!$            print *,'(serxst s1) '//trim(NOM_MAJUS),kount,lastout_prof(statnum(L),M),statnum(L),M,ord
          SERS(statnum(L),M) = FACS * SERS(statnum(L),M) &
                                 + FACF * F( IJSTAT(L,I) , 1 )
         ENDDO
!
      ENDIF
!
      RETURN
!
!
!
!     OPERATIONS SUR UNE VARIABLE DE PROFIL
!
   12 continue
!!$      print *,'(serxst p3) '//trim(NOM_MAJUS),kount,M,ord
      DO L=LPREM,LDERN
         if (lastout_prof(statnum(L),M) >= kount) then
!!$            print *,'(serxst p2) '//trim(NOM_MAJUS),kount,lastout_prof(statnum(L),M),statnum(L),M,ord
            if (ord == ORD_SKIP) then
               call msg(MSG_INFOPLUS,'(serxst) skipped series, called twice in same step for var: '//trim(NOM_MAJUS))
               return
            else
               call msg(MSG_INFOPLUS,'(serxst) series called twice in same step for var: '//trim(NOM_MAJUS))
            endif
         else
            lastout_prof(statnum(L),M) = kount
         endif
      enddo

      PROFILS(M,2) = NOM_MAJUS

      NK = MIN( NINJNK(3), nk0 )
      IF (NK.GT.MXNVO) THEN
          IF (HEURE.EQ.0.0) &
              WRITE (6,'(1X,A2,A,I3,A,I3,A)') &
              NOM_MAJUS,' NK = ',NK,' > MXNVO = ',MXNVO,' DANS SERXST'
          RETURN
      ENDIF
!
      IF (LPREM.EQ.0) RETURN
!
      IF (ORD1.EQ.0) THEN
!
         DO L=LPREM,LDERN
            DO K=1,NK
               SERP(K,statnum(L),M) = FACS * SERP(K,statnum(L),M) + F(1,1)
            ENDDO
         ENDDO
         IF (NK.EQ.NINJNK(3)-1) THEN
            DO L=LPREM,LDERN
               SERP(NK+1,statnum(L),M) = SERP(NK,statnum(L),M)
            ENDDO
         ENDIF
!
      ELSE IF (ORD1.EQ.1) THEN
!
         DO L=LPREM,LDERN
            DO K=1,NK
            SERP(K,statnum(L),M) = FACS * SERP(K,statnum(L),M) &
                                 + FACF * F(1,K)
            ENDDO
         ENDDO
         IF (NK.EQ.NINJNK(3)-1) THEN
            DO  L=LPREM,LDERN
               SERP(NK+1,statnum(L),M) = SERP(NK,statnum(L),M)
            ENDDO
         ENDIF
!
      ELSE IF (ORD1.LT.ORD_ALL) THEN
         K = -ORD1-1
         IF (K.GT.NK) THEN
             WRITE(6,'(1X,A,I3)') 'NIVEAU A EXTRAIRE INVALIDE DANS SERXST ',K
             RETURN
         ENDIF
         DO L=LPREM,LDERN
            IJ = IJSTAT(L,I)
            SERP(K,statnum(L),M) = FACS * SERP(K,statnum(L),M) &
                                 + FACF * F(IJ,1)
         ENDDO
      ELSE
!
         DO L=LPREM,LDERN
            IJ = IJSTAT(L,I)
            DO K=1,NK
               SERP(K,statnum(L),M) = FACS * SERP(K,statnum(L),M) &
                                    + FACF * F ( IJ , K )
            ENDDO
         ENDDO
!
!!$            print *,'(serxst p1) '//trim(NOM_MAJUS),kount,lastout_prof(statnum(L),M),statnum(L),M,ord
         IF (NK.EQ.NINJNK(3)-1) THEN
            DO L=LPREM,LDERN
               IJ = IJSTAT(L,I)
               SERP(NK+1,statnum(L),M) = SERP(NK,statnum(L),M)
            ENDDO
         ENDIF
!
      ENDIF
!
      RETURN
!
      END
