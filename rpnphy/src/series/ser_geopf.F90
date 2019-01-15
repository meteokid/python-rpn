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


subroutine ser_geopf2(BUSPER_3d, esp_busper, phydim_ni, phydim_nj, date, satuco)
   use phygetmetaplus_mod, only: phymetaplus, phygetmetaplus
   implicit none
#include <arch_specific.hf>
   !@object Prepares "first record" output for time series.
   !@arguments 
   logical :: satuco
   integer :: esp_busper, phydim_ni, phydim_nj, date(14)
   real    :: F_delt
   real    :: BUSPER_3d(esp_busper, phydim_nj)
   !@author Andre Methot - cmc - june 1994 v0_14
   !@revision
   ! v2_00 - Desgagne M.     - initial MPI version
   ! v2_20 - Lee V.          - extract geophysical fields for time-series 
   ! v2_20                     from physics permanent bus,not VMM variables
   ! v3_11 - A. Plante       - Adjust code for LAM time-series
   ! v3_20 - Winger K.       - correct time series handling in climate mode
   ! v3_30 - Winger K.       - Change serset to serset8 for HEURE
   ! v3_30 - Desgagne M.     - Remove Mem_phyncore_L
   !@description
   !      This subroutine is part of time serie's package
   !      initialisation. It extracts and produce output of constant
   !      fields to be used by the unwrapper.
   !	
   ! notes
   !     This code is done once per model's run.
   !
   !     The method used here is similar to SEF or RFE equivalent.
   !     The constraint here is to perform extractions and output
   !     of header and a hardcoded list of geophysical variables
   !     using the same calls as a real time serie's variable.
   !
   !     The user's given list of time serie's variables is then
   !     temporarly overwritten by the list of constant fields.
   !
   !     The constant fields are then loaded, and extracted.
   !
   !     Finally, the user's given list of time serie's variables is
   !     re-initialised.
   !
#include <rmnlib_basics.hf>
   include "thermoconsts.inc"
   include "series.cdk"

   character(len=SERG_STRING_LENGTH) ptgeonm(12), ptbidon
   character(len=3) :: bus
   integer mype, mycol, myrow
   integer pnsurf, i, j, m, pnerr, soit, lght, nrec, istat
   integer dlat,dlon,z0,mg,lhtg,alvis,snodp,twater,tsoil,glsea,wsoil
   real prcon,w1(phydim_ni),w2(phydim_ni),w3(phydim_ni)
   type(phymetaplus) :: metaplus
   !---------------------------------------------------------------

   call rpn_comm_mype(mype, mycol, myrow)

   P_serg_unf= 0
   if (mype.eq.Xst_master_pe) then
      P_serg_unf= 0
      pnerr = fnom(P_serg_unf, '../time_series.bin', 'SEQ+FTN+UNF', 0)
      nrec=0
600   read (P_serg_unf, end=700)
      nrec=nrec+1
      goto 600
700   backspace(P_serg_unf)
   endif
   call serset('NOUTSER', P_serg_unf, 1, pnerr)
   call RPN_COMM_bcast (nrec, 1, "MPI_INTEGER", Xst_master_pe, "grid",pnerr )

   if (nrec.gt.0) return

   !        -----------------------------------------------------------
   !C    1- skip this subroutine if in non-climate restart mode or 
   !        if no time series are requested
   !        -----------------------------------------------------------

   if (Xst_unout.gt.0) write(Xst_unout,1001)

   !        ---------------------------------------------------------------
   !C   3- Building of a list of variable names for geophysical fields
   !        ---------------------------------------------------------------

   pnsurf= 12
   if (pnsurf.gt.CNSRGEO) then
      if (Xst_unout.gt.0) write(Xst_unout,1002)
      Xst_nstat= 0
   endif

   ptgeonm( 1) = 'MA'
   ptgeonm( 2) = 'LA'
   ptgeonm( 3) = 'LO'
   ptgeonm( 4) = 'ZP'
   ptgeonm( 5) = 'MG'
   ptgeonm( 6) = 'LH'
   ptgeonm( 7) = 'AL'
   ptgeonm( 8) = 'SD'
   ptgeonm( 9) = 'TM'
   ptgeonm(10) = 'TP'
   ptgeonm(11) = 'GL'
   ptgeonm(12) = 'HS'

   !        ---------------------------------------------------------------
   !C    4- Temporarily over-writing the user time serie's variable list
   !        with a list of geophysical variables
   !        ---------------------------------------------------------------

   call sersetc('SURFACE', ptgeonm, pnsurf, pnerr)
   call sersetc('PROFILS', ptbidon,      0, pnerr)
   call serset ('KOUNT'  , 0      ,      1, pnerr)
   call serset8('HEURE'  , 0.d0   ,      1, pnerr)
   call serdbu()

   !        ---------------------------------------------------------------
   !C    5- Extract time-series values for geophysical variables
   !        ---------------------------------------------------------------

   prcon = 180./pi

   !#TODO: use metaplus%ptr instead of BUSPER_3d
   istat = phygetmetaplus(metaplus,'DLAT','V','P',F_quiet=.true.,F_shortmatch=.false.)   
   dlat  = metaplus%index
   istat = phygetmetaplus(metaplus,'DLON','V','P',F_quiet=.true.,F_shortmatch=.false.)
   dlon  = metaplus%index
   istat = phygetmetaplus(metaplus,'Z0','V','P',F_quiet=.true.,F_shortmatch=.false.)
   z0    = metaplus%index
   istat = phygetmetaplus(metaplus,'MG','V','P',F_quiet=.true.,F_shortmatch=.false.)
   mg    = metaplus%index
   istat = phygetmetaplus(metaplus,'LHTG','V','P',F_quiet=.true.,F_shortmatch=.false.)
   lhtg  = metaplus%index
   istat = phygetmetaplus(metaplus,'ALVIS','V','P',F_quiet=.true.,F_shortmatch=.false.)
   alvis = metaplus%index
   istat = phygetmetaplus(metaplus,'SNODP','V','P',F_quiet=.true.,F_shortmatch=.false.)
   snodp = metaplus%index
   istat = phygetmetaplus(metaplus,'TWATER','V','P',F_quiet=.true.,F_shortmatch=.false.)
   twater = metaplus%index
   istat = phygetmetaplus(metaplus,'TSOIL','V','P',F_quiet=.true.,F_shortmatch=.false.)
   tsoil = metaplus%index
   istat = phygetmetaplus(metaplus,'GLSEA','V','P',F_quiet=.true.,F_shortmatch=.false.)
   glsea = metaplus%index
   istat = phygetmetaplus(metaplus,'WSOIL','V','P',F_quiet=.true.,F_shortmatch=.false.)
   wsoil = metaplus%index

   w1 = 1.
   do j= 1, phydim_nj

      w2(1:phydim_ni) = BUSPER_3d (dlat:dlat+phydim_ni-1,j) * prcon
      do i= 1, phydim_ni
         w3(i) = BUSPER_3d (dlon+i-1,j) * prcon
         if (w3(i).lt.0) w3(i)=360.+w3(i)
      end do

      call serxst2(             w1(1), 'MA',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(             w2(1), 'LA',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(             w3(1), 'LO',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d   (z0,j), 'ZP',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d   (mg,j), 'MG',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d (lhtg,j), 'LH',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d(alvis,j), 'AL',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d(snodp,j), 'SD',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d(twater,j),'TM',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d(tsoil,j), 'TP',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d(glsea,j), 'GL',j, phydim_ni, 1, 0.0, 1.0, -1)
      call serxst2(BUSPER_3d(wsoil,j), 'HS',j, phydim_ni, 1, 0.0, 1.0, -1)

   end do

   call ser_out(.true., date, satuco)

   !        ---------------------------------------------------------------
   !C    7- Reset to extracting fields for the user time serie's variable list
   !        ---------------------------------------------------------------

   call sersetc('SURFACE', P_serg_srsrf_s, P_serg_srsrf, pnerr)
   call sersetc('PROFILS', P_serg_srprf_s, P_serg_srprf, pnerr)
   sers = 0. ; serp = 0.
   call serdbu

   if(Xst_unout.gt.0)then
      write(Xst_unout,*)'TIME SERIES VARIABLES REQUESTED BY USER :'
      write(Xst_unout,*)'NUMBER OF SURFACE VARIABLES=',P_serg_srsrf
      write(Xst_unout,*)'LISTE OF SURFACE VARIABLES :', &
           (trim(P_serg_srsrf_s(i))//' ',i=1,P_serg_srsrf)
      write(Xst_unout,*)'NUMBER OF PROFILE VARIABLES=',P_serg_srprf
      write(Xst_unout,*)'LISTE OF PROFILE VARIABLES :', &
           (trim(P_serg_srprf_s(i))//' ',i=1,P_serg_srprf)
   endif

1001 format( &
        /,'SER_GEOPF: INITIALISATION OF TIME SERIES PACKAGE', &
        /,'==============================================')
1002 format ('SER_GEOPF: CNSRGEO too small in series.cdk - NO SERIES')
   !---------------------------------------------------------------
   return
end subroutine ser_geopf2
