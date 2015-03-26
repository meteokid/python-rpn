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

!**s/r e_dynfld - reads and interpolated dynamics 3d fields
!                 for timeframe 'date1'
!
      subroutine e_dynfld (date1,errcode) 
      implicit none
#include <arch_specific.hf>

      character*16 date1
      integer errcode

!AUTHOR  M. Desgagne    April 2002
!
!revision 
! v3_21 - Desgagne M. - dayfrac calc displaced
! v3_30 - Desgagne/Lee - new LAM I/O interface
! v3_31 - Lee V.      - bugfix: eliminate save_bmf key
! v4_03 - Lee/Desgagne - ISST
! v4_05 - Desgagne M. - implement tailjob launcher
! v4_05 - Lee V. - bugfix: save Pil_bcs_hollow_L after e_specanal/ac_posi
!
#include "e_fu.cdk"
#include "e_cdate.cdk"

      integer, external :: fstfrm,fclos
      integer err
!
!---------------------------------------------------------------------
!
      errcode = -1
      call datp2f ( datev, date1 )
      write (6,105) date1

      call e_specanal2  (datev, err)
      if (err.lt.0) return

      call e_open_files (datev)

      call e_intthm (err)
      if (err.lt.0) then
         write(6,*) 'E_DYNFLD: PROBLEM WITH E_INTTHM'
         goto 999
      endif

      call e_intwind (err)
      if (err.lt.0) then
         write(6,*) 'E_DYNFLD: PROBLEM WITH E_INWIND'
         goto 999
      endif

      errcode = 0
     
 999  err = fstfrm ( e_fu_anal )
      err = fclos  ( e_fu_anal )
    
 105  format (/43('#'),/,1X,'PROCESSING DATASET VALID: ',a16,/43('#'))
!
!---------------------------------------------------------------------
      return
      end
