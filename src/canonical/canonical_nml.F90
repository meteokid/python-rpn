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

!**s/r canonical_nml - Establish canonical configuration
 
      integer function canonical_nml (F_namelistf_S, F_unout, F_dcmip_L, F_wil_L)
      use canonical
      implicit none
#include <arch_specific.hf>

      character* (*) F_namelistf_S
      logical F_dcmip_L,F_wil_L
      integer F_unout

      integer, external ::  dcmip_nml, wil_nml
      logical dum_L
      integer err,err_dcmip,err_wil
!
!-------------------------------------------------------------------
!
      canonical_nml = -1
      F_dcmip_L= .false. ; F_wil_L= .false.

      if ((F_namelistf_S.eq.'print').or.(F_namelistf_S.eq.'PRINT')) then
         canonical_nml = 0
         if ( F_unout.ge.0) then
            if (Canonical_dcmip_L) err= dcmip_nml ('print',F_unout,dum_L)
            if (Canonical_williamson_L) &
                              err= wil_nml ('print',F_unout,dum_L)
         endif
         return
      endif

      err_dcmip= dcmip_nml (F_namelistf_S,F_unout,Canonical_dcmip_L     )
      err_wil  = wil_nml   (F_namelistf_S,F_unout,Canonical_williamson_L)

      if ((err_dcmip.eq.1).and.(err_wil.eq.1)) then
         canonical_nml = 1
         F_dcmip_L= Canonical_dcmip_L
         F_wil_L= Canonical_williamson_L
      endif
!
!-------------------------------------------------------------------
!
      return
      end
