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

!**s/r from_ntr - reads a small communication file from GEMNTR
!
      integer function from_ntr ()
      implicit none
#include <arch_specific.hf>

!author michel desgagne - Spring 2015
!
!revision
! v4_80 - Desgagne M.   - initial version

#include "grd.cdk"
#include "bcsgrds.cdk"
#include "tr3d.cdk"
#include "ntr2mod.cdk"
#include "path.cdk"
#include "out3.cdk"
#include "lun.cdk"
#include "rstr.cdk"
#include "step.cdk"

      integer, external :: fnom,wkoffit

      character*512 fn
      integer  unf, ierr
      namelist /ntr2mod_cfgs/ NTR_runstrt_S, NTR_horzint_L       , &
                      NTR_Tr3d_name_S,NTR_Tr3d_wload,NTR_Tr3d_hzd, &
                      NTR_Tr3d_mono,NTR_Tr3d_mass,NTR_Tr3d_ntr
!
!     ---------------------------------------------------------------
!
      from_ntr= -1

      unf= 0
      fn = trim(Path_input_S)//'/MODEL_INPUT/Ntr2Mod.nml'

      if (Rstri_rstn_L) then
         from_ntr = 0	
      else
         if (wkoffit(fn).ge.-1) then
            if ( fnom (unf, fn, 'SEQ+OLD',ierr ) .eq. 0 ) then
               read (unf  ,nml=ntr2mod_cfgs)
               call fclos (unf)
               from_ntr= 0
            endif
         endif
      endif

      if (from_ntr .eq. 0) then
         Step_runstrt_S= NTR_runstrt_S
         Ana_horzint_L = NTR_horzint_L 
         Tr3d_ntr      = NTR_Tr3d_ntr
         Tr3d_name_S   = NTR_Tr3d_name_S
         Tr3d_wload    = NTR_Tr3d_wload 
         Tr3d_hzd      = NTR_Tr3d_hzd
         Tr3d_mono     = NTR_Tr3d_mono
         Tr3d_mass     = NTR_Tr3d_mass  
      else
         NTR_runstrt_S= 'NIL'
         Out3_date    = 0
         Tr3d_ntr     = 0
         Ana_horzint_L= .true.
         if (Grd_typ_S(1:1).eq.'L') then
            from_ntr = 0
         else
            if (Lun_out.gt.0) write (6,1005) trim(fn)
         endif
      endif

 1005 format (/' File ',a,' not available - ABORT -'/)
!
!     ---------------------------------------------------------------
!
      return
      end

