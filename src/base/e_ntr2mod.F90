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

!**s/r e_ntr2mod - writes a small file to communicate info to the model
!
      subroutine e_ntr2mod
      implicit none
#include <arch_specific.hf>

!author michel desgagne - Spring 2015
!
!revision
! v4_80 - Desgagne M.   - initial version

#include "tr3d.cdk"
#include "ntr2mod.cdk"
#include "e_anal.cdk"
#include "ptopo.cdk"
#include "path.cdk"
#include "step.cdk"
#include <clib_interface_mu.hf>

      integer, external :: fnom

      character*1024 rootfn
      integer labfl,err
      namelist /ntr2mod_cfgs/ NTR_runstrt_S, NTR_horzint_L       , &
                      NTR_Tr3d_name_S,NTR_Tr3d_wload,NTR_Tr3d_hzd, &
                      NTR_Tr3d_mono,NTR_Tr3d_mass,NTR_Tr3d_ntr
!
!     ---------------------------------------------------------------
!
      err = 0

      if (Ptopo_myproc.eq.0) then
         NTR_runstrt_S  = Step_runstrt_S
         NTR_horzint_L  = (Anal_hav(1).ne.11)
         NTR_Tr3d_ntr   = Tr3d_ntr
         NTR_Tr3d_name_S= Tr3d_name_S
         NTR_Tr3d_wload = Tr3d_wload
         NTR_Tr3d_hzd   = Tr3d_hzd
         NTR_Tr3d_mono  = Tr3d_mono
         NTR_Tr3d_mass  = Tr3d_mass
         labfl=0
         rootfn = trim(Path_output_S)//'/GEMDM_input/'
         err    = clib_mkdir (trim(rootfn))
         err    = fnom(labfl,trim(rootfn)//'/Ntr2Mod.nml','SEQ',0)
         if (err.ge.0) then
            write (labfl  ,nml=ntr2mod_cfgs)
            close (labfl)
         endif
      endif

      call handle_error(err,'e_ntr2modn','Trying to open file labfl.bin')
!
!     ---------------------------------------------------------------
!
      return
      end

