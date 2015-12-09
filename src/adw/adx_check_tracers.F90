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

!**s/p adx_check_tracers: Check if extended advection operations required 

      subroutine adx_check_tracers ()

      implicit none

#include <arch_specific.hf>

      !@author Monique Tanguay

      !@revisions
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation

#include "tr3d.cdk"
#include "adx_tracers.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "schm.cdk"
#include "lun.cdk"

      !---------------------------------------------------------------------
      integer n
      logical qw_L
      !---------------------------------------------------------------------

      adx_flux_L = .FALSE.
      adx_core_L = .FALSE.

      if (Schm_psadj_L.and.G_lam.and..not.Grd_yinyang_L) adx_flux_L = .TRUE. !PSADJ (FLUX) 

      do n=1,Tr3d_ntr

         qw_L= Tr3d_wload(n) .or. Tr3d_name_S(n)(1:2).eq.'HU'

         if (qw_L) cycle

         if (G_lam.and..not.Grd_yinyang_L) then !LAM

            if (Tr3d_mass(n)==1) adx_flux_L    = .true. !BC (FLUX)   

            if (Tr3d_mass(n)/=0) adx_core_L = .true. 
            if (Tr3d_mono(n)/=0) adx_core_L = .true. 

         endif

      end do

      adx_extension_L = adx_flux_L

      if (adx_extension_L.and.Lun_out>0) then
         write(Lun_out,*) ''
         write(Lun_out,*) 'ADX_CHECK_EXT: EXTENDED ADVECTION OPERATIONS REQUIRED'
         write(Lun_out,*) ''
      endif

      !---------------------------------------------------------------------

      return
end subroutine adx_check_tracers
