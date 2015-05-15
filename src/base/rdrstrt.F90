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

!**s/r rdrstrt - Read the restart file

      subroutine rdrstrt ()
      implicit none
#include <arch_specific.hf>

!author
!     M. Desgagne - Mars 2000
!
!revision

#include "lun.cdk"
#include "init.cdk"
#include "step.cdk"
#include "lctl.cdk"
#include "tr3d.cdk"
#include "ntr2mod.cdk"

!
!     ---------------------------------------------------------------
!
      rewind (Lun_rstrt)
      read (Lun_rstrt) Lctl_step,Step_kount,Init_mode_L
      read (Lun_rstrt) NTR_runstrt_S, NTR_horzint_L, &
        NTR_Tr3d_name_S,NTR_Tr3d_wload,NTR_Tr3d_hzd, &
        NTR_Tr3d_mono,NTR_Tr3d_mass,NTR_Tr3d_ntr
      close(Lun_rstrt)
      call fclos(Lun_rstrt) 
!
!     ---------------------------------------------------------------
!      
      return
      end
