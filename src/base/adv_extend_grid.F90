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
!
      subroutine adv_extend_grid (F_u_adw,  F_v_adw, F_w_adw, &
                              F_u_model,F_v_model, F_w_model, &
                             F_aminx,F_amaxx,F_aminy,F_amaxy, &
                           F_minx,F_maxx,F_miny,F_maxy,F_nk)
      use glb_ld
      use adv_grid
      use outgrid
                           implicit none 
#include <arch_specific.hf>

      integer :: F_aminx,F_amaxx,F_aminy,F_amaxy !I, adw local array bounds
      integer :: F_minx,F_maxx,F_miny,F_maxy !I, model's local array bounds
      integer :: F_nk           !I, number of levels
      real,dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk) :: F_u_model, F_v_model, F_w_model !I, Fields/winds on model-grid
      real,dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk) :: F_u_adw ,F_v_adw ,F_w_adw !O, fields/winds on adw-grid

   !@objective Extend the grid from model to adw with filled halos

    
      integer, parameter :: nrow=0   
!     
!---------------------------------------------------------------------
!     
      call rpn_comm_xch_halox( &
          F_u_model, F_minx,F_maxx,F_miny,F_maxy, &
          l_ni, l_nj, F_nk, adv_halox, adv_haloy, &
          G_periodx, G_periody, &
          F_u_adw, F_aminx,F_amaxx,F_aminy,F_amaxy, l_ni, nrow)
      
      call rpn_comm_xch_halox( &
          F_v_model, F_minx,F_maxx,F_miny,F_maxy, &
          l_ni, l_nj, F_nk, adv_halox, adv_haloy, &
          G_periodx, G_periody, &
          F_v_adw, F_aminx,F_amaxx,F_aminy,F_amaxy, l_ni, nrow)
 
      call rpn_comm_xch_halox( &
          F_w_model, F_minx,F_maxx,F_miny,F_maxy, &
          l_ni, l_nj, F_nk, adv_halox, adv_haloy, &
          G_periodx, G_periody, &
          F_w_adw, F_aminx,F_amaxx,F_aminy,F_amaxy, l_ni, nrow)
!     
!---------------------------------------------------------------------
!     
       return
       end subroutine adv_extend_grid
