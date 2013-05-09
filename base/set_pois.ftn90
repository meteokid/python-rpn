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

!**s/r set_pois - 64 bits interface for preparation of projection matrix
!                in the east-west or vertical direction
!             for staggered model symmetric and  nonsymmetric_ln(Z)
!             version
!
      subroutine set_pois ( F_eval_8,F_levec_8,F_evec_8,F_npts,F_dim )
      implicit none
#include <arch_specific.hf>
!
      integer F_npts, F_dim
      real*8  F_eval_8(F_dim), F_evec_8(F_dim*F_dim), &
              F_levec_8(F_dim*F_dim)
!
!author
!     Abdessamad Qaddouri  2007
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_eval_8     O    - eigenvalue vector
! F_evec_8     O    - eigenvector matrix
! F_npts       I    - number of points
! F_dim        I    - field dimension


      real*8  r_8(F_dim)
      real*8, dimension (F_dim*F_dim) :: br_8, bl_8
!
! --------------------------------------------------------------------
!
      call preverln ( r_8, bl_8, br_8, F_npts, F_dim )
!
!     transfer results back in input/output arrays
!
      F_evec_8 = br_8 ; F_levec_8 = bl_8 ; F_eval_8 = r_8
!
! --------------------------------------------------------------------
!
      return
      end
