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

!**s/r set_poic_par_U - preparation of projection matrices in the 
!                       east-west, C grid model using reflexion 
!                       symmetry of U-grid (global)
!

!
      subroutine set_poic_par_U ( F_eival_8, F_evvec_8, F_odvec_8,  &
                                             F_xg_8, F_ni, NSTOR )
      implicit none
#include <arch_specific.hf>
!
      integer F_ni, NSTOR
      real*8  F_eival_8(F_ni), F_evvec_8(NSTOR,NSTOR)
      real*8  F_odvec_8(NSTOR,NSTOR), F_xg_8(F_ni)
!
!author
!     A.Qaddouri    - spring 2002
!
! revision 
! v3_01 - Qaddouri & Toviessi    - initial version
!
!object
!     See above id
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_eival_8    O    - eigenvalue vector
! F_evvec_8    O    - even eigenvector matrix
! F_odvec_8    O    - od eigenvector matrix
! F_xg_8       I    - scalar grid points
! F_ni         I    - number of points
! NSTOR        I    - storage dimension
!

#include "dcst.cdk"
!
      integer i, j, nev, nod
      real*8  x0e_8 (NSTOR,NSTOR) , x0o_8 (NSTOR,NSTOR) , &
              r_8 (NSTOR), xx
!
      real*8 ZERO, HALF, ONE, TWO, FOUR
      parameter( ZERO = 0.0 , HALF = 0.5, ONE = 1.0,  &
                  TWO = 2.0 , FOUR = 4.0 )
!notes
!
!    NSTOR = (F_ni+2)/2 + ( 1 - mod((F_ni+2)/2,2) )
!    to minimize memory bank conflicts
! --------------------------------------------------------------------
!
!     Prepare even and odd matrices for generalized eigenvalue problems
!
      do i=1,NSTOR
      do j=1,NSTOR
         F_evvec_8(i,j) = ZERO
         F_odvec_8(i,j) = ZERO
         x0e_8    (i,j) = ZERO
         x0o_8    (i,j) = ZERO
      enddo
      enddo
!
      nev = ( F_ni  + 1 )/2
      nod =   F_ni - nev
!
!     even operators (upper triangular part only)
!
      do i=1, nev-1 
         F_evvec_8(i,i+1) =TWO /(HALF*( F_xg_8(i+2) - F_xg_8(i) ))
      enddo
!
      do i=1,nev-1
         x0e_8(i,i)= two*(F_xg_8(i+1)-F_xg_8(i))
      enddo
!
      F_evvec_8(1,1) = - F_evvec_8(1,2)
!
      do i=2, nev - 1
         F_evvec_8(i,i) = - ( F_evvec_8(i-1,i) + F_evvec_8(i,i+1) )
      end do
      F_evvec_8(nev,nev) = - F_evvec_8(nev-1,nev)
      if ( F_ni .eq. 2 * ( F_ni/2 ) ) then
         x0e_8(nev,nev) = two*(F_xg_8(nev+1)-F_xg_8(nev))
      else
         x0e_8(nev,nev) =   half*(F_xg_8(nev+2) - F_xg_8(nev))
      end if
!
!     odd operators (upper triangular part only)
!
      do i=1,nod
      do j=i,nod
         F_odvec_8(i,j) = F_evvec_8(i,j)
         x0o_8(i,j) = x0e_8(i,j)
      end do
      end do

      xx= Dcst_pi_8 + half*(F_xg_8(2)+F_xg_8(1)-F_xg_8(F_ni))
      F_odvec_8(1,1)=- 4./xx &
                     - 2./(half*( 2.*Dcst_pi_8 - F_xg_8(F_ni-1))) 
      if ( F_ni .eq. 2 * ( F_ni/2 ) )  &
          F_odvec_8(nod,nod)=-1./(half*( F_xg_8(nev+1) - F_xg_8(nev-1))) &
          -4./(half*( F_xg_8(nev+2) - F_xg_8(nev))) &
          -1./(half*(F_xg_8(nev+3) - F_xg_8(nev+1)))
!     
!     even modes and eigeivalues
!
      call geneigl3 ( r_8, F_evvec_8, x0e_8, nev, NSTOR, 3*F_ni-1 )
      F_eival_8(1:nev) = r_8(1:nev)
!
!     odd modes and eigeivalues
!
      call geneigl3 ( r_8, F_odvec_8, x0o_8, nod, NSTOR, 3*F_ni-1 )
      do i=1,nod
         F_eival_8(nev+i) = r_8(i)
      end do
!
!-------------------------------------------------------------------
!    
      return
      end
