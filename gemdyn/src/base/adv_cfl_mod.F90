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

module adv_cfl
   implicit none
   public
   save
!______________________________________________________________________
!                                                                      |
!  VARIABLES ASSOCIATED WITH CFL COMPUTATION FOR LAM (ADV_CFL_LAM)     |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! adv_cfl_i(i,j)     | i = the max value of the following:             |
!                    |     1: max cfl value found                      |
!                    |     2: the "I" grid point of max cfl found      |
!                    |     3: the "J" grid point of max cfl found      |
!                    |     4: the "K" grid point of max cfl found      |
!                    | j = the type of max cfl computed                |
!                    |     1: the largest horizontal courrant number   |
!                    |     2: the largest vertical   courrant number   |
!                    |     3: the largest 3-dimensional courrant number|
!                    |PE = the number of the PE (processor)            |
!                    |     The overall maximum cfl value of the entire |
!                    |     grid will be placed in PE 1 before printing |
! adv_cfl_8          | CFL value in each dir
!______________________________________________________________________|

      integer :: adv_cfl_i(3,3)
      real*8  :: adv_cfl_8(3)

end module adv_cfl
