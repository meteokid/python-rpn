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
      subroutine timing_start2 ( mynum, myname_S, mylevel )
      implicit none
#include <arch_specific.hf>

      integer mynum,mylevel
      character* (*) myname_S

!author
!     M. Desgagne   -- Winter 2012 --
!revision
! v4_40 - Desgagne - initial version
! v4_80 - Desgagne - introduce timer_level and timer_cnt

      include "timing.cdk"

      DOUBLE PRECISION omp_get_wtime

      if (Timing_S=='YES') call tmg_start ( mynum, myname_S )

      nam_subr_S(mynum) = myname_S ; timer_level(mynum) = mylevel
      tb        (mynum) = omp_get_wtime()
      timer_cnt (mynum) = timer_cnt(mynum) + 1

      return
      end
