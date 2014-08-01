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

!**s/r get_px - get 3D pressure using A,B, and p0 (or a 2D pressure field) for
!                    the purpose of obtaining the hydrostatic surface
!                    pressure
!
!
      integer function get_px(F_px, F_p0, NN, F_a_8, F_b_8, NK, F_vcode,F_logpx_L)
      implicit none
#include <arch_specific.hf>
!
      integer NN, NK, F_vcode
      logical F_logpx_L
      real F_px(NN,NK), F_p0(NN)
      real*8 F_a_8(NK), F_b_8(NK)
!
!author
!     V.Lee July 2008 (from GEMDM, hybrid code)
!     and contributions from Andre Plante, Cecilien Charette
!
!revision
!
! v4_05 - Lee V.    - assume F_p0 is in Pascals or ln(Pascals) only
!
!object
!       see id section
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_px           O    - 3D pressure field in Pascals
! F_p0           I    - incoming 2D surface pressure (P0 in Pa, or dimensionless)
! NN             I    - number of points in the 2D field of F_p0
! F_a_8          I    - A 
! F_b_8          I    - B
! NK             I    - Number of levels to the 3D pressure field
! F_vcode        I    - code for:
!
!                     2 formulae             p0             6 anal
!
!            0 => p =    A                         (B=0),  prs-anal
!            1 => p =    A+B*   ps       , F_p0=ps (A=0),  sig-anal
!            2 => p =    A+B*   ps       , F_p0=ps      ,  etasef-anal
!            3 => p =    A+B*   ps       , F_p0=ps      ,  eta-anal
!            4 => p =    A+B*   ps       , F_p0=ps      ,  hyb-anal
!            5 => p =    A+B*   ps       , F_p0=ps      ,  ecm-anal
!            6 => p =exp(A+B*ln(ps/pref)), F_p0=ps      ,  stg-anal
!

!
!*
      integer i,k
!
!     ---------------------------------------------------------------
!
      get_px = 0

      if (F_logpx_L) then
!     FOR result in LOG
          if (F_vcode.eq.0) then
!             Pressure levels
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = log(F_a_8(k)*100.)
              enddo
              enddo
          else if (F_vcode.eq.1) then
!             Sigma levels,p0 is in Pascals
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = log(F_b_8(k)*F_p0(i))
              enddo
              enddo
          else if (F_vcode.eq.2.or.F_vcode.eq.3.or.F_vcode.eq.4) then
!             ETASEF levels
!             ETA levels
!             Hyb unstaggered levels,p0 is in Pascals
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = log(F_a_8(k) + F_b_8(k)*F_p0(i))
              enddo
              enddo
          else if (F_vcode.eq.6) then
!             Staggered Hybrid levels (lnp-type),p0 is dimensionless
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = F_a_8(k) + F_b_8(k)*F_p0(i)
              enddo
              enddo
          else 
              print *,'Error in get_px: unrecognizable datatype: F_vcode=',F_vcode
              get_px = -1
          endif

      else

!     FOR result NOT in LOG
          if (F_vcode.eq.0) then
!             Pressure levels
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = F_a_8(k)*100.0
              enddo
              enddo
          else if (F_vcode.eq.1) then
!             Sigma levels
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = F_b_8(k)*F_p0(i)
              enddo
              enddo
          else if (F_vcode.eq.2.or.F_vcode.eq.3.or.F_vcode.eq.4) then
!             ETASEF levels
!             ETA levels
!             Hyb unstaggered levels
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = F_a_8(k) + F_b_8(k)*F_p0(i)
              enddo
              enddo
          else if (F_vcode.eq.6) then
!             Staggered Hybrid levels (lnp-type),p0 is dimensionless
              do k=1,NK
              do i=1,NN
                 F_px(i,k) = exp(F_a_8(k) + F_b_8(k)*F_p0(i))
              enddo
              enddo
          else 
              print *,'Error in get_px: unrecognizable datatype: F_vcode=',F_vcode
              get_px = -1
          endif
      endif
      return
      end
