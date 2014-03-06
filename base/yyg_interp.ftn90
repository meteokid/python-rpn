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

!**s/r yyg_interp - YY horizontal interpolation controler

      subroutine yyg_interp ( FF, F, Imx,Imy, geomgx,geomgy,   &
                              Minx,Maxx,Miny,Maxy,Xi,Yi,mono_l,&
                              F_interp_S ) 
      implicit none
#include <arch_specific.hf>
       
      character* (*) F_interp_S
      logical mono_l
      integer Imx,Imy, Minx,Maxx,Miny,Maxy
      real*8  FF, F(Minx:Maxx,Miny:Maxy), &
              geomgx(Minx:Maxx),geomgy(Miny:Maxy), Xi, Yi
!
!author   
!       Michel Desgagne - Spring 2014
!revision
! v4_70 - Desgagne M.   - initial version
!
!----------------------------------------------------------------------
!
       if (trim(F_interp_S) == 'CUBIC') then
          call int_cub_lag3 ( FF, F, Imx,Imy, geomgx,geomgy,   &
                              Minx,Maxx,Miny,Maxy,Xi,Yi,mono_l )
       elseif (trim(F_interp_S) == 'LINEAR') then
          print*, 'stop in yyg_interp for F_interp_S= ',trim(F_interp_S)
          stop
! to be completed
!          call int_lin_lag2  ( FF, F, Imx,Imy, geomgx,geomgy,   &
!                              Minx,Maxx,Miny,Maxy,Xi,Yi        )
       elseif (trim(F_interp_S) == 'NEAREST') then
          print*, 'stop in yyg_interp for F_interp_S= ',trim(F_interp_S)
          stop
! to be completed
!          call int_near_lag2 ( FF, F, Imx,Imy, geomgx,geomgy,   &
!                              Minx,Maxx,Miny,Maxy,Xi,Yi        )
       endif
!
!----------------------------------------------------------------------
!
       return
       end 
