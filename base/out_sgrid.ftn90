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
      subroutine out_sgrid2 ( F_x0,F_x1,F_y0,F_y1,F_ig1,F_ig2 ,&
                              F_periodx_L,F_stride,F_etikext_s )
      implicit none
#include <arch_specific.hf>
!
      integer F_x0, F_x1,F_y0,F_y1,F_stride,NI,NJ,F_ig1,F_ig2
      logical F_periodx_L
      character*(*) F_etikext_s
!
!AUTHOR   Michel Desgagne     July 2004
!
!REVISION
! v3_20 - Lee V.            - Adapted for GEMDM
! v3_30 - McTaggart-Cowan R.- Append user defined grid tag to namelist value
! v4_03 - Lee V.            - modification of Out_etik_S in out_sgrid only
! v4_06 - Lee V.            - add the grid descriptor ig1,ig2 in arguments
!
!
!ARGUMENTS
!    NAMES       I/O  TYPE  DESCRIPTION
!    F_x0        I    int   g_id
!    F_x1        I    int   g_if
!    F_y0        I    int   g_jd
!    F_y1        I    int   g_jf
!    F_stride    I    int   number of points to stride 
!    F_periodx   I    logic periodicity on X
!    F_etikext_s I    char  grid-specific tag extension

#include "out.cdk"
!
!----------------------------------------------------------------------
!
      Out_stride= F_stride
      Out_gridi0= F_x0
      Out_gridin= F_x1
      Out_gridj0= F_y0
      Out_gridjn= F_y1
      out_idl = max(F_x0 - out_bloci0 + 1, 1)
      out_ifl = min(F_x1 - out_bloci0 + 1, Out_blocni)
      out_jdl = max(F_y0 - out_blocj0 + 1, 1)
      out_jfl = min(F_y1 - out_blocj0 + 1, Out_blocnj)

      out_nisg  = 0
      out_njsg  = 0
      out_nisl  = 0
      out_njsl  = 0
      out_idg   = 0
      out_jdg   = 0

      if ((out_idl.le.Out_blocni).and.(out_ifl.ge.1).and. &
          (out_jdl.le.Out_blocnj).and.(out_jfl.ge.1) ) then
         out_idg = out_idl + out_bloci0 - F_x0
         out_ifg = out_ifl + out_bloci0 - F_x0
         out_jdg = out_jdl + out_blocj0 - F_y0
         out_jfg = out_jfl + out_blocj0 - F_y0
         out_nisg  = (F_x1 - F_x0) / Out_stride + 1
         out_njsg  = (F_y1 - F_y0) / Out_stride + 1
         out_nisl  = (out_ifg - out_idg) / Out_stride + 1
         out_njsl  = (out_jfg - out_jdg) / Out_stride + 1
      endif

      if (F_periodx_L) Out_nisg=Out_nisg+1

      Out_ig1 = F_ig1
      Out_ig2 = F_ig2
      Out_ig3 = out_idg
      Out_ig4 = out_jdg

      Out_etik_S = Out_etiket_S(1:min(len_trim(Out_etiket_S), &
         len(Out_etiket_S)-len_trim(F_etikext_s))) //trim(F_etikext_s)
!
!----------------------------------------------------------------------
!
      return
      end

