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

!***s/r ser_out - Write time series buffers
!
      subroutine ser_out (F_headr_L, date, satuco )
      implicit none
#include <arch_specific.hf>
!
      logical F_headr_L, satuco
      integer, dimension(14) :: date
!
!author
!     Desgagne M.    Spring 2013
!
!revision
! v4_60 - Desgagne M.       - initial version
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_headr_L     I         true: first record header will be written
!----------------------------------------------------------------
!

#include <WhiteBoard.hf>
 include "thermoconsts.inc"
#include "series.cdk"

      logical wr_L
      integer i, j, k, err, nkm, nkt, vcode, hgc(4), mype, mycol, myrow
      character*12 ptetik_S
      real dgrw
!
!     ---------------------------------------------------------------
!
      if (series_paused) return

      call rpn_comm_REDUCE (Xstb_sers,Xstb_sersx,Xst_dimsers, &
                        "MPI_INTEGER","MPI_BOR",Xst_master_pe,"grid",err)
      call rpn_comm_REDUCE (Xstb_serp,Xstb_serpx,Xst_dimserp, &
                        "MPI_INTEGER","MPI_BOR",Xst_master_pe,"grid",err)
      call rpn_comm_mype (mype, mycol, myrow)

      wr_L=.false.
      if (mype.eq.Xst_master_pe) then
         Xstb_sers = Xstb_sersx
         Xstb_serp = Xstb_serpx
         wr_L=(.not.P_serg_sroff_L)
      endif

      err= wb_get ('model/Output/etik',  ptetik_S)
      i= 4
      err= wb_get ('model/Hgrid/hgcrot'   , hgc, i)
      err= wb_get ('model/Vgrid/size-hybm', nkm)
      err= wb_get ('model/Vgrid/size-hybt', nkt)
      i= 2

!     Put the character 'G' in front of etiket to allow
!     correct rotation of wind related variables by
!     feseri program

      dgrw=0.
      ptetik_S(1:1)='G'

      call ser_write2( date, ptetik_S, hgc, dgrw             ,&
                       nkm, nkt, &
                       RGASD  , GRAV, .false., satuco, F_headr_L, wr_L )

      if (mype.eq.Xst_master_pe) then
         Xstb_sers = 0. ; Xstb_serp = 0.
      endif
!
!     ---------------------------------------------------------------
!
      return
      end
