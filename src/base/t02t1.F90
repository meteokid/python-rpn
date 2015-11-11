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

!**s/r t02t1 -  Rename time level t0 -> t1
!
      subroutine t02t1
      implicit none
#include <arch_specific.hf>

!author 
!     Michel Roch - rpn - nov 1993
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v2_30 - Edouard  S.       - remove pi' at the top
! v2_31 - Desgagne M.       - remove treatment of HU and QC and 
!                             re-introduce tracers
! v4_05 - Lepine M.         - VMM replacement with GMM
!
!object
!     Associate the variables at time t1 to the space on disk and memory
!     associated with the variables at time t0
	
#include "gmm.hf"
#include "glb_ld.cdk"
#include "schm.cdk"
#include "tr3d.cdk"
#include "vt0.cdk"
#include "vt1.cdk"

      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: ut_list  = (/ 'URT0', 'URT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: vt_list  = (/ 'VRT0', 'VRT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: tt_list  = (/ 'TT0', 'TT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: st_list  = (/ 'ST0', 'ST1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: wt_list  = (/ 'WT0', 'WT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: qt_list  = (/ 'QT0', 'QT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: zdt_list = (/ 'ZDT0', 'ZDT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: xdt_list = (/ 'XDT0', 'XDT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2), parameter :: qdt_list = (/ 'QDT0', 'QDT1' /)
      character(len=GMM_MAXNAMELENGTH) , dimension(2) :: tr_list
      integer i,istat
!
!     ---------------------------------------------------------------
!      
      istat = gmm_shuffle( ut_list)
      istat = gmm_shuffle( vt_list)
      istat = gmm_shuffle( tt_list)
      istat = gmm_shuffle( st_list)
      istat = gmm_shuffle(zdt_list)
      istat = gmm_shuffle(xdt_list)
      istat = gmm_shuffle( wt_list)
!
      if (.not. Schm_hydro_L) then
         istat = gmm_shuffle( qt_list)
         istat = gmm_shuffle(qdt_list)
      endif
!
      do i=1,Tr3d_ntr
         tr_list(1) = 'TR/'//trim(Tr3d_name_S(i))//':M'
         tr_list(2) = 'TR/'//trim(Tr3d_name_S(i))//':P'
         istat = gmm_shuffle(tr_list)
      end do
!
      return
      end
