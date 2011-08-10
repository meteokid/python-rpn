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

!**s/r e_bmfsplitxy2 - wrapper and call to bmf_splitwritexy
!
      subroutine e_bmfsplitxy2 ( F_s, F_ni, F_nj, F_v_S, F_k, F_gnk,  &
                                     F_gni, F_hgrid,F_vgrid,F_scat )
      implicit none
#include <arch_specific.hf>
!
      character* (*) F_v_S
      integer F_ni,F_nj,F_k,F_gnk,F_gni,F_hgrid,F_vgrid,F_scat
      real    F_s(F_ni,F_nj)
!
!author
!     L. Corbeil    - May 2001
!
!revision
! v2_30 - Corbeil, L.       - initial version
! v3_00 - Desgagne & Lee    - Lam configuration
!
!object
!  See ID section
!
!______________________________________________________________________ 
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! F_v_S (in)         | Name of variable contained in F_fi              |
! F_ni (in)          | Size of variable contained in F_fi              |
! F_nj (in)          | Size of variable contained in F_fi              |
! F_bmf_time1 (in)   | Time tag for bmf_splitwrite (yyyymmdd)          |
! F_bmf_time2 (in)   | Time tag for bmf_splitwrite (hhmmsscc)          |
! F_hgrid (in)       | Horizontal grid descriptor for bmf_splitwrite   |
! F_vgrid (in)       | Vertical grid descriptor for bmf_splitwrite     |
! F_bmf_dtyp  (in)   | Data type for bmf_splitwrite                    |
! F_scat  (in)       | Scatter list tag for bmf_splitwrite             |
! F_s  (in)          | Field to be wrapped and written                 |
!-----------------------------------------------------------------------
!
#include "bmf.cdk"
!
      external rpn_comm_split
      integer i, j, k, cnt, err
      real wrk (F_gni*F_nj)
!
!     ---------------------------------------------------------------
!
      cnt=0
!     Gathering 
      do j=1,F_nj 
      do i=1,F_gni
         cnt = cnt + 1
         wrk(cnt) = F_s(i,j)
      enddo
      enddo
!
      write(6,1000) F_v_S, F_gni, F_nj
      call bmf_splitwritexy2 ( RPN_COMM_split,F_v_S,F_gni,F_nj,F_gnk,F_k,F_k, &
                               bmf_time1,bmf_time2,F_hgrid,F_vgrid, &
                               bmf_dtyp,F_scat,wrk)
!
 1000 format('e_bmfsplitxy for ',A,' ni=',i6,' nj=',i6)
!
      return
      end 
