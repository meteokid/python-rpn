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
           subroutine adv_print_cliptrj_s(cnt,F_ni,F_nj,F_nk,k0,mesg)
           implicit none
#include <arch_specific.hf>

#include "msg.h"  
#include "adv_grid.cdk"

      Integer :: F_ni,F_nj,F_nk,k0
      Integer :: cnt,sum_cnt,n,totaln,err
      character(len=MSG_MAXLEN) :: msg_S
      character(len=*) :: mesg

      n = max(1,adv_maxcfl)
      totaln = (F_ni*n*2 + (F_nj-2*n)*n*2) * (F_nk-k0+1)
      call rpn_comm_Allreduce(cnt,sum_cnt,1,"MPI_INTEGER", "MPI_SUM","grid",err)
  
      if(sum_cnt .ne. 0 ) then
        write(msg_S,'(a,i5,a,f6.2,2x,a)')  &
         ' ADV trajtrunc : npts=',sum_cnt, &
         ', %=',real(sum_cnt)/real(totaln)*100., &
         mesg
         call msg(MSG_INFO,msg_S)
       endif

      end subroutine  adv_print_cliptrj_s


