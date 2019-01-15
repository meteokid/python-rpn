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

!/@*
subroutine phystats(F_stepcount, F_delt)
   use timestr_mod, only: timestr2step
   use statfld_dm_mod, only: statfld_dm
   use phy_itf, only: phy_get
   use phy_options, only: phystat_freq_S, phystat_dble_l, phystat_2d_l, phystat_list_s
   use phy_typedef
   use phy_getmeta_mod, only: phy_getmeta
   implicit none
   !@objective 
   integer, intent(in) :: F_stepcount     !Step kount incrementing from 0
   real,    intent(in) :: F_delt          !Time step length (sec)
   !@authors Desgagne, Chamberland, McTaggart-Cowan, Spacek -- Fall 2015
   !*@/
#include <arch_specific.hf>
#include <msg.h>
#include <rmnlib_basics.hf>
   logical, parameter :: NOSHORTMATCH_L = .false.

   integer, save :: phystat_freq = -1

   integer :: stat_precision, ivar, k, istat, nvars
   real, pointer :: tmpptr(:,:,:)
   character(len=64) :: msg_S,bpath_S
   type(phymeta), pointer :: metalist(:)
   !---------------------------------------------------------------
   if (phystat_freq == -1) then
      istat = timestr2step(phystat_freq, phystat_freq_S, dble(F_delt))
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_INFO,'(phystats) wrong format for phystat_freq_S: '//phystat_freq_S)
         phystat_freq = 0
      endif
   endif
   if (phystat_freq <= 0) then
      return
   else
      if (mod(F_stepcount,phystat_freq) /= 0) return
   endif
   if (phystat_list_s(1) == ' ') then
      call msg(MSG_INFO,'(phystats) No Stat Requested')
      return
   endif

   stat_precision = 4
   if (phystat_dble_l) stat_precision = 8
   
   bpath_S = ''
   if (phystat_list_s(1)(1:8) == 'ALLVARS=') then
      bpath_S = phystat_list_s(1)(9:)
   endif
   if (bpath_S /= '') then
      nullify(metalist)
      nvars = phy_getmeta(metalist, ' ', F_npath='V', F_bpath=bpath_S, &
           F_maxmeta=size(phystat_list_s), F_shortmatch=NOSHORTMATCH_L)
      phystat_list_s(:) = ' '
      do ivar = 1, nvars
         phystat_list_s(ivar) = metalist(ivar)%vname
      enddo
      if (associated(metalist)) deallocate(metalist, stat=istat)
   endif

   write(msg_S,'(a,i4.4,a)') '############### PHY BLOC STAT   ',F_stepcount,' ###################'
   call msg(MSG_INFO, msg_S)
   do ivar = 1, size(phystat_list_s)
      if (phystat_list_s(ivar) == ' ') exit

      nullify(tmpptr)
      istat = phy_get(tmpptr, phystat_list_s(ivar), F_quiet=.true.)
      if (associated(tmpptr)) then
         if (phystat_2d_l .and. ubound(tmpptr,3) > 1) then
            do k = lbound(tmpptr,3), ubound(tmpptr,3)
               write(msg_S,'(a,i4.4,a)') trim(phystat_list_s(ivar))//' (',k,')'

               call statfld_dm(tmpptr(:,:,k:k), msg_S, F_stepcount, 'phystats', stat_precision)
            enddo
         else
            call statfld_dm(tmpptr, phystat_list_s(ivar), F_stepcount, 'phystats', stat_precision)
         endif
         deallocate(tmpptr)
      else
         call msg(MSG_INFO,'(phystats) unable to get: '//phystat_list_s(ivar))
      endif

   enddo
   write(msg_S,'(a,i4.4,a)') '############### PHY BLOC STAT   ',F_stepcount,' DONE ##############'
   call msg(MSG_INFO, msg_S)
   !---------------------------------------------------------------
   return
end subroutine phystats
