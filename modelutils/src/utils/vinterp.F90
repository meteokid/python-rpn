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
module vinterp_mod
   use vGrid_Descriptors
   use vgrid_wb
   use samevgrid_mod
   implicit none
   private
   !@objective 
   !@author Stephane Chamberland,2011-04
   !@description
   ! Public functions
   public :: vinterp,vinterp0,vinterp1,vinterp10,vinterp01
   ! Public constants
   !*@/
#include <arch_specific.hf>
#include <rmnlib_basics.hf>
#include <msg.h>
#include <gmm.hf>

   interface vinterp
      module procedure vinterp0
      module procedure vinterp1
      module procedure vinterp10
      module procedure vinterp01
   end interface

   real,parameter :: EPSILON_R4 = 1.e-6

contains

   !/@*
   function vinterp0(F_dataout,F_vgridout,F_ip1listout,F_datain,F_vgridin,F_ip1listin,F_sfcfldout,F_sfcfldin,F_nlinbot,F_msg_S) result(F_istat)
      implicit none
      !@objective
      !@arguments
      real,pointer :: F_dataout(:,:,:),F_datain(:,:,:)
      type(vgrid_descriptor),intent(in) :: F_vgridout,F_vgridin
      integer,intent(in) :: F_ip1listout(:),F_ip1listin(:)
      real,pointer :: F_sfcfldout(:,:),F_sfcfldin(:,:)
      integer,intent(in),optional :: F_nlinbot
      character(len=*),intent(in),optional :: F_msg_S
      !@return
      integer :: F_istat
      !*@/
      character(len=256) :: msg_S,tmp_S
      integer,pointer :: samevert_list(:)
      integer :: istat,nlinbot,nijkout(3),nijkin(3),lijk(3),uijk(3),k
      integer, dimension(size(F_datain,dim=3),1) :: slist
      real,target :: sfcfld0(1,1)
      real,pointer :: levelsin(:,:,:),levelsout(:,:,:),sfcfldin(:,:)
      real,dimension(size(F_datain,dim=3)) :: scol
      real,dimension(size(F_datain,dim=1),size(F_datain,dim=2),size(F_datain,dim=3)) :: sdatain,slevelsin
      logical :: use_same_sfcfld_L,samevert_L
      !------------------------------------------------------------------
      call msg(MSG_DEBUG,'(vinterp) vinterp0 [BEGIN]')
      F_istat = RMN_ERR
      msg_S = ''
      if (present(F_msg_S)) msg_S = F_msg_S
      if (.not.associated(F_datain)) then
         call msg(MSG_WARNING,'(vinterp) Cannot Interpolate, null pointer: '//trim(msg_S))
         return
      endif
      nlinbot = 0 
      if (present(F_nlinbot)) nlinbot = F_nlinbot
      if (associated(F_sfcfldin) .and. .not.associated(F_sfcfldin,F_sfcfldout)) then
         use_same_sfcfld_L = .false.
         sfcfldin => F_sfcfldin
      else
         use_same_sfcfld_L = .true.
         sfcfldin => F_sfcfldout
      endif

      nijkin  = shape(F_datain)
      if (.not.associated(F_dataout)) then
         lijk = lbound(F_datain) ; uijk = ubound(F_datain)
         allocate(F_dataout(lijk(1):uijk(1),lijk(2):uijk(2),size(F_ip1listout))) !#,stat=istat
      endif
      nijkout = shape(F_dataout)
      if (.not.( &
           all(nijkout(1:2) == nijkin(1:2)) .and. &
           all(nijkout(1:2) == shape(F_sfcfldout)) .and. &
           all(nijkout(1:2) == shape(sfcfldin)) .and. &
           nijkout(3) == size(F_ip1listout) .and. &
           nijkin(3)  == size(F_ip1listin) &
           )) then
!!$         print *,'(vinterp)', &
!!$              (all(nijkout(1:2) == nijkin(1:2))), &
!!$              (all(nijkout(1:2) == shape(F_sfcfldout))), &
!!$              (all(nijkout(1:2) == shape(sfcfldin))), &
!!$              (nijkout(3) == size(F_ip1listout)), &
!!$              (nijkin(3)  == size(F_ip1listin))
!!$         print *,'(vinterp) a ',(all(nijkout(1:2) == nijkin(1:2))),' : ',nijkout(1:2),' : ',nijkin(1:2)
!!$         print *,'(vinterp) b ',(all(nijkout(1:2) == shape(F_sfcfldout))),' : ',nijkout(1:2),' : ',shape(F_sfcfldout)
!!$         print *,'(vinterp) c ',(all(nijkout(1:2) == shape(sfcfldin))),' : ',nijkout(1:2),' : ',shape(sfcfldin)
!!$         print *,'(vinterp) d ',(nijkout(3) == size(F_ip1listout)),' : ',nijkout(3),' : ',size(F_ip1listout)
!!$         print *,'(vinterp) e ',(nijkin(3)  == size(F_ip1listin)),' : ',nijkin(3),' : ',size(F_ip1listin)
!!$         print *,'(vinterp) use_same_sfcfld_L=',use_same_sfcfld_L
!!$         print *,'(vinterp) F_datain   ',lbound(F_datain),' : ',ubound(F_datain)
!!$         print *,'(vinterp) F_dataout  ',lbound(F_dataout),' : ',ubound(F_dataout)
!!$         print *,'(vinterp) ip1lstin   ',lbound(F_ip1listin),' : ',ubound(F_ip1listin),' : ',nijkin(3)
!!$         print *,'(vinterp) ip1lstout  ',lbound(F_ip1listout),' : ',ubound(F_ip1listout),' : ',nijkout(3)
!!$         print *,'(vinterp) sfcfldin   ',lbound(sfcfldin),' : ',ubound(sfcfldin)
!!$         print *,'(vinterp) F_sfcfldout',lbound(F_sfcfldout),' : ',ubound(F_sfcfldout)
         call msg(MSG_WARNING,'(vinterp) Cannot Interpolate, not same shape: '//trim(msg_S))
         return
      endif
      
      allocate(samevert_list(nijkout(3))) !#,stat=istat
      samevert_L = samevgrid(F_vgridin,F_ip1listin,F_vgridout,F_ip1listout,samevert_list)
      if (samevert_L .and. (.not.use_same_sfcfld_L) .and. &
           associated(F_sfcfldout) .and. associated(sfcfldin)) then
         if (any(abs(F_sfcfldout-sfcfldin) > EPSILON_R4)) samevert_L = .false.
      endif

      tmp_S = 'Interpolating'
      nullify(levelsout,levelsin)
      if (samevert_L) then
         tmp_S = 'No Interpolation for'
         sfcfld0 = 100000. !TODO: make sure F_vgridin is pressure based
         sfcfldin => sfcfld0
         F_istat = RMN_OK
      else
         !TODO: if vgrid needs sfcfield and sfcfld => null then error
         F_istat = vgd_levels(F_vgridout,F_ip1listout,levelsout,F_sfcfldout,in_log=.false.)
      endif
      F_istat = min(vgd_levels(F_vgridin,F_ip1listin,levelsin,sfcfldin,in_log=.false.),F_istat)

      if (.not.RMN_IS_OK(F_istat)) then
         call msg(MSG_WARNING,'(vinterp) Cannot Interpolate, problem getting levels: '//trim(msg_S))
         return
      endif
      call msg(MSG_INFO,'(vinterp) '//trim(tmp_S)//': '//trim(msg_S))


      if (samevert_L) then
!!$         !TODO: check that output levels are increasing as well
         do k=1,size(samevert_list)
            F_dataout(:,:,k) = F_datain(:,:,samevert_list(k))
         enddo
      else

         ! Sort input levels into increasing pressures
         !#TODO: sorting should be done in vgrid from file, could be ascending or descending depending on model (ascending for GEM), only check monotonicity in here
         scol = levelsin(1,1,:)
         do k=1,size(scol)
            slist(k,:) = minloc(scol)
            scol(slist(k,1)) = maxval(scol)+1.
         enddo

         do k=1,size(scol)
            sdatain(:,:,k) = F_datain(:,:,slist(k,1))
            slevelsin(:,:,k) = levelsin(:,:,slist(k,1))
         enddo
!!$         print '(2a,4i6,10e14.7,2l)','(input v)',trim(msg_S),&
!!$              minval(F_ip1listout), maxval(F_ip1listout), &
!!$              minval(F_ip1listin), maxval(F_ip1listin),&
         call vte_intvertx4(F_dataout,sdatain,slevelsin,levelsout, &
              nijkout(1)*nijkout(2),nijkin(3),nijkout(3),&
              msg_S,nlinbot)
!!$         print '(2a,10x,6(a,2e14.7,x),i,2l)',&
!!$              '(inp v) ',trim(msg_S),&
!!$              'o :',minval(F_dataout), maxval(F_dataout),&
!!$              'i :',minval(F_datain), maxval(F_datain),&
!!$              'is:',minval(F_sfcfldin),maxval(F_sfcfldin),&
!!$              'os:',minval(F_sfcfldout), maxval(F_sfcfldout),&
!!$              'il:',minval(slevelsin), maxval(slevelsin), &
!!$              'ol:',minval(levelsout), maxval(levelsout), &
!!$              F_nlinbot,samevert_L,use_same_sfcfld_L
      endif

!!$      if (maxval(F_datain) /= minval(F_datain)) then
!!$         ijk = maxloc(F_datain)
!!$         do k=1,nijkin(3)
!!$            print '(a,i3,2e,a,2e)','(vinterp) IN ',k,F_datain(ijk(1),ijk(2),k),levelsin(ijk(1),ijk(2),k),' : ',maxval(F_datain(:,:,k)),maxval(levelsin(:,:,k))
!!$         enddo
!!$         do k=1,nijkout(3)
!!$            print '(a,i3,2e,a,2e)','(vinterp) OUT',k,F_dataout(ijk(1),ijk(2),k),levelsout(ijk(1),ijk(2),k),' : ',maxval(F_dataout(:,:,k)),maxval(levelsout(:,:,k))
!!$         enddo
!!$      endif

      if (associated(samevert_list)) deallocate(samevert_list,stat=istat)
      if (associated(levelsout)) deallocate(levelsout,stat=istat)
      if (associated(levelsin)) deallocate(levelsin,stat=istat)
      write(tmp_S,'(i6)') F_istat
      call msg(MSG_DEBUG,'(vinterp) vinterp0 [END] '//trim(tmp_S))
      !------------------------------------------------------------------
      return
   end function vinterp0


   !/@*
   function vinterp1(F_dataout,F_vgridout_S,F_datain,F_vgridin_S,F_sfcfldout,F_sfcfldin,F_nlinbot,F_msg_S,F_use_same_sfcfld_L) result(F_istat)
      implicit none
      !@objective
      !@arguments
      real,pointer :: F_dataout(:,:,:),F_datain(:,:,:)
      character(len=*),intent(in) :: F_vgridout_S,F_vgridin_S
      real,pointer,optional :: F_sfcfldout(:,:),F_sfcfldin(:,:)
      integer,intent(in),optional :: F_nlinbot
      character(len=*),intent(in),optional :: F_msg_S
      logical,intent(in),optional :: F_use_same_sfcfld_L
      !@return
      integer :: F_istat
      !*@/
      character(len=256) :: msg_S,tmp_S
      integer :: nlinbot
      type(vgrid_descriptor) :: vgridout,vgridin
      real,pointer :: sfcfldout(:,:),sfcfldin(:,:)
      integer,pointer :: ip1listout(:),ip1listin(:)
      logical :: use_same_sfcfld_L
      !------------------------------------------------------------------
      call msg(MSG_DEBUG,'(vinterp) vinterp1 [BEGIN]')
      F_istat = RMN_ERR
      msg_S = ''
      if (present(F_msg_S)) msg_S = F_msg_S
      nlinbot = 0 
      if (present(F_nlinbot)) nlinbot = F_nlinbot
      use_same_sfcfld_L = .false.
      if (present(F_use_same_sfcfld_L)) use_same_sfcfld_L = F_use_same_sfcfld_L

      nullify(ip1listout,ip1listin,sfcfldout,sfcfldin)
      if (present(F_sfcfldout)) then
         if (associated(F_sfcfldout)) sfcfldout => F_sfcfldout
      endif
      if (present(F_sfcfldin)) then
         if (associated(F_sfcfldin)) sfcfldin => F_sfcfldin
      endif
      F_istat = priv_vgrid_details(F_vgridout_S,vgridout,ip1listout,sfcfldout)
      if (use_same_sfcfld_L) sfcfldin => sfcfldout
      if (RMN_IS_OK(F_istat)) &
           F_istat = priv_vgrid_details(F_vgridin_S,vgridin,ip1listin,sfcfldin)
      if (.not.RMN_IS_OK(F_istat)) then
         call msg(MSG_WARNING,'(vinterp) Cannot Interpolate, missing vgrid info: '//trim(msg_S))
         return
      endif

      F_istat = vinterp0(F_dataout,vgridout,ip1listout,F_datain,vgridin,ip1listin,sfcfldout,sfcfldin,nlinbot,msg_S)
      write(tmp_S,'(i6)') F_istat
      call msg(MSG_DEBUG,'(vinterp) vinterp1 [END] '//trim(tmp_S))
      !------------------------------------------------------------------
      return
   end function vinterp1


   !/@*
   function vinterp01(F_dataout,F_vgridout,F_ip1listout,F_datain,F_vgridin_S,F_sfcfldout,F_sfcfldin,F_nlinbot,F_msg_S) result(F_istat)
      implicit none
      !@objective
      !@arguments
      real,pointer :: F_dataout(:,:,:),F_datain(:,:,:)
      type(vgrid_descriptor),intent(in) :: F_vgridout
      integer,intent(in) :: F_ip1listout(:)
      character(len=*),intent(in) :: F_vgridin_S
      real,pointer,optional :: F_sfcfldout(:,:),F_sfcfldin(:,:)
      integer,intent(in),optional :: F_nlinbot
      character(len=*),intent(in),optional :: F_msg_S
      !@return
      integer :: F_istat
      !*@/
      character(len=256) :: msg_S,tmp_S
      integer :: nlinbot
      type(vgrid_descriptor) :: vgridin
      real,pointer :: sfcfldout(:,:),sfcfldin(:,:)
      integer,pointer :: ip1listin(:)
      !------------------------------------------------------------------
      call msg(MSG_DEBUG,'(vinterp) vinterp01 [BEGIN]')
      F_istat = RMN_ERR
      msg_S = ''
      if (present(F_msg_S)) msg_S = F_msg_S
      nlinbot = 0 
      if (present(F_nlinbot)) nlinbot = F_nlinbot

      nullify(ip1listin,sfcfldout,sfcfldin)
      if (present(F_sfcfldout)) then
         if (associated(F_sfcfldout)) sfcfldout => F_sfcfldout
      endif
      if (present(F_sfcfldin)) then
         if (associated(F_sfcfldin)) sfcfldin => F_sfcfldin
      endif
      F_istat = priv_vgrid_details(F_vgridin_S,vgridin,ip1listin,sfcfldin)
      if (.not.RMN_IS_OK(F_istat)) then
         call msg(MSG_WARNING,'(vinterp) Cannot Interpolate, vgrid info: '//trim(msg_S))
         return
      endif

      F_istat = vinterp0(F_dataout,F_vgridout,F_ip1listout,F_datain,vgridin,ip1listin,sfcfldout,sfcfldin,nlinbot,msg_S)
      write(tmp_S,'(i6)') F_istat
      call msg(MSG_DEBUG,'(vinterp) vinterp01 [END] '//trim(tmp_S))
     !------------------------------------------------------------------
      return
   end function vinterp01


   !/@*
   function vinterp10(F_dataout,F_vgridout_S,F_datain,F_vgridin,F_ip1listin,F_sfcfldout,F_sfcfldin,F_nlinbot,F_msg_S) result(F_istat)
      implicit none
      !@objective
      !@arguments
      real,pointer :: F_dataout(:,:,:),F_datain(:,:,:)
      character(len=*),intent(in) :: F_vgridout_S
      type(vgrid_descriptor),intent(in) :: F_vgridin
      integer,intent(in) :: F_ip1listin(:)
      real,pointer,optional :: F_sfcfldout(:,:),F_sfcfldin(:,:)
      integer,intent(in),optional :: F_nlinbot
      character(len=*),intent(in),optional :: F_msg_S
      !@return
      integer :: F_istat
      !*@/
      character(len=256) :: msg_S,tmp_S
      integer :: nlinbot
      type(vgrid_descriptor) :: vgridout
      real,pointer :: sfcfldout(:,:),sfcfldin(:,:)
      integer,pointer :: ip1listout(:)
      !------------------------------------------------------------------
      call msg(MSG_DEBUG,'(vinterp) vinterp10 [BEGIN]')
      F_istat = RMN_ERR
      msg_S = ''
      if (present(F_msg_S)) msg_S = F_msg_S
      nlinbot = 0 
      if (present(F_nlinbot)) nlinbot = F_nlinbot

      nullify(ip1listout,sfcfldout,sfcfldin)
      if (present(F_sfcfldout)) then
         if (associated(F_sfcfldout)) sfcfldout => F_sfcfldout
      endif
      if (present(F_sfcfldin)) then
         if (associated(F_sfcfldin)) sfcfldin => F_sfcfldin
      endif
      F_istat = priv_vgrid_details(F_vgridout_S,vgridout,ip1listout,sfcfldout)
      if (.not.RMN_IS_OK(F_istat)) then
         call msg(MSG_WARNING,'(vinterp) Cannot Interpolate, vgrid info: '//trim(msg_S))
         return
      endif

      F_istat = vinterp0(F_dataout,vgridout,ip1listout,F_datain,F_vgridin,F_ip1listin,sfcfldout,sfcfldin,nlinbot,msg_S)
      write(tmp_S,'(i6)') F_istat
      call msg(MSG_DEBUG,'(vinterp) vinterp10 [END] '//trim(tmp_S))
      !------------------------------------------------------------------
      return
   end function vinterp10


   !==== Private Functions =================================================


   function priv_vgrid_details(F_vgrid_S,F_vgrid,F_ip1list,F_sfcfld) result(F_istat)
      implicit none
      character(len=*),intent(in) :: F_vgrid_S
      type(vgrid_descriptor),intent(out) :: F_vgrid
      integer,pointer :: F_ip1list(:)
      real,pointer :: F_sfcfld(:,:)
      integer :: F_istat,istat,vtype
      character(len=32) :: sfcfld_S
      !------------------------------------------------------------------
      F_istat = vgrid_wb_get(F_vgrid_S,F_vgrid,F_ip1list,vtype,sfcfld_S)
      if (RMN_IS_OK(F_istat) .and. sfcfld_S /= ' ' .and. .not.associated(F_sfcfld)) &
           istat = gmm_get(sfcfld_S,F_sfcfld)

      if (.not.(RMN_IS_OK(F_istat) .and. (associated(F_sfcfld) .or. sfcfld_S == ' '))) then
         istat = vgd_free(F_vgrid)
         deallocate(F_ip1list,stat=istat)
         F_istat = RMN_ERR
!!$      else
!!$         if (associated(F_sfcfld).and.associated(F_ip1list)) then
!!$            print *,'(vinterp) vgrid=',trim(F_vgrid_S),'; ref=',trim(sfcfld_S),'; nijk=',shape(F_sfcfld),size(F_ip1list)
!!$         else
!!$            print *,'(vinterp) vgrid=',trim(F_vgrid_S),'; ref=',trim(sfcfld_S)
!!$         endif
      endif
      !------------------------------------------------------------------
      return
   end function priv_vgrid_details

end module vinterp_mod

