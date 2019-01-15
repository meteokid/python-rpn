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

#include <msg.h>

integer function mpi_comm_free(i)
   integer i
   mpi_comm_free = -1
   return
end function mpi_comm_free

!/@
module test_vgrid_wb_mod
   use testutils
   use vGrid_Descriptors
   use vgrid_wb
   private
   public :: test_vgrid_wb_put,test_vgrid_wb_get
   !@/
#include <rmnlib_basics.hf>


!!$   type(vgrid_descriptor),target,save :: myvgrid,myvgrid1,myvgrid2
!!$   type(vgrid_descriptor),pointer,save :: p_myvgrid,p_myvgrid1,p_myvgrid2

contains
   
   !/@
   subroutine test_vgrid_wb_put()
      implicit none
      !@objective 
      !@author Stephane Chamberland, 2012-01
      !@/

      character(len=512) :: name1_S,name2_S
      logical :: ok_L
      integer :: istat,nbip1

      integer,pointer,dimension(:) :: ip1_m,ip1_t,ip1_m1,ip1_t1,ip1_m2,ip1_t2,&
           ip1list,ip1list1,ip1list2

      type(vgrid_descriptor),target :: myvgrid,myvgrid1,myvgrid2
      type(vgrid_descriptor),pointer :: p_myvgrid,p_myvgrid1,p_myvgrid2
      ! ---------------------------------------------------------------------
      call priv_vgrid_wb_defvgrid(1,name1_S,myvgrid1,p_myvgrid1,ip1_m1,ip1_t1,ip1list1)
      call priv_vgrid_wb_defvgrid(2,name2_S,myvgrid2,p_myvgrid2,ip1_m2,ip1_t2,ip1list2)

      istat = vgrid_wb_put(' ',p_myvgrid1,ip1list1)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','put no name')

      istat = vgrid_wb_put('qw',99,ip1list1)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','put invalid type')

      allocate(ip1list(5:10),stat=istat)
      ip1list = -1
      istat = vgrid_wb_put('vinconsis',p_myvgrid1,ip1list)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','put inconsistent ip1')

      istat = vgrid_wb_put(name1_S,myvgrid1,ip1list1)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','put 1')

      istat = vgrid_wb_exists(name1_S)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','exists')

      istat = vgrid_wb_put(name1_S,p_myvgrid1,ip1list1)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','put duplicate')

      istat = vgrid_wb_put(name2_S,p_myvgrid2,ip1list2)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','put 2')


      istat = vgrid_wb_put(trim(name2_S)//'ref',p_myvgrid2,ip1list2,'P1234567890:P')
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','put 2ref')


      ip1list => ip1list1(1:1)
      istat = vgrid_wb_put('vground',VGRID_GROUND_TYPE,ip1list)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','put VGRID_GROUND_TYPE')

      ip1list => ip1list1(1:2)
      istat = vgrid_wb_put('vsurf',VGRID_SURF_TYPE,ip1list)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','put VGRID_SURF_TYPE')
      ! ---------------------------------------------------------------------
      return
   end subroutine test_vgrid_wb_put


   !/@
   subroutine test_vgrid_wb_get()
      implicit none
      !@objective 
      !@author Stephane Chamberland, 2012-01
      !@/

      character(len=512) :: name1_S,name2_S,sfcfld_S
      logical :: ok_L
      integer :: istat,nbip1,type

      integer,pointer,dimension(:) :: ip1_m,ip1_t,ip1_m1,ip1_t1,ip1_m2,ip1_t2,&
           ip1list,ip1list1,ip1list2

      type(vgrid_descriptor),target :: myvgrid,myvgrid1,myvgrid2
      type(vgrid_descriptor),pointer :: p_myvgrid,p_myvgrid1,p_myvgrid2
      ! ---------------------------------------------------------------------
      call priv_vgrid_wb_defvgrid(1,name1_S,myvgrid1,p_myvgrid1,ip1_m1,ip1_t1,ip1list1)
      call priv_vgrid_wb_defvgrid(2,name2_S,myvgrid2,p_myvgrid2,ip1_m2,ip1_t2,ip1list2)

      nullify(ip1list)
      istat = vgrid_wb_get(' ',myvgrid,ip1list)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','get no name')

      nullify(ip1list)
      istat = vgrid_wb_get('vnone',myvgrid,ip1list)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','get not found')

      allocate(ip1list(1:2),stat=istat)
      istat = vgrid_wb_get(name1_S,myvgrid,ip1list)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','get ip1list too short')

      allocate(ip1list(50:100),stat=istat)
      istat = vgrid_wb_get(name1_S,myvgrid,ip1list)
      call testutils_assert_ok(.not.RMN_IS_OK(istat),'test_vgrid_wb','get ip1list lbound')

      allocate(ip1list(0:100),stat=istat)
      nbip1 = vgrid_wb_get(name1_S,myvgrid,ip1list)
      call testutils_assert_ok(RMN_IS_OK(nbip1),'test_vgrid_wb','get')
      istat = vgd_get(myvgrid,key='VIPM',value=ip1_m)
      istat = min(vgd_get(myvgrid,key='VIPT',value=ip1_t),istat)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_CRITICAL,'(test_vgrid_wb) vgd_get vip')
         return
      endif
      ok_L = (all(ip1_m==ip1_m1).and.all(ip1_t==ip1_t1))
      call testutils_assert_ok(ok_L,'test_vgrid_wb','get - vgrid values')
      ok_L = (all(ip1list(0:nbip1-1)==ip1list1(0:nbip1-1)))
      call testutils_assert_ok(ok_L,'test_vgrid_wb','get - ip1 values')

      istat = vgrid_wb_get(name2_S,myvgrid)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_vgrid_wb','get no ip1')
      istat = vgd_get(myvgrid,key='VIPM',value=ip1_m)
      istat = min(vgd_get(myvgrid,key='VIPT',value=ip1_t),istat)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_CRITICAL,'(test_vgrid_wb) vgd_get vip')
         return
      endif
      ok_L = (all(ip1_m==ip1_m2).and.all(ip1_t==ip1_t2))
      call testutils_assert_ok(ok_L,'test_vgrid_wb','get no ip1 - vgrid values')

      nullify(ip1list)
      nbip1 = vgrid_wb_get(name2_S,myvgrid,ip1list,type,sfcfld_S)
      call testutils_assert_ok(RMN_IS_OK(nbip1),'test_vgrid_wb','get allocate')
      call testutils_assert_eq(type,VGRID_UPAIR_TYPE,'get type')
      call testutils_assert_eq(sfcfld_S,'P0','get sfcfld')
      istat = vgd_get(myvgrid,key='VIPM',value=ip1_m)
      istat = min(vgd_get(myvgrid,key='VIPT',value=ip1_t),istat)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_CRITICAL,'(test_vgrid_wb) vgd_get vip')
         return
      endif
      ok_L = (all(ip1_m==ip1_m2).and.all(ip1_t==ip1_t2))
      call testutils_assert_ok(ok_L,'test_vgrid_wb','get alloate - vgrid values')
      ok_L = (all(ip1list(1:nbip1)==ip1list2(1:nbip1)))
      call testutils_assert_ok(ok_L,'test_vgrid_wb','get alloate - ip1 values')


      nullify(ip1list)
      nbip1 = vgrid_wb_get(trim(name2_S)//'ref',myvgrid,ip1list,type,sfcfld_S)
      call testutils_assert_ok(RMN_IS_OK(nbip1),'test_vgrid_wb','get allocate ref')
      call testutils_assert_eq(type,VGRID_UPAIR_TYPE,'get type ref')
      call testutils_assert_eq(sfcfld_S,'P1234567890:P','get sfcfld ref')


      nullify(ip1list)
      nbip1 = vgrid_wb_get('vground',myvgrid,ip1list,type)
      call testutils_assert_ok(RMN_IS_OK(nbip1),'test_vgrid_wb','get VGRID_GROUND_TYPE')
      call testutils_assert_eq(type,VGRID_GROUND_TYPE,'get VGRID_GROUND_TYPE type')

      nullify(ip1list)
      nbip1 = vgrid_wb_get('vsurf',myvgrid,ip1list,type)
      call testutils_assert_ok(RMN_IS_OK(nbip1),'test_vgrid_wb','get VGRID_SURF_TYPE')
      call testutils_assert_eq(type,VGRID_SURF_TYPE,'get VGRID_SURF_TYPE type')
      ! ---------------------------------------------------------------------
      return
   end subroutine test_vgrid_wb_get


   !==== Private functions ==================================================


   !/@
   subroutine priv_vgrid_wb_defvgrid(F_id,F_name_S,F_vgrid,p_vgrid,F_ip1_m,F_ip1_t,F_ip1list)
      integer,intent(in) :: F_id
      character(len=*),intent(out) :: F_name_S
      type(vgrid_descriptor),intent(out),target :: F_vgrid
      type(vgrid_descriptor),pointer :: p_vgrid
      integer,pointer :: F_ip1_m(:),F_ip1_t(:),F_ip1list(:)

      !@/
      real,parameter :: hyb(5) = (/0.3905587,0.5492443,0.7299818,0.8791828,0.9950425/)
      integer :: istat
      real(8) :: ptop_8
      ! ---------------------------------------------------------------------
      IF_1: if (F_id == 1) then
         F_name_S = 'v1'
         ptop_8 = 9575.0d0
      else ! IF_1
         F_name_S = 'v2'
         ptop_8 = 9500.0d0
      endif IF_1

      istat = vgd_new(F_vgrid, &
           kind     = VGRID_HYBS_KIND, &
           version  = VGRID_HYBS_VER, &
           hyb      = hyb, &
           ptop_8   = ptop_8, &
           pref_8   = 100000.d0, &
           rcoef1   = 1., &
           rcoef2   = 1.)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_CRITICAL,'(test_vgrid_wb) vgd_new')
         return
      endif
      p_vgrid => F_vgrid

      nullify(F_ip1_m,F_ip1_t)
      istat = vgd_get(F_vgrid,key='VIPM',value=F_ip1_m)
      istat = min(vgd_get(F_vgrid,key='VIPT',value=F_ip1_t),istat)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_CRITICAL,'(test_vgrid_wb) vgd_get vip')
         return
      endif

      !   print *,'ip1_m1',lbound(ip1_m1,1),ubound(ip1_m1,1),size(ip1_m1),';',ip1_m1(2),ip1_m1(size(ip1_m1))

      IF_2: if (F_id == 1) then
         allocate(F_ip1list(0:size(F_ip1_m)-2),stat=istat)
         F_ip1list(0:size(F_ip1_m)-2) = F_ip1_m(2:size(F_ip1_m))
      else ! IF_2
         allocate(F_ip1list(1:size(F_ip1_t)-2),stat=istat)
         F_ip1list = F_ip1_t(2:size(F_ip1_t)-1)
      endif IF_2

      !   print *,'ip1list2',lbound(ip1list2,1),ubound(ip1list2,1),size(ip1list2),';',ip1list2(1),ip1list2(size(ip1_t2)-2)
      ! ---------------------------------------------------------------------
      return
   end subroutine priv_vgrid_wb_defvgrid

end module test_vgrid_wb_mod


!/@
subroutine test_vgrid_wb()
   use testutils
   use test_vgrid_wb_mod
   implicit none
   !@objective 
   !@author Stephane Chamberland, 2012-01
!@/
   ! ---------------------------------------------------------------------
!!$   myproc = testutils_initmpi()
   call testutils_verbosity()
   call testutils_set_name('test_vgrid_wb')
   !call msg_set_minMessageLevel(MSG_DEBUG)
   !call msg_set_minMessageLevel(MSG_WARNING)
   !call msg_set_minMessageLevel(MSG_ERROR)
   !call msg_set_minMessageLevel(MSG_CRITICAL)
   
   call test_vgrid_wb_put()
   call test_vgrid_wb_get()

!!$   call rpn_comm_finalize(istat)
   ! ---------------------------------------------------------------------
   return
end subroutine test_vgrid_wb

