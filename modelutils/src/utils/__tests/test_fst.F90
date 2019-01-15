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

!/@
subroutine test_fst()
   use testutils
   implicit none
   !@objective 
   !@author Stephane Chamberland, 2011-04
!@/
#include <clib_interface_mu.hf>
#include <rmnlib_basics.hf>
   integer :: istat
   character(len=512) :: dfiles_S,bcmk_S
   ! ---------------------------------------------------------------------
   call testutils_verbosity()
   call testutils_set_name('test_fst')
   istat = clib_getenv('ATM_MODEL_DFILES',dfiles_S)
   if (.not.RMN_IS_OK(istat)) then
      print *,'ERROR: ATM_MODEL_DFILES not defined'
      return
   endif
   bcmk_S = trim(dfiles_S)//'/bcmk/'
   call test_fst_open_notfound(bcmk_S)
   call test_fst_find_read(bcmk_S)
!!$   call test_fst_find_fuzz(bcmk_S)
   call test_fst_write()
   ! ---------------------------------------------------------------------
   return
end subroutine test_fst


!/@
subroutine test_fst_open_notfound(F_bcmk_S)
   use fst_mod
   implicit none
   !@objective 
   !@author Stephane Chamberland, 2011-04
   !@argument
   character(len=*), intent(in) :: F_bcmk_S
!@/
#include <rmnlib_basics.hf>
   integer :: funit, istat
   ! ---------------------------------------------------------------------
   funit = fst_open(trim(F_bcmk_S)//'__does_not_exists__',FST_READONLY)
   call testutils_assert_ok(.not.RMN_IS_OK(funit),'test_fst_open_notfound','')
   if (funit > 0) istat = fst_close(funit)
   ! ---------------------------------------------------------------------
   return
end subroutine test_fst_open_notfound


!/@
subroutine test_fst_find_read(F_bcmk_S)
   use fst_mod
   implicit none
   !@objective 
   !@author Stephane Chamberland, 2011-04
   !@argument
   character(len=*), intent(in) :: F_bcmk_S
!@/
#include <rmnlib_basics.hf>
   logical,parameter :: DIR_IS_OK_L = .true.
   real(RDOUBLE),parameter :: SEC_PER_HR = 3600.d0
   integer :: funit, istat,datev,datev2,datev3,key,gridid,datevfuzz, &
        ni, nj, ig1, ig2, ig3, ig4,ip1,kind
   real, pointer :: data(:,:,:)
   character(len=512) :: filename_S,datev_S,dummy_S
   character(len=2) :: grtyp_S
   real :: zp1
   real(RDOUBLE) :: nhours_8
   ! ---------------------------------------------------------------------
   datev_S = '20090427.000000'
   call datp2f(datev,datev_S)
   filename_S = trim(F_bcmk_S)//'2009042700_000'

   funit = fst_open(filename_S,FST_READONLY)
   call testutils_assert_ok(RMN_IS_OK(funit),'test_fst_find_read','fst_open')

   key = fst_find(funit,'NONE',datev,RMN_ANY_I,0,0)
   call testutils_assert_ok(.not.RMN_IS_OK(key),'test_fst_find_read','fst_find not found')

   key = fst_find(funit,'TT',datev,RMN_ANY_I,0,0)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find')

   istat = fst_read(key,data,funit)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_read')
   istat = RMN_ERR
   if (associated(data) .and. &
        size(data,1) == 200 .and. &
        size(data,2) == 100 .and. &
        size(data,3) == 1  .and. &
        nint(minval(data)*100.) == -5089 .and. & 
        nint(maxval(data)*100.) == -2618) istat = RMN_OK
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_read valuse')
   if (associated(data)) deallocate(data,stat=istat)

   gridid = fst_get_gridid(funit,key)
   call testutils_assert_ok(RMN_IS_OK(gridid),'test_fst_find_read','fst_get_gridid')
   istat = ezgprm(gridid, grtyp_S(1:1), ni, nj, ig1, ig2, ig3, ig4)
   istat = RMN_ERR
   if (grtyp_S(1:1) == 'G' .and. ni == 200 .and. nj == 100 .and. &
        ig1+ig2+ig3+ig4 == 0) istat = RMN_OK
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_get_gridid values')
!!$   ier = ezgxprm(gdid, ni, nj, grtyp, ig1, ig2, ig3, ig4, grref, ig1ref, ig2ref, ig3ref, ig4ref)
!!$   ier = ezgfstp(gdid, nomvarx, typvarx, etikx, nomvary, typvary, etiky, ip1, ip2, ip3, dateo, deet, npas, nbits)
!!$   ier = gdgaxes(gdid, ax, ay)
   istat = gdrls(gridid)

   datev_S = '20090427.020000'
   call datp2f(datev2,datev_S)
   datevfuzz = 3600
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz)
   call testutils_assert_ok(.not.RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_near not found')

   datevfuzz = 3600*6

   datev_S = '20090427.020000'
   call datp2f(datev2,datev_S)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_near')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_near value')

   datev_S = '20090427.020000'
   call datp2f(datev2,datev_S)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_LE)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_le')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_le value')

   datev_S = '20090427.020000'
   call datp2f(datev2,datev_S)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_GE)
   call testutils_assert_ok(.not.RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_ge not found')

   datev_S = '20090426.220000'
   call datp2f(datev2,datev_S)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_GE)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_ge')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_ge value')


   datev_S = '20090427.000000'
   call datp2f(datev2,datev_S)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_GT)
   call testutils_assert_ok(.not.RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_gt not found')

   datev_S = '20090427.000000'
   call datp2f(datev2,datev_S)
   datevfuzz = 3600*6
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_LT)
   call testutils_assert_ok(.not.RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_lt not found')

   datev_S = '20090427.000000'
   call datp2f(datev3,datev_S)
   nhours_8 = -40.D0/SEC_PER_HR
   call incdatr(datev2,datev3,nhours_8)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_GT)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_gt as eq')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_gt as eq value')
   call datf2p(datev_S,datev2)


   datev_S = '20090427.000000'
   call datp2f(datev3,datev_S)
   nhours_8 = -1.D0
   call incdatr(datev2,datev3,nhours_8)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_GT)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_gt')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_gt value')


   datev_S = '20090427.000000' 
   call datp2f(datev3,datev_S)
   nhours_8 = 40.D0/SEC_PER_HR
   call incdatr(datev2,datev3,nhours_8)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_LT)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_lt as eq')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_lt as eq value')


   datev_S = '20090427.000000' 
   call datp2f(datev3,datev_S)
   nhours_8 = 1.D0
   call incdatr(datev2,datev3,nhours_8)
   key = fst_find(funit,'TT',datev2,RMN_ANY_I,0,0,datevfuzz,FST_FIND_LT)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find_fuzz_lt')
   call testutils_assert_ok(datev2==datev,'test_fst_find_read','fst_find_fuzz_lt value')
   call datf2p(datev_S,datev2)

   istat = fst_close(funit)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_close')

   !----

   filename_S = trim(F_bcmk_S)//'geophy/Gem_geophy.fst'
   funit = fst_open(filename_S,FST_READONLY)
   call testutils_assert_ok(RMN_IS_OK(funit),'test_fst_find_read2','fst_open')

   datev = RMN_ANY_DATE
   zp1 = 1.
   kind = RMN_CONV_ARBITRARY
   call convip(ip1, zp1, kind, RMN_CONV_P2IPOLD, dummy_S, .not.RMN_CONV_USEFORMAT_L)
   key = fst_find(funit,'J1',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find ip1>0 old')

   datev = RMN_ANY_DATE
   call convip(ip1, zp1, kind, RMN_CONV_P2IPNEW, dummy_S, .not.RMN_CONV_USEFORMAT_L)
   key = fst_find(funit,'J1',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find ip1>0 new')

   datev = RMN_ANY_DATE
   ip1 = 1200
   key = fst_find(funit,'ME',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find ip1=1200')

   datev = RMN_ANY_DATE
   ip1 = 0
   key = fst_find(funit,'ME',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find ip1=0 for 1200')

   datev = RMN_ANY_DATE
   ip1 = 0
   key = fst_find(funit,'MG',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find ip1=0')

   datev = RMN_ANY_DATE
   ip1 = 1200
   key = fst_find(funit,'MG',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find ip1=1200 for 0')

   istat = fst_close(funit)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read2','fst_close')

   !----

   filename_S = trim(F_bcmk_S)
   funit = fst_open(filename_S,FST_READONLY,DIR_IS_OK_L)
   call testutils_assert_ok(RMN_IS_OK(funit),'test_fst_find_read','fst_open dir')
   key = fst_find(funit,'TT',datev,RMN_ANY_I,0,0)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find dir')

   istat = fst_read(key,data,funit)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_read dir')
   istat = RMN_ERR
   if (associated(data) .and. &
        size(data,1) == 200 .and. &
        size(data,2) == 100 .and. &
        size(data,3) == 1  .and. &
        nint(minval(data)*100.) == -5089 .and. & 
        nint(maxval(data)*100.) == -2618) istat = RMN_OK
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_read dir valuse')
   if (associated(data)) deallocate(data,stat=istat)

   gridid = fst_get_gridid(funit,key)
   call testutils_assert_ok(RMN_IS_OK(gridid),'test_fst_find_read','fst_get_gridid dir')
   istat = ezgprm(gridid, grtyp_S(1:1), ni, nj, ig1, ig2, ig3, ig4)
   istat = RMN_ERR
   if (grtyp_S(1:1) == 'G' .and. ni == 200 .and. nj == 100 .and. &
        ig1+ig2+ig3+ig4 == 0) istat = RMN_OK
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read','fst_get_gridid dir values')
!!$   ier = ezgxprm(gdid, ni, nj, grtyp, ig1, ig2, ig3, ig4, grref, ig1ref, ig2ref, ig3ref, ig4ref)
!!$   ier = ezgfstp(gdid, nomvarx, typvarx, etikx, nomvary, typvary, etiky, ip1, ip2, ip3, dateo, deet, npas, nbits)
!!$   ier = gdgaxes(gdid, ax, ay)
   istat = gdrls(gridid)

   datev = RMN_ANY_DATE
   zp1 = 1.
   kind = RMN_CONV_ARBITRARY
   call convip(ip1, zp1, kind, RMN_CONV_P2IPOLD, dummy_S, .not.RMN_CONV_USEFORMAT_L)
   key = fst_find(funit,'J1',datev,ip1,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_find_read','fst_find dir J1 ip1>0 old')

   istat = fst_close(funit)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_find_read2','fst_close dir')
   ! ---------------------------------------------------------------------
   return
end subroutine test_fst_find_read


!/@
subroutine test_fst_write()
   use testutils
   use fst_mod
   use ezgrid_mod
   implicit none
   !@objective 
   !@author Stephane Chamberland, 2012-01
   !@argument
!@/
#include <rmnlib_basics.hf>
#include <clib_interface_mu.hf>
   real,parameter :: MYVALUE = 3.3
   integer,parameter :: NI0=50,NJ0=30,NK0=3
   character(len=256) :: nomvar_S,filename_S
   character(len=2) :: grtyp_S,grtyp2_S,grref_S,grref2_S
   logical :: ok_L,ok2_L
   integer :: funit,istat,gridid,gridid2,lvlid,dateo,deet,npas,key,ig1,ig2,ig3,ig4,ip1,i,j,k,datev,ip1list(NK0),ni,nj,ni2, nj2, ig12, ig22, ig32, ig42,nij(2),nij2(2),ij0(2),ij02(2),ig14(4),ig142(4)
   real,pointer :: data2d(:,:),data3d(:,:,:)
   real :: ax(NI0,1),ay(1,NJ0)
   real,pointer :: ax1(:,:),ax2(:,:),ay1(:,:),ay2(:,:)
   ! ---------------------------------------------------------------------
   filename_S = '__test_fst_to-be-removed__.fst'
   istat = clib_unlink(trim(filename_S))
   funit = fst_open(filename_S)
   call testutils_assert_ok(RMN_IS_OK(funit),'test_fst_write:open','')

   nomvar_S = 'ZX'

   ig1=900 ; ig2=0 ; ig3=43200 ; ig4=43100
   do i=1,NI0
      ax(i,1) = 10.+float(i)*0.25
   enddo
   do j=1,NJ0
      ay(1,j) = float(j)*0.25
   enddo
   gridid = ezgdef_fmem(NI0,NJ0, 'Z',  'E', ig1,ig2,ig3,ig4, ax, ay)

   allocate(data2d(NI0,NJ0),data3d(NI0,NJ0,NK0),stat=istat)
   data2d = MYVALUE
   data3d = MYVALUE
   ip1 = 0
!!$   lvlid = 
   do k=1,NK0
      ip1list(k) = (k+2)*3
   enddo

   nomvar_S = 'ZX2d'
   istat = fst_write(funit,nomvar_S,data2d,gridid,ip1,F_npak=FST_NPAK_FULL32)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_write:write_2d_r4','')

   nomvar_S = 'ZX3d'
   istat = fst_write(funit,nomvar_S,data3d,gridid,ip1list,F_npak=FST_NPAK_FULL32)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_write:write_3d_r4','')

   !TODO: test with a vgrid instead of ip1list

   if (funit > 0) istat = fst_close(funit)
   deallocate(data2d,data3d,stat=istat)

   !- Checking

   funit = fst_open(filename_S,FST_READONLY)
   call testutils_assert_ok(RMN_IS_OK(funit),'test_fst_write:open','')

   nomvar_S = 'ZX2d'
   datev = RMN_ANY_DATE
   key = fst_find(funit,nomvar_S,datev,RMN_ANY_I,RMN_ANY_I,RMN_ANY_I)
   call testutils_assert_ok(RMN_IS_OK(key),'test_fst_write:find','2d')

   istat = fst_read(key,data3d,funit,gridid2)
   call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_write:read','2d')
   ok_L = .false. ; ok2_L = .false.
   if (associated(data3d)) then
      ok_L = all(shape(data3d)==(/NI0,NJ0,1/))
      ok2_L = all(abs(data3d-MYVALUE)<1.e-5)
   endif
   call testutils_assert_ok(ok_L,'test_fst_write:read','data2d shape')
   call testutils_assert_ok(ok2_L,'test_fst_write:read','data2d')
   if (.not.ok2_L) then
      print *,'data2d min,max:',minval(data3d),maxval(data3d)
   endif
   ok_L = ezgrid_samegrid(gridid,gridid2)
   call testutils_assert_ok(ok_L,'test_fst_write:read','data2d grid')
   if (.not.ok_L) then
      print *,'gridid1,2=',gridid,gridid2
      istat = ezgprm(gridid, grtyp_S(1:1), ni, nj, ig1, ig2, ig3, ig4)
      istat = ezgprm(gridid2, grtyp2_S(1:1), ni2, nj2, ig12, ig22, ig32, ig42)
      call testutils_assert_eq(grtyp2_S(1:1),grtyp_S(1:1),'grtyp_S')
      call testutils_assert_eq((/ni2, nj2/),(/ni, nj/),'nij')
      call testutils_assert_eq((/ig12, ig22, ig32, ig42/),(/ig1, ig2, ig3, ig4/),'ig1234')
      istat  = ezgrid_params(gridid,nij,grtyp_S,grref_S,ig14,ij0,ax1,ay1)
      istat  = ezgrid_params(gridid2,nij2,grtyp2_S,grref2_S,ig142,ij02,ax2,ay2)
      call testutils_assert_eq(grtyp2_S(1:1)//grref2_S(1:1),grtyp_S(1:1)//grref_S(1:1),'grtyp_S//grref_S')
      call testutils_assert_eq(grtyp2_S(1:1)//grref2_S(1:1),'ZE','grtyp_S//grref_S ZE')
      call testutils_assert_eq(nij2,nij,'nij')
      call testutils_assert_eq(nij2,(/NI0,NJ0/),'NIJ0')
      call testutils_assert_eq(ig142,ig14,'ig14')
      call testutils_assert_eq(ij02,ij0,'ij0')
      call testutils_assert_eq(ax2(:,1),ax1(:,1),'ax')
      call testutils_assert_eq(ay2(1,:),ay1(1,:),'ay')
      call testutils_assert_eq(ax2(:,1),ax(:,1),'ax0')
      call testutils_assert_eq(ay2(1,:),ay(1,:),'ay0')
   endif


   nomvar_S = 'ZX3d'

   do k=1,NK0
      if (associated(data3d)) deallocate(data3d,stat=istat)
      datev = RMN_ANY_DATE
      key = fst_find(funit,nomvar_S,datev,ip1list(k),RMN_ANY_I,RMN_ANY_I)
      call testutils_assert_ok(RMN_IS_OK(key),'test_fst_write:find','3d')

      istat = fst_read(key,data3d,funit,gridid2)
      call testutils_assert_ok(RMN_IS_OK(istat),'test_fst_write:read','3d')
      ok_L = .false. ; ok2_L = .false.
      if (associated(data3d)) then
         ok_L = all(shape(data3d)==(/NI0,NJ0,1/))
         ok2_L = all(abs(data3d-MYVALUE)<1.e-5)
      endif
      call testutils_assert_ok(ok_L,'test_fst_write:read','data3d shape')
      call testutils_assert_ok(ok2_L,'test_fst_write:read','data3d')
      if (.not.ok2_L) then
         print *,'data3d min,max:',minval(data3d),maxval(data3d)
      endif
      ok_L = ezgrid_samegrid(gridid,gridid2)
      call testutils_assert_ok(ok_L,'test_fst_write:read','data3d grid')
   enddo

   if (funit > 0) istat = fst_close(funit)

   if (associated(data2d)) deallocate(data2d,stat=istat)
   if (associated(data3d)) deallocate(data3d,stat=istat)

   istat = clib_unlink(trim(filename_S))
   ! ---------------------------------------------------------------------
   return
end subroutine test_fst_write
