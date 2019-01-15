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

!/@
module fst_read_mod
   use vGrid_Descriptors
   use cmcdate_mod
   implicit none
   private
   !@objective 
   !@author  Stephane Chamberland, 2011-04
   !@description
   ! Public functions
   public :: fst_find,fst_read,fst_get_hgridid,fst_get_gridid,fst_get_vgrid,fst_getmeta
   ! Public constants
   integer,parameter,public :: FST_FIND_LT    = -2 !find nearest datev .lt. F_datev
   integer,parameter,public :: FST_FIND_LE    = -1 !find nearest datev .le. F_datev
   integer,parameter,public :: FST_FIND_NEAR  = 0  !find nearest datev
   integer,parameter,public :: FST_FIND_GE    = 1  !find nearest datev .ge. F_datev
   integer,parameter,public :: FST_FIND_GT    = 2  !find nearest datev .gt. F_datev
!@/

#include <rmnlib_basics.hf>
#include <arch_specific.hf>

   interface fst_get_gridid !# Name kept for backward compatibility
      module procedure fst_get_hgridid
   end interface

   interface fst_read
      module procedure fst_read_3d_r4
   end interface

   integer,parameter :: FST_MIN_TIME_RES = 40  !fstinf min time/datev (sec) resolution in search
   integer,parameter :: FST_MIN_TIME_RES_PRE80 = 1840  !fstinf min time/datev (sec) resolution in search
   real(RDOUBLE),parameter :: SEC_PER_HR = 3600.d0

contains

   !/@
   function fst_find(F_unf,F_nomvar,F_datev,F_ip1,F_ip2,F_ip3, &
        F_datevfuzz,F_fuzzopr,F_typvar_S) result(F_key)
      implicit none
      !@objective Try to find rec key for given field params
      !@arguments
      character(len=*),intent(in) :: F_nomvar
      integer,intent(inout) :: F_datev
      integer,intent(in) :: F_unf,F_ip1,F_ip2,F_ip3
      integer,intent(in),optional :: F_datevfuzz,F_fuzzopr
      character(len=*),intent(in),optional :: F_typvar_S
      !@return
      integer :: F_key
      !@author
      !@revision
      !@/
      integer,parameter :: NMAX = 9999
!!$      character(len=12) :: dummy_S
      character(len=1) :: grtyp_S
      character(len=2) :: typvar_S,typvar1_S
      character(len=4) :: nomvar_S
      character(len=12):: etiket_S
      character(len=128) :: msg_S
      integer :: i,istat,keylist(NMAX),nkeys,keylist2(NMAX),nkeys2,dist,dist0,&
           datevfuzz,fuzzopr,min_time_res
      integer :: ni1,nj1,nk1, datev, datev0, searchdate, &
           dateo,deet,npas,nbits, datyp, ip1, ip2, ip3, &
           ig1, ig2, ig3, ig4, swa, lng, dltf, ubc, extra1, extra2, extra3
!!$      real :: zp1
      real(RDOUBLE) :: nhours_8
      !---------------------------------------------------------------------
      nomvar_S = F_nomvar
      write(msg_S,'(a,i12,a,3i10,a)') '(fst) Find, looking for: '//nomvar_S(1:4)//' [datev=',F_datev,'] [ip123=',F_ip1,F_ip2,F_ip3,']'; call flush(6)
      call msg(MSG_DEBUG,msg_S)
      F_key = RMN_ERR
      if (F_unf <= 0) return
      datevfuzz = 0
      fuzzopr = FST_FIND_NEAR
      typvar_S = RMN_ANY_TYP
      if (present(F_datevfuzz)) datevfuzz = F_datevfuzz
      if (present(F_fuzzopr)) fuzzopr = F_fuzzopr
      if (present(F_typvar_S)) typvar_S = F_typvar_S

      searchdate = F_datev
      min_time_res = FST_MIN_TIME_RES
      if (cmcdate_year(searchdate) <= 1980) &
           min_time_res = FST_MIN_TIME_RES_PRE80
      select case(fuzzopr)
      case(FST_FIND_GT)
         nhours_8 = dble(min_time_res)/SEC_PER_HR
         call incdatr(searchdate,F_datev,nhours_8)
         fuzzopr = FST_FIND_GE
      case(FST_FIND_LT)
         nhours_8 = dble(-min_time_res)/SEC_PER_HR
         call incdatr(searchdate,F_datev,nhours_8)
         fuzzopr = FST_FIND_LE
      end select

      ip1 = F_ip1
      if (ip1 == 1200) ip1 = 0
      if (ip1 <= 0 ) then
         F_key = fstinf(F_unf,ni1,nj1,nk1,searchdate,RMN_ANY_ETK,  &
              ip1,F_ip2,F_ip3,typvar_S,F_nomvar)
         if (RMN_IS_OK(F_key)) then
            F_datev = searchdate
            write(msg_S,'(a,i12,a,3i10,a,i12,a)') '(fst) Found: '//nomvar_S(1:4)//' [datev=',F_datev,'] [ip123=',F_ip1,F_ip2,F_ip3,'] [key=',F_key,']'
            call msg(MSG_DEBUG,msg_S)
            return
         endif
      endif

      ip1 = F_ip1
      if (ip1 == 0) ip1 = 1200
      if (F_ip1 >= 0) then
!!$         call convip(ip1, zp1, kind, RMN_CONV_IP2P, dummy_S, .not.RMN_CONV_USEFORMAT_L)
!!$         ip1 = ip1_all(zp1,kind)
         ip1 = priv_ip1_all(ip1)
         F_key = fstinf(F_unf,ni1,nj1,nk1,searchdate,RMN_ANY_ETK,  &
              ip1,F_ip2,F_ip3,typvar_S,F_nomvar)
!!$            F_key = fstinf(F_unf,ni1,nj1,nk1,searchdate,RMN_ANY_ETK,  &
!!$                 ip1_all(zp1,kind),F_ip2,F_ip3,typvar_S,F_nomvar)
      endif

      if (RMN_IS_OK(F_key) .or. datevfuzz <= 0) then
         F_datev = searchdate
         return
      endif

      nkeys = 0
      if (F_ip1 >= 0) then
!!$         ip1 = ip1_all(zp1,kind)
         ip1 = priv_ip1_all(ip1)
         istat = fstinl(F_unf,ni1,nj1,nk1,RMN_ANY_DATE,RMN_ANY_ETK,  &
              ip1,F_ip2,F_ip3,typvar_S,F_nomvar, &
              keylist,nkeys,NMAX)
         if (F_ip1 == 0) then
            ip1 = F_ip1
            istat = fstinl(F_unf,ni1,nj1,nk1,RMN_ANY_DATE,RMN_ANY_ETK,  &
                 ip1,F_ip2,F_ip3,typvar_S,F_nomvar, &
                 keylist2,nkeys2,NMAX)
            do i=1,nkeys2
               if (nkeys >= NMAX) exit
               nkeys = nkeys + 1
               keylist(nkeys) = keylist2(i)
            enddo
         endif
      else
         ip1 = F_ip1
         istat = fstinl(F_unf,ni1,nj1,nk1,RMN_ANY_DATE,RMN_ANY_ETK,  &
              ip1,F_ip2,F_ip3,typvar_S,F_nomvar, &
              keylist,nkeys,NMAX)
      endif
      if (.not.RMN_IS_OK(istat) .or. nkeys == 0 .or. nkeys == NMAX) then
         write(msg_S,'(a,i12,a,3i10,a,i12,a)') '(fst) Not Found: '//nomvar_S(1:4)//' [datev=',F_datev,'] [ip123=',F_ip1,F_ip2,F_ip3,'] [key=',F_key,']'
         call msg(MSG_DEBUG,msg_S)
         return
      endif

      dist = datevfuzz
      if (datevfuzz /= huge(datevfuzz)) dist = datevfuzz+min_time_res
      do i=1,nkeys
         istat = fstprm(keylist(i), dateo,deet,npas, ni1,nj1,nk1, &
              nbits, datyp, ip1, ip2, ip3, &
              typvar1_S, nomvar_S, etiket_S, &
              grtyp_S, ig1, ig2, ig3, ig4, swa, lng, dltf, &
              ubc, extra1, extra2, extra3)
         if (.not.RMN_IS_OK(istat)) then
            F_key = RMN_ERR
            call msg(MSG_WARNING,'(fst_find) problem getting record meta')
            return
         endif
         nhours_8 = (DBLE(deet)*DBLE(npas))/SEC_PER_HR
         call incdatr(datev0,dateo,nhours_8)
         call difdatr(searchdate,datev0,nhours_8)
         dist0 = nint(real(nhours_8*SEC_PER_HR))
         select case(fuzzopr)
         case(FST_FIND_GE)
            dist0 = -dist0
         case(FST_FIND_NEAR)
            dist0 = abs(dist0)
         !case(FST_FIND_LE)
         end select
         if (dist0 >= 0 .and. dist0 < dist) then
            F_key = keylist(i)
            dist  = dist0
            datev = datev0
         endif
      end do

      if (RMN_IS_OK(F_key)) then
         F_datev = datev
         write(msg_S,'(a,i12,a,3i10,a,i12,a)') '(fst) Found: '//nomvar_S(1:4)//' [datev=',F_datev,'] [ip123=',F_ip1,F_ip2,F_ip3,'] [key=',F_key,']'
      else
         write(msg_S,'(a,i12,a,3i10,a,i12,a)') '(fst) Not Found: '//nomvar_S(1:4)//' [datev=',F_datev,'] [ip123=',F_ip1,F_ip2,F_ip3,'] [key=',F_key,']'
      endif
      call msg(MSG_DEBUG,msg_S)
      !---------------------------------------------------------------------
      return
   end function fst_find


   !/@
   function fst_read_3d_r4(F_key,F_data,F_fileid,F_gridid,&
        F_nomvar_S,F_etiket_S,F_dateo,F_deet,F_npas,F_ip1,F_ip2,F_ip3,&
        F_typvar_S) result(F_istat)
      implicit none
      !@objective 
      !@arguments
      integer,intent(in) :: F_key
      real,pointer :: F_data(:,:,:)
      integer,intent(in),optional :: F_fileid
      integer,intent(out),optional :: F_gridid
      character(len=*),intent(out),optional :: F_nomvar_S, F_etiket_S,F_typvar_S
      integer,intent(out),optional :: F_dateo,F_deet,F_npas,F_ip1, F_ip2, F_ip3
      !@author
      !@return
      integer :: F_istat
      !@/
      integer :: istat
      character(len=2) :: typvar_S
      character(len=12) :: etiket_S
      character(len=4) :: nomvar_S
      character(len=2) :: grtyp_S
      integer :: ni1,nj1,nk1, &
           dateo,deet,npas,nbits,datyp,ip1,ip2,ip3,&
           ig1, ig2, ig3, ig4, swa, lng, dltf, ubc, extra1, extra2, extra3
      ! ---------------------------------------------------------------------
      F_istat = RMN_ERR
      if (present(F_gridid)) F_gridid = RMN_ERR
      if (F_key < 0) return

      istat = fstprm(F_key, dateo,deet,npas, ni1,nj1,nk1, &
           nbits, datyp, ip1, ip2, ip3, &
           typvar_S, nomvar_S, etiket_S, &
           grtyp_S(1:1), ig1, ig2, ig3, ig4, swa, lng, dltf, &
           ubc, extra1, extra2, extra3)
      if (.not.RMN_IS_OK(istat) .or. ni1<1 .or. nj1<1 .or. nk1<1) then
         call msg(MSG_WARNING,'(fst_read) Cannot get field dims')
         return
      endif
!!$      print *,'(fst_read) found:',trim(nomvar_S),ni1,nj1,':',ip1, ip2, ip3,':',grtyp_S(1:1),ig1, ig2, ig3, ig4
     
      if (.not.associated(F_data)) then
         allocate(F_data(ni1,nj1,nk1),stat=istat)
         if (istat /= 0) return
      endif
      if (size(F_data,1) /= ni1 .or. &
           size(F_data,2) /= nj1  .or. &
           size(F_data,3) /= nk1) then
         call msg(MSG_WARNING,'(fst_read) Fields dims in file != provided array dims')
         return
      endif

      F_istat = fstluk(F_data,F_key,ni1,nj1,nk1)

      if (present(F_gridid).and.present(F_fileid)) then
!!$         print *,'(fst_read) looking for gridL:',grtyp_S(1:1),ig1,ig2,ig3,ig4 ; call flush(6)
         if (any(grtyp_S(1:1) == (/'u','U'/))) then
            ni1 = -1 ; nj1 = -1
         endif
         F_gridid = ezqkdef(ni1,nj1, grtyp_S(1:1), ig1, ig2, ig3, ig4, F_fileid)
      endif
      if (present(F_nomvar_S)) F_nomvar_S = nomvar_S
      if (present(F_etiket_S)) F_etiket_S = etiket_S
      if (present(F_typvar_S)) F_typvar_S = typvar_S
      if (present(F_dateo)) F_dateo = dateo
      if (present(F_deet)) F_deet = deet
      if (present(F_npas)) F_npas = npas
      if (present(F_ip1)) F_ip1 = ip1
      if (present(F_ip2)) F_ip2 = ip2
      if (present(F_ip3)) F_ip3 = ip3
      ! ---------------------------------------------------------------------
      return
   end function fst_read_3d_r4


   !/@
   function fst_getmeta(F_key,F_nomvar_S,F_dateo,F_deet,F_npas,&
        F_ip1, F_ip2, F_ip3, F_etiket_S,F_typvar_S) result(F_istat)
      implicit none
      !@objective 
      !@arguments
      integer,intent(in) :: F_key
      character(len=*),intent(out),optional :: F_nomvar_S, F_etiket_S,F_typvar_S
      integer,intent(out),optional :: F_dateo,F_deet,F_npas,F_ip1, F_ip2, F_ip3
      !@author
      !@return
      integer :: F_istat
      !@/
      integer :: istat
      !wrong !!character(len=32) :: typvar_S,nomvar_S, etiket_S, grtyp_S
      character(len=1) :: grtyp_S
      character(len=2) :: typvar_S
      character(len=4) :: nomvar_S
      character(len=12):: etiket_S
      integer :: ni1,nj1,nk1, &
           dateo,deet,npas,nbits,datyp,ip1,ip2,ip3,&
           ig1, ig2, ig3, ig4, swa, lng, dltf, ubc, extra1, extra2, extra3
      ! ---------------------------------------------------------------------
      F_istat = RMN_ERR
      if (F_key < 0) return

      istat = fstprm(F_key, dateo,deet,npas, ni1,nj1,nk1, &
           nbits, datyp, ip1, ip2, ip3, &
           typvar_S, nomvar_S, etiket_S, &
           grtyp_S, ig1, ig2, ig3, ig4, swa, lng, dltf, &
           ubc, extra1, extra2, extra3)
      if (.not.RMN_IS_OK(istat) .or. ni1<1 .or. nj1<1 .or. nk1<1) return

      if (present(F_nomvar_S)) F_nomvar_S = nomvar_S
      if (present(F_etiket_S)) F_etiket_S = etiket_S
      if (present(F_typvar_S)) F_typvar_S = typvar_S
      if (present(F_dateo)) F_dateo = dateo
      if (present(F_deet)) F_deet = deet
      if (present(F_npas)) F_npas = npas
      if (present(F_ip1)) F_ip1 = ip1
      if (present(F_ip2)) F_ip2 = ip2
      if (present(F_ip3)) F_ip3 = ip3
      ! ---------------------------------------------------------------------
      return
   end function fst_getmeta


   !/@
   function fst_get_hgridid(F_fileid,F_key) result(F_gridid)
      implicit none
      !@objective 
      !@arguments
      integer,intent(in) :: F_fileid,F_key
      !@author
      !@return
      integer :: F_gridid
      !@/
      integer :: istat
      character(len=1) :: grtyp_S
      character(len=2) :: typvar_S
      character(len=4) :: nomvar_S
      character(len=12):: etiket_S
      integer :: ni1,nj1,nk1, &
           dateo,deet,npas,nbits,datyp,ip1,ip2,ip3,&
           ig1, ig2, ig3, ig4, swa, lng, dltf, ubc, extra1, extra2, extra3
      ! ---------------------------------------------------------------------
      F_gridid = RMN_ERR
      if (F_fileid <= 0 .or. F_key < 0) return

      istat = fstprm(F_key, dateo,deet,npas, ni1,nj1,nk1, &
           nbits, datyp, ip1, ip2, ip3, &
           typvar_S, nomvar_S, etiket_S, &
           grtyp_S, ig1, ig2, ig3, ig4, swa, lng, dltf, &
           ubc, extra1, extra2, extra3)

      if (any(grtyp_S(1:1) == (/'u','U'/))) then
         ni1 = -1 ; nj1 = -1
      endif
      F_gridid = ezqkdef(ni1,nj1, grtyp_S, ig1, ig2, ig3, ig4, F_fileid)
      ! ---------------------------------------------------------------------
      return
   end function fst_get_hgridid


   !/@
   function fst_get_vgrid(F_fileid,F_key,F_vgrid,F_ip1list,F_lvltyp_S) result(F_istat)
      implicit none
      !@objective 
      !@arguments
      integer,intent(in) :: F_fileid,F_key
      type(vgrid_descriptor),intent(out) :: F_vgrid
      integer,pointer :: F_ip1list(:)
      character(len=*),intent(out) :: F_lvltyp_S
      !@author Ron McTaggartCowan, Aug 2012
      !@return
      integer :: F_istat
      !@/
      integer,parameter :: NMAX = 9999
      integer :: istat,keylist(NMAX),nkeys,k
      character(len=1) :: grtyp_S
      character(len=2) :: typvar_S
      character(len=4) :: nomvar_S
      character(len=12):: etiket_S
      integer :: ni1,nj1,nk1, &
           dateo,datev,deet,npas,nbits,datyp,ip1,ip2,ip3,&
           ig1, ig2, ig3, ig4, swa, lng, dltf, ubc, extra1, extra2, extra3
      ! ---------------------------------------------------------------------
      F_istat = RMN_ERR
      if (F_fileid <= 0 .or. F_key < 0) return

      istat = fstprm(F_key, dateo,deet,npas, ni1,nj1,nk1, &
           nbits, datyp, ip1, ip2, ip3, &
           typvar_S, nomvar_S, etiket_S, &
           grtyp_S, ig1, ig2, ig3, ig4, swa, lng, dltf, &
           ubc, extra1, extra2, extra3)
      if (.not.RMN_IS_OK(istat)) return

      istat = vgd_new(F_vgrid,unit=F_fileid,ip1=ig4,ip2=-1)
      if (.not.RMN_IS_OK(istat)) &
           istat = vgd_new(F_vgrid,unit=F_fileid,ip1=ig1,ip2=ig2)
      if (.not.RMN_IS_OK(istat)) &
           istat = vgd_new(F_vgrid,unit=F_fileid,ip1=-1,ip2=-1)
      if (.not.RMN_IS_OK(istat)) return

      F_lvltyp_S = 'M'
      istat = vgd_get(F_vgrid,'VIP'//trim(F_lvltyp_S),F_ip1list)
      if (.not.(RMN_IS_OK(istat) .and. associated(F_ip1list))) return
      if (any(ip1 == F_ip1list)) then
         F_istat = size(F_ip1list)
         return
      endif

      F_lvltyp_S = 'T'
      istat = vgd_get(F_vgrid,'VIP'//trim(F_lvltyp_S),F_ip1list)
      if (.not.(RMN_IS_OK(istat) .and. associated(F_ip1list))) return
      if (any(ip1 == F_ip1list)) then
         F_istat = size(F_ip1list)
         return
      endif

      F_lvltyp_S = 'SFC'
      call incdatr(datev,dateo,(dble(deet)*dble(npas))/SEC_PER_HR)
      istat = fstinl(F_fileid,ni1,nj1,nk1,datev,'',ip1,-1,-1,'',nomvar_S,keylist,nkeys,size(keylist))
      if (.not.RMN_IS_OK(istat) .or. nkeys < 1) return
      if (associated(F_ip1list)) deallocate(F_ip1list, stat=istat)
      allocate(F_ip1list(nkeys),stat=istat)
      if (.not.associated(F_ip1list)) return
      istat = RMN_OK
      do k=1,nkeys
         istat = min(fstprm(F_key, dateo,deet,npas, ni1,nj1,nk1, &
              nbits, datyp, F_ip1list(k), ip2, ip3, &
              typvar_S, nomvar_S, etiket_S, &
              grtyp_S, ig1, ig2, ig3, ig4, swa, lng, dltf, &
              ubc, extra1, extra2, extra3),istat)
      enddo
      if (.not.RMN_IS_OK(istat)) then
         if (associated(F_ip1list)) deallocate(F_ip1list, stat=istat)
         return
      endif
      !TODO: check that it's sfc (kind 3) levels
      !TODO: sort_unique
      F_istat = size(F_ip1list)
      ! ---------------------------------------------------------------------
      return
   end function fst_get_vgrid


   !==== Private Functions =================================================


   function priv_ip1_all(F_ip1) result(F_ip1out)
      implicit none
      integer,intent(in) :: F_ip1
      integer :: F_ip1out
!!$      integer,parameter :: NMAXIP1ALL = 1024
!!$      integer,save :: m_ip1all(NMAXIP1ALL,2) = -1
!!$      integer,save :: m_nip1all = 0
!!$      integer :: nn
      character(len=12) :: dummy_S = ' '
      integer :: ip1, ikind
      real :: zp1
      ! -----------------------------------------------------------------
!!$      do nn=1,m_nip1all
!!$         if (m_ip1all(nn,1) == F_ip1) then
!!$            F_ip1out = m_ip1all(nn,2)
!!$            return
!!$         endif
!!$      enddo
      ip1 = F_ip1
      call convip(ip1, zp1, ikind, RMN_CONV_IP2P, dummy_S, .not.RMN_CONV_USEFORMAT_L)
      F_ip1out = ip1_all(zp1,ikind)
!!$      if (m_nip1all < NMAXIP1ALL)  then
!!$         m_nip1all = m_nip1all + 1
!!$         m_ip1all(m_nip1all,1) = F_ip1
!!$         m_ip1all(m_nip1all,2) = F_ip1out
!!$         if (all(m_ip1all(1:m_nip1all,1) /= F_ip1out) &
!!$              .and. m_nip1all < NMAXIP1ALL)  then
!!$            m_nip1all = m_nip1all + 1
!!$            m_ip1all(m_nip1all,1) = F_ip1out
!!$            m_ip1all(m_nip1all,2) = F_ip1out
!!$         endif
!!$      endif
      ! -----------------------------------------------------------------
      return
   end function priv_ip1_all

end module fst_read_mod
