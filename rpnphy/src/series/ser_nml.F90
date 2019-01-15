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
function ser_nml2(F_namelistf_S) result(F_istat)
   implicit none
#include <arch_specific.hf>
   !@object: Default configuration and reading series namelist
   !@params
   character(len=*), intent(in) :: F_namelistf_S
   !@return
   integer :: F_istat
   !@author
   !     R. McTaggart-Cowan    - Apr 2009
   !@revisions
   ! v4_05 - McTaggart-Cowan, R. - initial version
   ! v4_50 - Desgagne M.         - rename the routine & other minors mods
   !*@/
#include <msg.h>
#include <rmnlib_basics.hf>
#include <WhiteBoard.hf>
#include <clib_interface_mu.hf>
   include "series.cdk"

   integer, parameter :: SER_NML_ERR  = RMN_ERR
   integer, parameter :: SER_NML_NONE = RMN_OK
   integer, parameter :: SER_NML_OK   = RMN_OK + 1

   integer :: err
   !-------------------------------------------------------------------
   F_istat = SER_NML_ERR

   err = ser_nml_init()
   if (.not.RMN_IS_OK(err)) return
   err = ser_nml_read(F_namelistf_S)
   if (.not.RMN_IS_OK(err)) return
   err = ser_nml_check()
   if (.not.RMN_IS_OK(err)) return
   err = ser_nml_post_init()
   if (.not.RMN_IS_OK(err)) return
   call ser_nml_print()

   F_istat = SER_NML_OK
   !-------------------------------------------------------------------
   return


contains
   
   function ser_nml_init() result(m_istat)
      implicit none
      integer :: m_istat
      integer :: i
      !----------------------------------------------------------------
      m_istat = RMN_OK
      P_serg_srsus_L = .false.

      !# Defaults values for namelist variables
      P_serg_srsrf_s = 'UNDEFINED'
      P_serg_srprf_s = 'UNDEFINED'
      P_serg_srwri   = 1
      P_serg_serstp  = 99999
      do i=1,MAXSTAT
         Xst_stn_latlon(i) =  &
              station_latlon('UNDEFINED',real(STN_MISSING),real(STN_MISSING))
      enddo
      !----------------------------------------------------------------
      return
   end function ser_nml_init


   function ser_nml_read(m_namelist) result(m_istat)
      implicit none
      character(len=*), intent(in) :: m_namelist
      integer :: m_istat
      character(len=1024) :: msg_S, namelist_S, name_S
      integer :: istat, unf
      !----------------------------------------------------------------
      m_istat = RMN_ERR
      name_S = '(ser_nml)'
      namelist_S = '&series'

      unf = 0
      istat = clib_isreadok(m_namelist)
      if (RMN_IS_OK(istat)) istat = fnom(unf, m_namelist, 'SEQ+OLD', 0)
      if (.not.RMN_IS_OK(istat)) then
         write(msg_S,'(a,a,a,a)') trim(name_S), &
              ' Using default config, ', &
              'Namelist file Not found/readable: ', trim(m_namelist)
         call msg(MSG_INFO, msg_S)
         m_istat = RMN_OK
         return
      endif

      read(unf, nml=series, iostat=istat)
      if (istat == 0) then     !# Read ok
         m_istat = RMN_OK + 1
      else if (istat < 0) then !# EOF, nml not found
         write(msg_S,'(a,a,a,a,a)') trim(name_S), &
              ' No Series, Namelist ',&
              trim(namelist_S), ' not available in file: ', trim(m_namelist)
         call msg(MSG_INFO, msg_S)
         m_istat = RMN_OK
      else !# Error
         write(msg_S,'(a,a,a,a,a)') trim(name_S), &
              ' Namelist', trim(namelist_S), &
              ' invalid in file: ', trim(m_namelist)
         call msg(MSG_ERROR, msg_S)
      endif
      istat = fclos(unf)
      !----------------------------------------------------------------
      return
   end function ser_nml_read


   subroutine ser_nml_print()
      implicit none
      integer, external :: msg_getUnit
      integer :: unout, i
      !----------------------------------------------------------------
      unout = msg_getUnit(MSG_INFO)
      if (unout > 0 .and. P_serg_srsus_L) then
         write(unout,*) "&series"
         write(unout,*) "P_serg_serstp =",P_serg_serstp
         write(unout,*) "P_serg_srwri  =",P_serg_srwri
         do i=1,size(P_serg_srprf_s)
            if (trim(P_serg_srprf_s(i)).ne.'UNDEFINED') then
               write (unout,*) "i,P_serg_srprf_s(i)=",i,trim(P_serg_srprf_s(i))
            endif
         enddo
         do i=1,size(P_serg_srsrf_s)
            if (trim(P_serg_srsrf_s(i)).ne.'UNDEFINED') then
               write(unout,*) "i,P_serg_srsrf_s(i)=",i,trim(P_serg_srsrf_s(i))
            endif
         enddo
         write(unout,*) "/"
      endif
      !----------------------------------------------------------------
      return
   end subroutine ser_nml_print


   function ser_nml_check() result(m_istat)
      implicit none
      integer :: m_istat
      !----------------------------------------------------------------
      m_istat = RMN_OK
      !----------------------------------------------------------------
      return
   end function ser_nml_check


   function ser_nml_post_init() result(m_istat)
      implicit none
      integer :: m_istat

      integer, external :: msg_getUnit
      integer :: istat, unout, options, iverb, i, j 
      character(len=1024) :: str512
      !----------------------------------------------------------------
      m_istat = RMN_ERR

      !# Determine if the user has specified any stations
      j = 0
      do while(int(Xst_stn_latlon(j+1)%lat) /= STN_MISSING .and. &
           &   int(Xst_stn_latlon(j+1)%lon) /= STN_MISSING)
         j = j + 1 
         Xst_stn(j)%name   = Xst_stn_latlon(j)%name
         Xst_stn(j)%lat    = Xst_stn_latlon(j)%lat
         Xst_stn(j)%lon    = Xst_stn_latlon(j)%lon
         Xst_stn(j)%i      = STN_MISSING
         Xst_stn(j)%j      = STN_MISSING
         Xst_stn(j)%stcori = STN_MISSING
         Xst_stn(j)%stcorj = STN_MISSING
      enddo
      Xst_nstat = j

      P_serg_srsus_L = .false.
      IF_NSTAT: if (Xst_nstat <= 0) then
         call msg(MSG_INFO,'(ser_nml) NO STATION ARE REQUESTED FOR TIME SERIES.')
      else
         write(str512,'(i5,a)') Xst_nstat, ' STATIONS ARE SPECIFIED FOR TIME SERIES.'
         call msg(MSG_INFO,'(ser_nml) '//str512)

         P_serg_srsrf= 0
         do i=1, cnsrsfm
            if (P_serg_srsrf_s(i) /= 'UNDEFINED') P_serg_srsrf = P_serg_srsrf+1
         end do

         P_serg_srprf= 0
         do i=1, cnsrpfm
            if (P_serg_srprf_s(i) /= 'UNDEFINED') P_serg_srprf = P_serg_srprf+1
         end do

         if (P_serg_srsrf + P_serg_srprf > 0) then
            P_serg_srsus_L = .true.
         else
            call msg(MSG_INFO,'(ser_nml) NO TIME SERIES REQUESTED')
         endif
      endif IF_NSTAT

      options = WB_REWRITE_NONE + WB_IS_LOCAL
      err = wb_put('model/series/P_serg_srsus_L', P_serg_srsus_L, options)

      m_istat = RMN_OK
      !----------------------------------------------------------------
      return
   end function ser_nml_post_init

end function ser_nml2
