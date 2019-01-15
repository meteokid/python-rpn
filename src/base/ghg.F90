!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer,
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer
!version 3 or (at your option) any later version that should be found at:
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software;
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec),
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------

!/@*
module ghg_mod
   use iso_c_binding
   use mu_jdate_mod, only: jdate_year
   use str_mod
   use phy_options
   implicit none
   private

   public :: ghg_init

#include <arch_specific.hf>
#include <msg.h>
#include <rmnlib_basics.hf>
#include <clib_interface_mu.hf>

contains

   function ghg_init(F_path, F_jdateo) result(F_istat)
      implicit none
      character(len=*), intent(in) :: F_path
      integer(IDOUBLE), intent(in) :: F_jdateo
      integer :: F_istat
      
      character(len=256), parameter :: GHG_FILENAME = 'ghg-table-1950-2015_v1'
      character(len=256), parameter :: GHG_VERSION  = 'version=ghg_concentrations_v1'
      !# File format: YYYY co2 n2o ch4 cfc11 cfc12
      character(len=256) :: fullpath_S, string_S, firstyear_S, lastyear_S, parts_S(6)
      integer :: fileid, istat, yyyy, yyyy0
      logical :: version_ok, found_L

      real :: ghg_co2, ghg_n2o, ghg_ch4, ghg_cfc11, ghg_cfc12
      !----------------------------------------------------------------------
      F_istat = RMN_ERR
      
      if (.not.(radia == 'CCCMARAD2' .and. radghg_L)) then
         if (qcfc11 < 0.) qcfc11 = QCFC11_DEFAULT
         if (qcfc12 < 0.) qcfc12 = QCFC12_DEFAULT
         if (qch4 < 0.)   qch4   = QCH4_DEFAULT
         if (qco2 < 0.)   qco2   = QCO2_DEFAULT
         if (qn2o < 0.)   qn2o   = QN2O_DEFAULT
      endif

      if  (qcfc11 >= 0. .and. &
           qcfc12 >= 0. .and. &
           qch4   >= 0. .and. &
           qco2   >= 0. .and. &
           qn2o   >= 0. ) then
         write(string_S, '(a,f9.4,a,f9.4,a,f9.4,a,f9.4,a,f9.4)') &
              'CO2=', qco2, ', N2O=', qn2o, ', CH4=', qch4, &
              ', CFC11=', qcfc11, ', CFC12=', qcfc12
         call msg(MSG_INFO,'(ghg_init) Using: '//trim(string_S))
         F_istat = RMN_OK
         return
      endif

      fullpath_S = trim(F_path)//'/'//trim(GHG_FILENAME)
      istat = clib_isreadok(fullpath_S)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_WARNING,'(ghg_init) File not found or not readable: '//trim(fullpath_S))
         return
      endif
      fileid = 0
      istat = fnom(fileid, fullpath_S, 'SEQ/FMT+R/O+OLD', 0)
      if (.not.RMN_IS_OK(istat) .or. fileid <= 0) then
         call msg(MSG_WARNING,'(ghg_init) Problem opening file: '//trim(fullpath_S))
         return
      endif
      call msg(MSG_INFO,'(ghg_init) Opened: '//trim(fullpath_S))

      yyyy0 = jdate_year(F_jdateo)
      version_ok = .false.
      found_L = .false.
      firstyear_S = ''
      lastyear_S = ''
      DOFILE: do
         read(fileid, '(a)', iostat=istat) string_S
         if (istat /=0) exit DOFILE
         call str_tab2space(string_S)
         string_S = adjustl(string_S)
         istat = clib_tolower(string_S)
         if (string_S(1:1) /= '#' .and. string_S /= ' ') then
            if (.not.version_ok) then
               if (string_S == GHG_VERSION) then
                  version_ok = .true.
                  call msg(MSG_INFOPLUS,'(ghg_init) Verified Version: '//trim(string_S))
                  cycle
               else
                  istat = fclos(fileid)
                  call msg(MSG_WARNING,'(ghg_init) Wrong file version, Expecting'//trim(GHG_VERSION)//', Got: '//trim(string_S))
                  return
               endif
            else
               istat = str_toint(yyyy, string_S(1:5))
               if (.not.RMN_IS_OK(istat)) then
                  call msg(MSG_WARNING,'(ghg_init) Ignoring malformed line: '//trim(string_S))
                  cycle
               endif
               if (firstyear_S == '') firstyear_S = string_S
               lastyear_S = string_S
               if (yyyy == yyyy0) then
                  found_L = .true.
                  exit
               endif
            endif
         else
            call msg(MSG_INFOPLUS,'(ghg_init) Ignoring: '//trim(string_S))
         endif
      enddo DOFILE

      istat = fclos(fileid)
      if (lastyear_S == '') then
         call msg(MSG_WARNING,'(ghg_init) : Did not find any valid data')
         return
      endif
      if (.not.found_L) then
         istat = str_toint(yyyy, firstyear_S(1:5))
         if (yyyy0 <= yyyy) lastyear_S = firstyear_S
         write(string_S,'(i4)') yyyy0
         call msg(MSG_WARNING,'(ghg_init) : Did not find data for year='//trim(string_S)//', using value of year='//lastyear_S(1:5))
      endif
      call str_split2list(parts_S, lastyear_S, ' ', size(parts_S))
      istat = str_toreal(ghg_co2, parts_S(2))
      istat = min(str_toreal(ghg_n2o, parts_S(3)), istat)
      istat = min(str_toreal(ghg_ch4, parts_S(4)), istat)
      istat = min(str_toreal(ghg_cfc11, parts_S(5)), istat)
      istat = min(str_toreal(ghg_cfc12, parts_S(6)), istat)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_WARNING,'(ghg_init) Problem interpreting: '//trim(lastyear_S))
         return
      endif

      if (qcfc11 < 0.) qcfc11 = ghg_cfc11
      if (qcfc12 < 0.) qcfc12 = ghg_cfc12
      if (qch4 < 0.)   qch4   = ghg_ch4
      if (qco2 < 0.)   qco2   = ghg_co2
      if (qn2o < 0.)   qn2o   = ghg_n2o

      write(string_S, '(a,i4,a,f9.4,a,f9.4,a,f9.4,a,f9.4,a,f9.4)') &
           '(', yyyy0, &
           ') CO2=',ghg_co2, ', N2O=', ghg_n2o, ', CH4=', ghg_ch4, &
           ', CFC11=', ghg_cfc11, ', CFC12=', ghg_cfc12
      call msg(MSG_INFOPLUS,'(ghg_init) Found: '//trim(string_S))

      write(string_S, '(a,i4,a,f9.4,a,f9.4,a,f9.4,a,f9.4,a,f9.4)') &
           '(', yyyy0, &
           ') CO2=', qco2, ', N2O=', qn2o, ', CH4=', qch4, &
           ', CFC11=', qcfc11, ', CFC12=', qcfc12
      call msg(MSG_INFO,'(ghg_init) Using: '//trim(string_S))

      F_istat = RMN_OK
      !----------------------------------------------------------------------
      return
   end function ghg_init

end module ghg_mod
