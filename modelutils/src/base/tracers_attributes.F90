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
!** s/r tracers_attributes
      subroutine tracers_attributes2 (F_attributes_S, F_wload, F_hzd,&
                                             F_monot, F_massc, F_min )
      implicit none
#include <arch_specific.hf>
      character*(*) F_attributes_S
      integer F_wload, F_hzd, F_monot, F_massc
      real F_min

      character(len=2048) :: attributes
      logical change_default_L
      integer ind,deb,eqi,indvarn
      integer, save :: default_wload= 0
      integer, save :: default_hzd  = 0
      integer, save :: default_monot= 1
      integer, save :: default_massc= 0
      real   , save :: default_vmin = -1*huge(1.)
!
!     ---------------------------------------------------------------
!
      F_wload= default_wload ; F_hzd  = default_hzd
      F_monot= default_monot ; F_massc= default_massc
      F_min  = default_vmin

      if (trim(F_attributes_S) == '') return
      ind=0
      call low2up ( F_attributes_S, attributes )

      change_default_L = .false.
 44   if (ind .gt. len(attributes)) return

      deb= ind+1
      ind= index(attributes(deb:),",")
      if (ind .eq. 0) then
         ind= len(attributes) + 1
      else
         ind= ind + deb - 1
      endif

      if (trim(attributes(deb:ind-1)) == 'DEFAULT') then
         change_default_L = .true.
         goto 44
      endif

      eqi= index(attributes(deb:ind-1),"=")

      if (change_default_L) then

         if (eqi .gt. 0) then
            eqi= eqi + deb - 1
            
            if (trim(attributes(deb:eqi-1)) == 'WLOAD') then
               read(attributes(eqi+1:ind-1),*) default_wload
               F_wload=default_wload
            endif
            if (trim(attributes(deb:eqi-1)) == 'HZD'  ) then
               read(attributes(eqi+1:ind-1),*) default_hzd
               F_hzd  =default_hzd
            endif                           
            if (trim(attributes(deb:eqi-1)) == 'MONO' ) then
               read(attributes(eqi+1:ind-1),*) default_monot
               F_monot=default_monot
            endif
            if (trim(attributes(deb:eqi-1)) == 'MASS' ) then
               read(attributes(eqi+1:ind-1),*) default_massc
               F_massc=default_massc
            endif
            if (trim(attributes(deb:eqi-1)) == 'MIN' ) then
               read(attributes(eqi+1:ind-1),*) default_vmin
               F_min  =default_vmin
            endif
         endif

      else

         if (eqi .eq. 0) then
            if (trim(attributes(deb:ind-1)) == 'WLOAD') &
            F_wload=default_wload
            if (trim(attributes(deb:ind-1)) == 'HZD'  ) &
            F_hzd  =default_hzd
            if (trim(attributes(deb:ind-1)) == 'MONO' ) &
            F_monot=default_monot
            if (trim(attributes(deb:ind-1)) == 'MASS' ) &
            F_massc=default_massc
            if (trim(attributes(deb:eqi-1)) == 'MIN'  ) &
            F_min  =default_vmin
         else
            eqi= eqi + deb - 1

            if (trim(attributes(deb:eqi-1)) == 'WLOAD') &
            read(attributes(eqi+1:ind-1),*) F_wload
            if (trim(attributes(deb:eqi-1)) == 'HZD'  ) &
            read(attributes(eqi+1:ind-1),*) F_hzd
            if (trim(attributes(deb:eqi-1)) == 'MONO' ) &
            read(attributes(eqi+1:ind-1),*) F_monot
            if (trim(attributes(deb:eqi-1)) == 'MASS' ) &
            read(attributes(eqi+1:ind-1),*) F_massc
            if (trim(attributes(deb:eqi-1)) == 'MIN' ) &
            read(attributes(eqi+1:ind-1),*) F_min
         endif

      endif

      goto 44
!
!     ---------------------------------------------------------------
!
      return
      end
