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

      subroutine e_trinit ()
      implicit none
#include <arch_specific.hf>

!author
!     M. Desgagne -  summer 2014
!
!revision
! v4_70 - Desgagne M.   - initial version

#include "lun.cdk"
      include "tr3d.cdk"

      character*512 varname,attributes
      integer i,j,ind,wload,hzd,monot,massc,dejala,err
!
!     ---------------------------------------------------------------
!
      Tr3d_ntr = 0

      do i=1,MAXTR3D
         if (Tr3d_list_s(i)=='') exit
         ind= index(Tr3d_list_s(i),",")
         if (ind .eq. 0) then
            call low2up(Tr3d_list_s(i), varname)
            attributes = ''
         else
            call low2up(Tr3d_list_s(i)(1:ind-1),varname   )
            call low2up(Tr3d_list_s(i)(ind+1: ),attributes)
         endif
         if (trim(varname)=='HU') cycle
         call tracers_attributes (attributes, wload,hzd,monot,massc)
         dejala=0
         do j=1,Tr3d_ntr
            if (trim(Tr3d_name_S(j))==trim(varname)) dejala=j
         enddo
         if (dejala==0) then
            Tr3d_ntr = Tr3d_ntr + 1
            dejala   = Tr3d_ntr
            Tr3d_name_S(dejala)= trim(varname)
         endif
         Tr3d_hzd (dejala)= (hzd>0) ; Tr3d_wload(dejala)= (wload>0)
         Tr3d_mono(dejala)= monot   ; Tr3d_mass (dejala)= massc
      end do

      if (Lun_out.gt.0) then
         write (Lun_out,1001)
         do i=1,Tr3d_ntr
            write(Lun_out,1002) Tr3d_name_S(i),Tr3d_wload(i),&
                       Tr3d_hzd(i),Tr3d_mono(i),Tr3d_mass(i)
         end do
      endif

 1001 format (/' Final liste of tracers:'/3x,' Name   Wload  Hzd   Mono  Mass')
 1002 format (4x,a4,2l6,2i6)
!
!     ---------------------------------------------------------------
!
      return
      end
