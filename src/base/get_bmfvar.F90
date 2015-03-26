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

!**s/r get_bmfvar - read the dynamics fields from entrance programs
!

!
      subroutine get_bmfvar (F_niu, F_nju, F_niv, F_njv, F_nka_m, F_nka_t)
      implicit none
#include <arch_specific.hf>
!
      integer F_niu, F_nju, F_niv, F_njv, F_nka_m, F_nka_t
!author 
!     Luc Corbeil, mai 2002
!
!revision
! v3_01 - Corbeil L.           - initial version
! v3_10 - Lee V.               - unique bmfscraps...
! v3_11 - Gravel S.            - provide for variable topography
! v3_12 - Dugas B. & Winger K. - read TD in pressure-mode rather than HU
! v3_21 - Dugas B.             - replace TD by ES in pressure mode
! v3_30 - Tanguay M.           - Modify Check topo when no interpolation 
! v3_30 - McTaggart-Cowan R.   - update implementation of variable orography
! v4_03 - Lee V.               - ISST
!
#include "bmf.cdk"
#include "ptopo.cdk"
#include "path.cdk"
#include <clib_interface_mu.hf>
!
      integer  bmf_gobe
      external bmf_gobe
!
      character*1024 pe_file
      integer hh,mm,ss, length, i, err
      integer, allocatable, dimension(:) :: bmfni,bmfnj,bmfnk, &
               bmfdatyp,bmfvtime1,bmfvtime2, &
               bmfscrap,bmfscrap1,bmfscrap2,bmfscrap3,bmfscrap4,bmfscrap5, &
                        bmfscrap6,bmfscrap7,bmfscrap8,bmfscrap9
      character*4, allocatable, dimension(:) :: bmfnom
!*
!     ---------------------------------------------------------------
!
      call bmf_init
!
      hh=bmf_time2/1000000
      mm=bmf_time2/10000-hh*100
      ss=bmf_time2/100-hh*10000-mm*100
!
      call bmf_splitname ( pe_file,Ptopo_mycol, Ptopo_myrow, &
                           trim(Path_ind_S), 'BM',bmf_time1,hh,mm,ss )
      err = clib_fileexist (trim(pe_file))

      if (err .lt. 0) then
         write(*,1001) trim(pe_file)
      endif
      call handle_error(err,'get_bmfvar','get_bmfvar')	
!
!     Read the BMF file associated to Ptopo_myproc
!
      length=bmf_gobe(pe_file)
!
!     Build a catalog to allow proper dimensionning of some variables
!
      allocate (bmfnom(length),bmfni(length),bmfnj(length),         &
                bmfnk(length), bmfvtime1(length),bmfvtime2(length), &
                bmfdatyp(length),bmfscrap(length),                  &
                bmfscrap1(length),bmfscrap2(length),bmfscrap3(length), &
                bmfscrap4(length),bmfscrap5(length),bmfscrap6(length), &
                bmfscrap7(length),bmfscrap8(length),bmfscrap9(length))

      call bmf_catalog ( bmfnom,bmfni,bmfscrap,bmfscrap1,bmfnj, &
           bmfscrap2,bmfscrap3,bmfnk,bmfscrap4,bmfscrap5,bmfvtime1, &
           bmfvtime2,bmfscrap6,bmfscrap7,bmfdatyp,bmfscrap8,bmfscrap9 )
!
      F_nka_m = -1 ; F_nka_t = -1
      do i=1,length
         if(bmfnom(i).eq.'ZA  ') then
            F_nka_m = bmfni(i)
            cycle
         else if(bmfnom(i).eq.'ZAT ') then
            F_nka_t = bmfni(i)
            cycle
         else if(bmfnom(i).eq.'UU  ') then
            F_niu = bmfni(i)
            F_nju = bmfnj(i)
            cycle
         else if(bmfnom(i).eq.'VV  ') then
            F_niv = bmfni(i)
            F_njv = bmfnj(i)
            cycle
         endif
      enddo
!
      deallocate (bmfni,bmfnj,bmfnk,bmfdatyp,bmfvtime1,              &
                  bmfvtime2,bmfnom,bmfscrap,                         &
                  bmfscrap1,bmfscrap2,bmfscrap3,bmfscrap4,bmfscrap5, &
                  bmfscrap6,bmfscrap7,bmfscrap8,bmfscrap9)
!
 1001 format (/' FILE ',a,' NOT AVAILABLE. Consider re-running Entry'/)
!
!     ---------------------------------------------------------------
!
      return
      end
