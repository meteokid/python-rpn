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
module theo_options
   use bubble_options
   use mtn_options
   implicit none
   public
   save

   !# Choices of theoretical case:
   !# * from Robert, JAS 1993
   !# * 'BUBBLE': uniform convective bubble
   !# * from Schar et al, MWR 2002
   !# * 'MTN_SCHAR': case with N=0.01
   !# * 'MTN_SCHAR2': case with N=0.01871
   !# * from Pinty et al, MWR 1995
   !# * 'MTN_PINTY': linear isothermal with U=32m/s
   !# * 'MTN_PINTY2': linear isothermal with U=8m/s
   !# * 'MTN_PINTYNL': nonlinear regime

   character(len=15) :: Theo_case_S = 'NONE'
   namelist /theo_cfgs/ Theo_case_S

contains

      integer function theocases_nml (F_namelistf_S,F_theo_L)
      use lun
      implicit none

      character(len=*) F_namelistf_S
      logical F_theo_L


      integer, external :: fnom
      character(len=64) :: dumc_S
      integer unf
!
!-------------------------------------------------------------------
!
      theocases_nml = -1

      if ((F_namelistf_S == 'print').or.(F_namelistf_S == 'PRINT')) then
         theocases_nml = 0
         if ( Lun_out >= 0) write (Lun_out,nml=theo_cfgs)
         return
      endif

      if (F_namelistf_S /= '') then

         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', 0) /= 0) goto 9110
         rewind(unf)
         read (unf, nml=theo_cfgs, end= 1000, err=9130)
 1000    call fclos (unf)

      endif

      call low2up (Theo_case_S ,dumc_S)
      Theo_case_S = dumc_S

      if (  Theo_case_S == 'NONE' ) then
         theocases_nml= 0 ; F_theo_L= .false.
      else
         F_theo_L= .true.
         if (   Theo_case_S == 'MTN_SCHAR'   &
           .or. Theo_case_S == 'MTN_SCHAR2'  &
           .or. Theo_case_S == 'MTN_PINTY'   &
           .or. Theo_case_S == 'MTN_PINTY2'  &
           .or. Theo_case_S == 'MTN_PINTYNL' &
           .or. Theo_case_S == 'NOFLOW' ) then
           theocases_nml= mtn_nml (F_namelistf_S)
        else if (  Theo_case_S == 'BUBBLE' ) then
           theocases_nml= bubble_nml (F_namelistf_S)
        else
           if (Lun_out > 0) then
              write (Lun_out, 9200) Theo_case_S
              write (Lun_out, 8000)
           endif
        endif
      endif

      goto 9999

 9110 if (Lun_out > 0) then
         write (Lun_out, 9050) trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9130 call fclos (unf)
      if (Lun_out >= 0) then
         write (Lun_out, 9150) 'theo_cfgs',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif

 8000 format (/,'========= ABORT IN S/R theocases_nml ============='/)
 9050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 9150 format (/,' NAMELIST ',A,' INVALID IN FILE: ',A/)
 9200 format (/,' Unsupported theoretical case: ',a/)
 9999 return
      end function theocases_nml
!
!-------------------------------------------------------------------
!
      subroutine theo_cfg()
      use lun
      implicit none

      integer err
!
!     ---------------------------------------------------------------
!
      if (  Theo_case_S == 'NONE' ) return

      err= -1
      if (  Theo_case_S == 'MTN_SCHAR' &
           .or. Theo_case_S == 'MTN_SCHAR2' &
           .or. Theo_case_S == 'MTN_PINTY' &
           .or. Theo_case_S == 'MTN_PINTY2' &
           .or. Theo_case_S == 'MTN_PINTYNL' &
           .or. Theo_case_S == 'NOFLOW' ) then
         err = mtn_cfg ()
      else if (  Theo_case_S == 'BUBBLE' ) then
         err = bubble_cfg ()
      else
         if (Lun_out > 0) then
            write (Lun_out, 9200) Theo_case_S
         endif
      endif

      call gem_error (err, 'theo_cfg', Theo_case_S)

      if (Lun_out > 0) write (Lun_out, 7050) Theo_case_S

 7050 format (/' THEORETICAL CASE IS: ',a/)
 9200 format (/,' Unsupported theoretical case: ',a/)

      return
      end subroutine theo_cfg
!
!     ---------------------------------------------------------------
!
      subroutine theo_data ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo)

      use glb_ld
      implicit none
#include <arch_specific.hf>

      real F_u(*), F_v(*), F_w (*), F_t(*), F_zd(*), &
           F_s(*), F_topo(*), F_q(*)

!
!---------------------------------------------------------------------
!
      if (      Theo_case_S == 'MTN_SCHAR'   &
           .or. Theo_case_S == 'MTN_SCHAR2'  &
           .or. Theo_case_S == 'MTN_PINTY'   &
           .or. Theo_case_S == 'MTN_PINTY2'  &
           .or. Theo_case_S == 'MTN_PINTYNL' &
           .or. Theo_case_S == 'NOFLOW' ) then

         call mtn_data ( F_u, F_v, F_t, F_s, F_q, F_topo, &
                         l_minx, l_maxx, l_miny, l_maxy, G_nk, Theo_case_S )

      elseif ( Theo_case_S == 'BUBBLE' ) then

         call bubble_data ( F_u, F_v, F_t, F_s, F_q, F_topo, &
                            l_minx, l_maxx, l_miny, l_maxy, G_nk )
      else

         call gem_error(-1,'WRONG THEO CASE in theo_data',Theo_case_S)

      endif
!
!---------------------------------------------------------------------
      return
      end subroutine theo_data





end module theo_options
