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

subroutine adx_tracers_interp (F_water_tracers_only_L)
   implicit none
#include <arch_specific.hf>

   !@objective Perform advection of all tracers
   !@arguments
   logical, intent(IN) :: F_water_tracers_only_L

   !@author  Michel Desgagne
   !@revisions
   ! v4_70 - Desgagne          - Initial Version
   ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation

#include "adx_dims.cdk"
#include "adx_poles.cdk"
#include "adx_pos.cdk"
#include "tr3d.cdk"
#include "adx_tracers.cdk"

   logical qw_L
   integer  n,i0,j0,in,jn
   integer, dimension(adx_mlni, adx_mlnj, adx_lnk) :: exch_c1
   real   , dimension(adx_mlni, adx_mlnj, adx_lnk) :: exch_n1, exch_xgg1, exch_xdd1
   real   , dimension(:), allocatable :: capx2,capy2,capz2
   real    dummy

   !---------------------------------------------------------------------

   if (F_water_tracers_only_L.or..not.adx_core_L) then
   call adx_get_ij0n (i0,in,j0,jn)
   else
   call adx_get_ij0n_core (i0,in,j0,jn)
   endif

   if (.not.adx_lam_L) then
      call adx_exch_1c ( exch_n1, exch_xgg1, exch_xdd1, exch_c1, &
                         pxt, pyt, pzt, adx_mlni, adx_mlnj, adx_gbpil_t+1,adx_lnk)

      allocate ( capx2(max(1,adx_fro_a)), &
                 capy2(max(1,adx_fro_a)), &
                 capz2(max(1,adx_fro_a)) )

      call adx_exch_2 ( capx2, capy2, capz2, dummy, dummy,           &
                        exch_n1, exch_xgg1, exch_xdd1, dummy, dummy, &
                        adx_fro_n, adx_fro_s, adx_fro_a, &
                        adx_for_n, adx_for_s, adx_for_a, 3)
   endif

   do n=1,Tr3d_ntr
      qw_L= Tr3d_wload(n) .or. Tr3d_name_S(n)(1:2).eq.'HU'
      if (F_water_tracers_only_L) then
         if (.not. qw_L) cycle
      else
         if (qw_L) cycle
      endif
      call adx_interp_gmm7 ( 'TR/'//trim(Tr3d_name_S(n))//':M'         , &
                             'TR/'//trim(Tr3d_name_S(n))//':P', .false., &
                              pxt, pyt, pzt, capx2, capy2, capz2       , &
                              exch_c1, adx_lnk, i0, in, j0, jn, adx_gbpil_t+1, 't', Tr3d_mono(n), Tr3d_mass(n) )
   end do

   if (.not.adx_lam_L) deallocate(capx2, capy2, capz2)


   !---------------------------------------------------------------------
   return
end subroutine adx_tracers_interp

