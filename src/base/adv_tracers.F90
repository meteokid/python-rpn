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
!
   subroutine adv_tracers (F_water_tracers_only_L)

   implicit none
#include <arch_specific.hf>

   
   !@objective Perform advection of all tracers
   !@arguments
   logical, intent(IN) :: F_water_tracers_only_L

   !@author  Michel Desgagne
   !@revisions
   ! v4_70 - Desgagne          - Initial Version
   ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation

#include "grd.cdk"
#include "glb_ld.cdk"
#include "adv_pos.cdk"
#include "lam.cdk"
#include "tr3d.cdk" 
#include "gmm.hf"

   logical qw_L      
   integer  n,i0,j0,in,jn ,jext
   type(gmm_metadata) :: mymeta
   real, pointer, dimension (:,:,:) :: fld_in, fld_out
   integer  err

     
      jext=1
      if (Grd_yinyang_L) jext=2
      if (l_west)  i0 =        pil_w - jext
      if (l_east)  in = l_ni - pil_e + jext
      if (l_south) j0 =        pil_s - jext
      if (l_north) jn = l_nj - pil_n + jext


    do n=1,Tr3d_ntr
      qw_L= Tr3d_wload(n) .or. Tr3d_name_S(n)(1:2).eq.'HU'
      if (F_water_tracers_only_L) then
         if (.not. qw_L) cycle
      else
         if (qw_L) cycle
      endif
          
   err =     gmm_get('TR/'//trim(Tr3d_name_S(n))//':P' ,fld_in ,mymeta)
   err = min(gmm_get('TR/'//trim(Tr3d_name_S(n))//':M' ,fld_out,mymeta),err)
           
        call adv_cubic ('TR/'//trim(Tr3d_name_S(n))//':M', fld_out , fld_in,  &
                        pxt, pyt, pzt,l_ni, l_nj, l_nk,l_minx, l_maxx, l_miny, l_maxy, &
                        i0, in, j0, jn, Lam_gbpil_t+1, 't', Tr3d_mono(n), Tr3d_mass(n) ) 
     end do
          
    
  end subroutine adv_tracers



   
  
