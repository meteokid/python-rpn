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

!**s/r stat_mass_tracers - Calculate and print the mass of each tracer 

   subroutine stat_mass_tracers (F_time,F_comment_S) 
      use gem_options
   implicit none
 
   !arguments
   integer,          intent(in) :: F_time       !I, Time 0 or Time 1 
   character(len=*), intent(in) :: F_comment_S  !I, Comment

   !object
   !================================================
   !     Calculate and print the mass of each tracer 
   !================================================

#include "glb_ld.cdk"
#include "gmm.hf"
#include "tr3d.cdk"
#include "lun.cdk"
#include "ptopo.cdk"
#include "tracers.cdk"

   !---------------------------------------------------------------------

   integer :: err,n,k0,scaling_KEEP,count
   real, pointer, dimension (:,:,:) :: fld_tr
   real*8 tracer_8
   real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk):: air_mass,bidon,fld_ONE
   character(len=21) type_S
   character(len= 7) time_S
   character(len=GMM_MAXNAMELENGTH) in_S
   logical,save :: done_L=.false.
   real*8, save :: KEEP_tracer_8(MAXTR3D)

   !---------------------------------------------------------------------

   call canonical_terminator_0 (count)

   k0 = Lam_gbpil_T+1 

   call get_density (bidon,air_mass,F_time,l_minx,l_maxx,l_miny,l_maxy,l_nk,k0)

   if( F_time==1) time_S = "TIME T1"
   if (F_time==0) time_S = "TIME T0"

   type_S = "Mass of Mixing  (WET)"
   if (Schm_dry_mixing_ratio_L) type_S = "Mass of Mixing  (DRY)"

   do n=1,Tr3d_ntr

      if (Tr3d_mass(n)==0) cycle

      if (F_time==1) in_S = 'TR/'//trim(Tr3d_name_S(n))//':P'
      if (F_time==0) in_S = 'TR/'//trim(Tr3d_name_S(n))//':M'

      err = gmm_get(in_S, fld_tr)
      
      call canonical_terminator_1 (fld_tr,in_S,count,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,l_nk)

      call mass_tr (tracer_8,Tr3d_name_S(n)(1:4),fld_tr,air_mass,l_minx,l_maxx,l_miny,l_maxy,l_nk-k0+1,k0)

      if (.not.done_L) KEEP_tracer_8(n) = tracer_8

      if (Lun_out>0.and.Ptopo_couleur==0) write(Lun_out,1002) 'TRACERS: ',type_S,time_S,' C= ',tracer_8/KEEP_tracer_8(n),Tr3d_name_S(n)(1:4),F_comment_S

      call canonical_terminator_2 (air_mass,tracer_8,count,l_minx,l_maxx,l_miny,l_maxy,l_nk,k0,Lun_out,Ptopo_couleur,type_S,time_S,F_comment_S)

   enddo

   fld_ONE = 1.

   type_S = "Mass of Density (WET)"
   if (Schm_dry_mixing_ratio_L) type_S = "Mass of Density (DRY)"

   scaling_KEEP = Tr_scaling
   Tr_scaling   = 0

   call mass_tr (tracer_8,'RHO ',fld_ONE,air_mass,l_minx,l_maxx,l_miny,l_maxy,l_nk-k0+1,k0)

   if (Lun_out>0.and.Ptopo_couleur==0) write(Lun_out,1002) 'TRACERS: ',type_S,time_S,' C= ',tracer_8,'RHO ',F_comment_S

   Tr_scaling = scaling_KEEP

   done_L = .true.

   !---------------------------------------------------------------------

   return

1002 format(1X,A9,A21,1X,A7,A4,E19.12,1X,A4,1X,A16)

end subroutine stat_mass_tracers 
