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

module tracers
   implicit none
   public
   save
!
!___________________________________________________________________________________
!                              |                                                   |
! NAME                         | DESCRIPTION                                       |
!------------------------------|---------------------------------------------------|
! extension_L                  | IF .T. do extended advection operations           |
! core_L                       | IF .T. TR3D_MASS/=0 and TR3D_MONO/=0 LAM (LEGACY) |
! flux_L                       | IF .T. PSADJ LAM or Bermejo-Conde LAM             |
! slice_L                      | IF .T. if some tracers use SLICE                  |
! do_only_once_each_tim.._L    | IF .T. done only once each timestep (SLICE)       |
! verbose                      | Activate conservation diagnostics if /=0          |
! BC_min_max_L                 |switch:.T.:MONO(CLIP) after Bermejo-Conde Def.=T   |
! ILMC_min_max_L               |switch:.T.:MONO(CLIP) after ILMC          Def.=T   |
! ILMC_sweep_max               |Number of neighborhood zones in ILMC      Def.=2   |
! scaling                      |Scaling for mass of tracer: 1=CO2,2=O3,0=None      |
! SLICE_build                  |Type of reconstruction in SLICE Def.=2 (CW)        |
! SLICE_mono                   |Type of monotonicity in SLICE Def.=0 (NONE)        |
! pil_sub_s                    | South boundary in GY for an embedded LAM          |
! pil_sub_n                    | North boundary in GY for an embedded LAM          |
! pil_sub_w                    |  West boundary in GY for an embedded LAM          |
! pil_sub_e                    |  East boundary in GY for an embedded LAM          |
! ----------------------------------------------------------------------------------

      logical :: Tr_extension_L,Tr_core_L,Tr_flux_L, &
                 Tr_do_only_once_each_timestep_L,Tr_slice_L, &
                 Tr_BC_min_max_L,Tr_ILMC_min_max_L

      integer :: Tr_pil_sub_s,Tr_pil_sub_n,Tr_pil_sub_w,Tr_pil_sub_e,Tr_verbose,Tr_ILMC_sweep_max, &
                 Tr_SLICE_rebuild,Tr_SLICE_mono,Tr_scaling

end module tracers
