!/* libdescrip - Vertical grid descriptor library for FORTRAN programming
! * Copyright (C) 2016  Direction du developpement des previsions nationales
! *                     Centre meteorologique canadien
! *
! * This library is free software; you can redistribute it and/or
! * modify it under the terms of the GNU Lesser General Public
! * License as published by the Free Software Foundation,
! * version 2.1 of the License.
! *
! * This library is distributed in the hope that it will be useful,
! * but WITHOUT ANY WARRANTY; without even the implied warranty of
! * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
! * Lesser General Public License for more details.
! *
! * You should have received a copy of the GNU Lesser General Public
! * License along with this library; if not, write to the
! * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
! * Boston, MA 02111-1307, USA.
program tests
  use vGrid_Descriptors, only: vgrid_descriptor,vgd_new,vgd_get,vgd_dpidpis,VGD_OK
  implicit none
  integer :: stat,lu=0,fnom,fstouv,fstfrm
  integer, dimension(:), pointer :: ip1_list
  real, dimension(:), pointer :: dpidpis_profil
  real, dimension(:,:,:), pointer :: dpidpis_cube
  real, dimension(:,:), pointer :: sfc_field_2d
  real :: sfc_field
  type(vgrid_descriptor) :: vgd

  nullify(ip1_list,dpidpis_profil,dpidpis_cube,sfc_field_2d)

  stat=fnom(lu,"../tests/data/dm_5002_from_model_run","RND",0)
  if(stat.lt.0)then
     print*,'ERROR with fnom'
     call abort
  endif
  stat=fstouv(lu,'RND')
  if(stat.lt.0)then
     print*,'No record in RPN file'
     call abort
  endif
  
  stat = vgd_new(vgd,unit=lu,format="fst")
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_new'
     call abort
  endif
  
  stat = vgd_get(vgd,key='VIPT - level ip1 list (t)'        ,value=ip1_list)
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_get on VIPT'
     call abort
  endif
  
  sfc_field=100000.
  stat = vgd_dpidpis(vgd,sfc_field=sfc_field,ip1_list=ip1_list,dpidpis=dpidpis_profil)
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_dpidpis profile'
     call abort
  endif
  print*,'dpidpis profile'
  print*,dpidpis_profil

  ! Now get derivative for a 2d sfc_field
  allocate(sfc_field_2d(2,2))
  sfc_field_2d(1,1)=100000.
  sfc_field_2d(1,2)= 99900.
  sfc_field_2d(2,1)= 99800.
  sfc_field_2d(2,2)= 99700.

  stat = vgd_dpidpis(vgd,sfc_field=sfc_field_2d,ip1_list=ip1_list,dpidpis=dpidpis_cube)
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_dpidpis cube'
     call abort
  endif
  print*,'dpidpis cube'
  print*,dpidpis_cube(1,1,:)
  print*,dpidpis_cube(1,2,:)
  print*,dpidpis_cube(2,1,:)
  print*,dpidpis_cube(2,2,:)

  stat=fstfrm(lu)
end program tests
