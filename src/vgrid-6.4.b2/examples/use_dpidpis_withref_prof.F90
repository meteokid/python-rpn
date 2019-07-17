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
  real :: sfc_field
  type(vgrid_descriptor) :: vgd
  
  nullify(ip1_list, dpidpis_profil)

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

  ! Get vertical structure object
  stat = vgd_new(vgd,unit=lu,format="fst")
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_new'
     call abort
  endif

  ! Get thermo ip1
  stat = vgd_get(vgd,key='VIPT - level ip1 list (t)'        ,value=ip1_list)
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_get on VIPT'
     call abort
  endif

  ! Get dpidpis
  ! sfc_field is the surface pressure in Pa
  sfc_field=100000.
  stat = vgd_dpidpis(vgd,sfc_field=sfc_field,ip1_list=ip1_list,dpidpis=dpidpis_profil)
  if(stat.ne.VGD_OK)then
     print*,'ERROR: problem with vgd_dpidpis'
     call abort
  endif

  print*,dpidpis_profil

  stat=fstfrm(lu)

end program tests
