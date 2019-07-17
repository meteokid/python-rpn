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
program use_new_gen

  use vGrid_Descriptors, only: vgrid_descriptor,vgd_new,vgd_print,VGD_OK

  implicit none

  type(vgrid_descriptor) :: vgd
  integer :: stat
  real, dimension(57) :: hyb= &
     (/0.0134575, 0.0203980, 0.0333528, 0.0472815, 0.0605295, 0.0720790, &
       0.0815451, 0.0889716, 0.0946203, 0.0990605, 0.1033873, 0.1081924, &
       0.1135445, 0.1195212, 0.1262188, 0.1337473, 0.1422414, 0.1518590, &
       0.1627942, 0.1752782, 0.1895965, 0.2058610, 0.2229843, 0.2409671, &
       0.2598105, 0.2795097, 0.3000605, 0.3214531, 0.3436766, 0.3667171, &
       0.3905587, 0.4151826, 0.4405679, 0.4666930, 0.4935319, 0.5210579, &
       0.5492443, 0.5780612, 0.6074771, 0.6374610, 0.6679783, 0.6989974, &
       0.7299818, 0.7591944, 0.7866292, 0.8123021, 0.8362498, 0.8585219, &
       0.8791828, 0.8983018, 0.9159565, 0.9322280, 0.9471967, 0.9609448, &
       0.9735557, 0.9851275, 0.9950425/)
  real :: rcoef1=0.,rcoef2=1.
  
  real*8 :: ptop=805d0,pref=100000d0

  stat = vgd_new(vgd,kind=5,version=2,hyb=hyb,rcoef1=rcoef1,rcoef2=rcoef2,ptop_8=ptop,pref_8=pref)
  
  if ( stat /= VGD_OK )then
     print*,'ERROR'
     call exit(1)
  endif

  stat = vgd_print(vgd)

end program use_new_gen
