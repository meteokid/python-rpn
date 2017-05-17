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

!*module out_vref_mod - output vertical coordinate tags
!
module out_vref_mod
      use grid_options
      use grid_options
      use gem_options
  implicit none
  private

  public :: out_vref

  interface out_vref
     module procedure out_vref_model
     module procedure out_vref_pres
  end interface

#include <arch_specific.hf>

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine out_vref_model(ig1,ig2,etiket)
    use vGrid_Descriptors, only: vgrid_descriptor,vgd_put,vgd_write,vgd_print,VGD_OK,vgd_get
    use vgrid_wb, only: vgrid_wb_get
    ! Write the vertical coordinate descriptor (model levels)

    implicit none

    integer, intent(in), optional :: ig1,ig2            !override 'out.cdk' values of ig1,ig2
    character(len=*), optional :: etiket               !override etiket value

    !author
    !     v.lee - rpn march 2008
    !
    !revision
    ! v4_03 - Lee V.            - initial MPI version (from wrvref MC2)
    ! 
    !implicits

#include "glb_ld.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "dimout.cdk"
#include "cstv.cdk"
#include "level.cdk"
#include "lun.cdk"
#include "out.cdk"

    ! Internal variables
    integer :: err,my_ig1,my_ig2
    integer, dimension(:), pointer :: ip1m
    type(vgrid_descriptor) :: vgd

    ! Set default values
    my_ig1 = Out_ig1
    if (present(ig1)) my_ig1 = ig1
    my_ig2 = Out_ig2
    if (present(ig2)) my_ig2 = ig2

    ! Write coordinate descriptor
    if (writeDescriptor()) then
       nullify(ip1m)
       err = vgrid_wb_get('ref-m',vgd,ip1m)
       deallocate(ip1m); nullify(ip1m)
       err = vgd_put(vgd,'IP_1 - record ip1',my_ig1) 
       err = vgd_put(vgd,'IP_2 - record ip2',my_ig2) 

       if(present(etiket))then
          err = vgd_put(vgd,key='ETIK - record stamp',value=etiket)
          ! there should be en error trapping
       endif

       if (report(Out_ig1,Out_ig2)) err = vgd_print(vgd,Lun_out)
       err = vgd_write(vgd,unit=Out_unf,format='fst')

    endif

    return

  end subroutine out_vref_model

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine out_vref_pres(F_rf,ig1,ig2,etiket)
    use vGrid_Descriptors, only: vgrid_descriptor,vgd_new,vgd_put,vgd_write,vgd_print,VGD_OK
    ! Write the vertical coordinate descriptor (pressure levels)

    implicit none

    real, dimension(:), intent(in) :: F_rf              !List of pressure levels to output
    integer, intent(in), optional :: ig1,ig2            !override 'out.cdk' values of ig1,ig2
    character(len=*), optional :: etiket               !override etiket value
    !author
    !     R. McTaggart-Cowan - Aug 2009
    !
    !revision
    !
    !object
    !     Write vertical coordinate descriptor for pressure-coordinate data
    !
    !implicits
#include "out.cdk"
#include "dimout.cdk"
#include "level.cdk"
#include "lun.cdk"

    ! Local variables
    integer :: k,err,my_ig1,my_ig2
    integer, dimension(size(F_rf)) :: ip1s
    real*8, dimension(size(F_rf)) :: zero
    type(vgrid_descriptor) :: vgd
    logical, save :: reported=.false.

    ! Set default values
    my_ig1 = Out_ig1
    if (present(ig1)) my_ig1 = ig1
    my_ig2 = Out_ig2
    if (present(ig2)) my_ig2 = ig2

    ! Set initializing values
    zero = 0.d0
    do k=1,size(F_rf)
       call convip(ip1s(k),F_rf(k),2,1,'',.false.)
    enddo

    ! Write coordinate descriptor
    if (writeDescriptor()) then

       err = vgd_new(vgd,               &
            kind     = 2,               &
            version  = 1,               &
            nk       = size(F_rf),      &
            ip1      = my_ig1,          &
            ip2      = my_ig2,          &
            a_m_8    = dble(F_rf*100.), &
            b_m_8    = zero,            &
            ip1_m    = ip1s)
       ! there should be en error trapping

       if(present(etiket))then
          err = vgd_put(vgd,key='ETIK - record stamp',value=etiket)
          ! there should be en error trapping
       endif

       if (report(Out_ig1,Out_ig2)) err = vgd_print(vgd,Lun_out)
       err = vgd_write(vgd,unit=Out_unf,format='fst')
       ! there should be en error trapping

    endif
    
  end subroutine out_vref_pres

  logical function writeDescriptor()
    ! Decide whether or not to write descriptor based on local tile
    implicit none
#include "out3.cdk"
#include "ptopo.cdk"

    writeDescriptor = (Out3_iome >= 0) .and. (Ptopo_couleur == 0)

    return
  end function writeDescriptor

  logical function report(ig1,ig2)
    ! Decide whether or not to report output from the descriptor
    implicit none
#include "lun.cdk"
#include "dimout.cdk"
#include "grid.cdk"
#include "level.cdk"
    integer , parameter :: LEVEL_NTYP=2
    integer :: ig1,ig2
    integer :: i
    integer, save :: report_count=0
    integer, dimension(MAXGRID1*LEVEL_NTYP,2), save :: reported
    logical :: found
    report = .false.
    if (.not.Lun_debug_L) return
    i = 1; found = .false.
    do while (.not.found .and. i<=report_count)
       if (ig1 == reported(i,1) .and. ig2 == reported(i,2)) found = .true.
       i = i+1
    enddo
    if (.not.found) then
       report_count = min(report_count+1,size(reported,dim=1))
       reported(report_count,:) = (/ig1,ig2/)
       report = .true.
    endif
    return
  end function report

end module out_vref_mod
