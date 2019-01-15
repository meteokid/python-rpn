!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer, 
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms 
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer 
!version 3 or (at your option) any later version that should be found at: 
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html 
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software; 
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec), 
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------

!**s/r phy_simple_transforms - simple checks and transforms of entry fields
subroutine physimple_transforms3d2(F_var_S,values,ni,nj,nk)
    implicit none
#include <arch_specific.hf>
    integer,intent(in) :: ni,nj,nk
    real, intent(inout)          :: values(ni,nj,nk)
    character(len=*), intent(in) :: F_var_S
!Author
!     Lubos Spacek - October 2009
!
!Revision
! v5_10 - Spacek, L.     - First revision
!
!Object
!     Checks the values after intrepolation, preform
!     conversions of units and simple transformations
!
!Arguments
!          - Input/Output -
! values   transformed values
!
!          - Input -
! var      variable name
!
!Implicits
#include <msg.h>
#include <clib_interface_mu.hf>
#include "surface.cdk"
include "thermoconsts.inc"

    real*8,parameter :: ONE_8   = 1.0d0
    real*8,parameter :: CLXXX_8 = 180.0d0
    real*8 :: deg2rad_8
    character(len=32) :: var_S
    integer :: istat
    logical :: done_L
    ! ---------------------------------------------------------------------
    var_S = ' '
    var_S = trim(F_var_S)
    istat = clib_toupper(var_S)

    deg2rad_8 = acos(-ONE_8)/CLXXX_8
    done_L = .true.
    select case(var_S)
    case('DLAT')
       values = deg2rad_8*values
    case('DLON')
       where(values>=0.)
          values = deg2rad_8*values
       elsewhere
          values = deg2rad_8*(values+360.)
       endwhere
    case('GLACEN')
       where(values<0.) values = 0.0
       where(values>1.) values = 1.0 
    case('GLSEAEN')
       where(values<0.) values = 0.0
       where(values>1.) values = 1.0        
    case('MGEN')
       where(values<0.) values = 0.0
       where(values>1.) values = 1.0
    case('MT')
       where(values<0.) values = 0.0
    case('SNODPEN')
       values = 0.01*values
    case('TGLACEN')
       where(values<150.) values = values+tcdk
    case('TMICEN')
       where(values<150.) values = values+tcdk
    case('TSOILEN')
       where(values<150.) values = values+tcdk
    case('TWATEREN')
       where(values<150.) values = values+tcdk
    case('Z0EN')
       values = exp(values)
    case default
       done_L = .false.
    end select

    ! Physical world unit conversions
    if (.not.done_L) then
       done_L = .true.
       select case(var_S(1:5))
       case('PW_TT')
          where(values<150.) values = values+tcdk
       case('PW_UU' , 'PW_VV')
          values = KNAMS*values
       case DEFAULT
          done_L = .false.
       end select
    endif

    if (done_L) call msg(MSG_INFO,'(physimple_transforms3d) '//trim(var_S))
    ! ---------------------------------------------------------------------
    return
 end subroutine physimple_transforms3d2

