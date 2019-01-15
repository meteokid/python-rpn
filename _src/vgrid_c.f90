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

#include <msg.h>

!/*
module c_vgrid
   use ISO_C_BINDING
   use vGrid_Descriptors
   use vgrid_wb
   implicit none
   private
   !@objective Interface to use vgrid from C, python, ...
   !@author  Stephane Chamberland, 2015-03
   !@description
   ! Public functions
   public ::
   ! C finctions

  interface vgd_new
     module procedure new_read
     module procedure new_from_table
     module procedure new_build_vert
     module procedure new_gen
  end interface

  interface vgd_free
     module procedure garbage_collection
  end interface

  interface vgd_get
     module procedure get_int
     module procedure get_int_1d
     module procedure get_real
     module procedure get_real_1d
     module procedure get_real8
     module procedure get_real8_1d
     module procedure get_real8_3d
     module procedure get_char
     module procedure get_logical
  end interface

  interface vgd_put
     module procedure put_int
     module procedure put_int_1d
     module procedure put_real_1d
     module procedure put_real8
     module procedure put_real8_1d
     module procedure put_real8_3d
     module procedure put_char
  end interface

  interface vgd_getopt
     module procedure getopt_logical
  end interface

  interface vgd_putopt
     module procedure putopt_logical
  end interface
  
  interface vgd_print
     module procedure print_desc
     module procedure print_vcode_description
  end interface

  interface vgd_write
     module procedure write_desc
  end interface

  interface vgd_levels
     module procedure levels_toplevel
     module procedure levels_readref
     module procedure levels_withref
     module procedure levels_withref_8
     module procedure levels_withref_prof
     module procedure levels_withref_prof_8
  end interface

  interface operator (==)
     module procedure test_equality
  end interface

  interface vgd_dpidpis
     module procedure dpidpis_withref
     module procedure dpidpis_withref_8
     module procedure dpidpis_withref_prof
     module procedure dpidpis_withref_prof_8
  end interface

  interface set_vcode
     module procedure set_vcode_d
     module procedure set_vcode_i
  end interface

contains

!!$ function encode_ip_0(IP1,IP2,IP3,RP1,RP2,RP3) result(status) bind (C,name='EncodeIp')
!!$  integer(C_INT) :: status
!!$  integer(C_INT), intent(OUT) :: IP1,IP2,IP3
!!$  real(C_FLOAT),  value, intent(IN)  :: RP1,RP2,RP3
!!$ end function encode_ip_0
   
   
   function f_new_read(iself,unit,ip1,ip2,kind,version) &
        result(status) bind(C,name='c_new_read')
      implicit none
      !@objective Coordinate constructor
      !           read from a file and initialize instance
      integer(C_INT), intent(in) :: iself   ! unique int id for the vgrid
      integer(C_INT), intent(in) :: unit    ! File unit
      integer(C_INT) :: ip1,ip2             ! ip1,2 of the desired descriptors
      integer(C_INT), intent(in) :: kind    ! Level kind requested by user.
      integer(C_INT), intent(in) :: version ! Level version requested by user.
      integer :: status
      type(vgrid_descriptor) :: self
      character(len=3) :: ftype = 'fst'  !File type ('fst' or 'bin')
      status = new_read(self,unit,ftype,ip1,ip2,kind,version)
      if (status < 0) return
      status = priv_store_vgrid(iself,self)
      return
   end function f_new_read


!!$   function f_new_from_table(self,table) &
!!$        result(status) bind(C,name='c_new_from_table')
!!$      implicit none
!!$      !@objective  Coordinate constructor
!!$      !            build vertical descriptor from table input
!!$      !            Set internal vcode (if all above was successful)
!!$      type(vgrid_descriptor), intent(inout) :: self      !Vertical descriptor instance    
!!$      real(kind=8), dimension(:,:,:), pointer :: table   !Raw table of vgrid records
!!$      integer :: status
!!$      type(vgrid_descriptor) :: self
!!$   end function f_new_from_table


!!$   integer function new_build_vert(self,kind,version,nk,ip1,ip2, &
!!$        ptop_8,pref_8,rcoef1,rcoef2,a_m_8,b_m_8,a_t_8,b_t_8, &
!!$        ip1_m,ip1_t) result(status)
!!$      ! Coordinate constructor - build vertical descriptor from arguments
!!$      type(vgrid_descriptor) :: self                    !Vertical descriptor instance    
!!$      integer, intent(in) :: kind,version               !Kind,version to create
!!$      integer, intent(in) :: nk                         !Number of levels
!!$      integer, optional, intent(in) :: ip1,ip2          !IP1,2 values for FST file record [0,0]
!!$      real, optional, intent(in) :: rcoef1,rcoef2       !R-coefficient values for rectification
!!$      real*8, optional, intent(in) :: ptop_8            !Top-level pressure (Pa)
!!$      real*8, optional, intent(in) :: pref_8            !Reference-level pressure (Pa)
!!$      real*8, optional, dimension(:) :: a_m_8,a_t_8     !A-coefficients for momentum(m),thermo(t) levels
!!$      real*8, optional, dimension(:) :: b_m_8,b_t_8     !B-coefficients for momentum(m),thermo(t) levels
!!$      integer, optional, dimension(:) :: ip1_m,ip1_t    !Level ID (IP1) for momentum(m),thermo(t) levels


!!$   integer function new_gen(self,kind,version,hyb,rcoef1,rcoef2,ptop_8,pref_8,ptop_out_8,ip1,ip2,stdout_unit,dhm,dht) result(status)
!!$      use vdescript_1001,      only: vgrid_genab_1001
!!$      use vdescript_1002_5001, only: vgrid_genab_1002_5001
!!$      use vdescript_2001,      only: vgrid_genab_2001
!!$      use vdescript_5002,      only: vgrid_genab_5002
!!$      ! Coordinate constructor - build vertical descriptor from hybrid coordinate entries
!!$      type(vgrid_descriptor),intent(inout) :: self      !Vertical descriptor instance    
!!$      integer, intent(in) :: kind,version               !Kind,version to create
!!$      real, dimension(:),intent(in) :: hyb              !List of hybrid levels
!!$      real, optional, intent(in) :: rcoef1,rcoef2       !R-coefficient values for rectification
!!$      real*8, optional, intent(in) :: ptop_8            !Top-level pressure (Pa) inout
!!$      real*8, optional, intent(out):: ptop_out_8        !Top-level pressure (Pa) output if ptop_8 < 0
!!$      real*8, optional, intent(in) :: pref_8            !Reference-level pressure (Pa)
!!$      integer, optional, intent(in) :: ip1,ip2          !IP1,2 values for FST file record [0,0]
!!$      integer, optional, intent(in) :: stdout_unit      !Unit number for verbose output [STDERR]
!!$      real, optional, intent(in) :: dhm,dht             !Diag levels Height for Momentum/Thermo vaiables


!!$   integer function garbage_collection(self) result(status)
!!$     ! Free all memory associated with with structure and uninitialize it
!!$     type(vgrid_descriptor), intent(inout) :: self          !Vertical descriptor instance


!!$  integer function get_logical(self,key,value,quiet) result(status)
!!$    use utils, only: up
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self          !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key                 !Descriptor key to retrieve
!!$    logical, intent(out) :: value                       !Retrieved value
!!$    logical, optional, intent(in) :: quiet              !Do not print massages    


!!$  integer function get_int(self,key,value,quiet) result(status)
!!$    use utils, only: up,get_error
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self          !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key                 !Descriptor key to retrieve
!!$    integer, intent(out) :: value                       !Retrieved value
!!$    logical, optional, intent(in) :: quiet              !Do not print massages


!!$  integer function get_int_1d(self,key,value,quiet) result(status)
!!$    use utils, only: get_allocate,up,get_error
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self          !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key                 !Descriptor key to retrieve
!!$    integer, dimension(:), pointer :: value             !Retrieved value
!!$    logical, optional, intent(in) :: quiet              !Do not print massages


!!$  integer function get_real(self,key,value,quiet) result(status)
!!$    use utils, only: up,get_error
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self   !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key          !Descriptor key to retrieve
!!$    real, intent(out) :: value                   !Retrieved value
!!$    logical, optional, intent(in) :: quiet       !Do not print massages


!!$  integer function get_real_1d(self,key,value,quiet) result(status)
!!$    use utils, only: get_allocate,up,get_error
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self  !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to retrieve
!!$    real, dimension(:), pointer :: value        !Retrieved value
!!$    logical, optional, intent(in) :: quiet      !Do not print massages


!!$  integer function get_real8(self,key,value,quiet) result(status)
!!$    use utils, only: up,get_error
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self  !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to retrieve
!!$    real(kind=8), intent(out) :: value          !Retrieved value
!!$    logical, optional, intent(in) :: quiet      !Do not print massages


!!$  integer function get_real8_1d(self,key,value,quiet) result(status)
!!$    use utils, only: get_allocate,up,get_error
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self  !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to retrieve
!!$    real(kind=8), dimension(:), pointer :: value !Retrieved value
!!$    logical, optional, intent(in) :: quiet      !Do not print massages


!!$  integer function get_real8_3d(self,key,value,quiet) result(status)
!!$    use utils, only: get_allocate,up
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self  !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to retrieve
!!$    real(kind=8), dimension(:,:,:), pointer :: value !Retrieved value
!!$    logical, optional, intent(in) :: quiet      !Do not print massages


!!$  integer function get_char(self,key,value,quiet) result(status)
!!$    use utils, only: up,get_error,printingCharacters
!!$    ! Retrieve the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(in) :: self  !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to retrieve
!!$    character(len=*), intent(out) :: value      !Retrieved value
!!$    logical, optional, intent(in) :: quiet      !Do not print massages


!!$  integer function getopt_logical(key,value,quiet) result(status)
!!$     character(len=*), intent(in) :: key           !Descriptor key to retrieve
!!$     logical, intent(out) :: value                    !Retrieved value
!!$     logical, intent(in), optional :: quiet        !Do not generate messages


!!$  integer function put_int(self,key,value) result(status)
!!$    use utils, only: up,comp_diag_a
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to set
!!$    integer, intent(in) :: value                !Value to set


!!$  integer function put_int_1d(self,key,value) result(status)
!!$    use utils, only: size_ok,up,put_error
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self       !Vertical descriptor instance
!!$    character(len=*), intent(in) :: key                 !Descriptor key to set
!!$    integer, dimension(:), pointer :: value             !Value to set


!!$  integer function put_real_1d(self,key,value) result(status)
!!$    use utils, only: up
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self!Descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to set
!!$    real, dimension(:), pointer :: value !Value to set


!!$  integer function put_real8(self,key,value) result(status)
!!$    use utils, only: up,put_error
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self!Descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to set
!!$    real(kind=8), intent(in) :: value           !Value to set


!!$  integer function put_real8_1d(self,key,value) result(status)
!!$    use utils, only: size_ok,up,put_error
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self!Descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to set
!!$    real(kind=8), dimension(:), pointer :: value !Value to set


!!$  integer function put_real8_3d(self,key,value) result(status)
!!$    use utils, only: size_ok, up
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self!Descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to set
!!$    real(kind=8), dimension(:,:,:), pointer :: value !Value to set


!!$  integer function put_char(self,key,value) result(status)
!!$    use utils, only: up,put_error
!!$    ! Set the value of the requested instance variable
!!$    type(vgrid_descriptor), intent(inout) :: self!Descriptor instance
!!$    character(len=*), intent(in) :: key         !Descriptor key to set
!!$    character(len=*), intent(in) :: value       !Value to set


!!$  integer function putopt_logical(key,value) result(status)
!!$     character(len=*), intent(in) :: key           !Descriptor key to retrieve
!!$     logical, intent(in) :: value                    !Retrieved value


!!$  integer function write_desc(self,unit,format) result(status)     
!!$    use utils, only: up
!!$    ! Write descriptors to the requested file
!!$    type(vgrid_descriptor), intent(in) :: self       !Vertical descriptor instance
!!$    integer, intent(in) :: unit                      !File unit to write to
!!$    character(len=*), optional, intent(in) :: format !File format ('fst' or 'bin' ) default is 'fst'


!!$  integer function levels_toplevel(unit,fstkeys,levels,in_log) result(status)
!!$    ! Top-level interface for computing physical levelling information
!!$    integer, intent(in) :: unit                         !File unit associated with the key
!!$    integer, dimension(:), intent(in) :: fstkeys        !Key of prototype field
!!$    real, dimension(:,:,:), pointer :: levels           !Physical level values
!!$    logical, optional, intent(in) :: in_log             !Compute levels in ln() [.false.]


!!$  integer function levels_readref(self,unit,fstkeys,levels,in_log) result(status)
!!$    ! Reading referent, compute physical levelling information from the vertical description
!!$    type(vgrid_descriptor), intent(in) :: self          !Vertical descriptor instance
!!$    integer, intent(in) :: unit                         !File unit associated with the key
!!$    integer, dimension(:), intent(in) :: fstkeys        !Key of prototype field
!!$    real, dimension(:,:,:), pointer :: levels           !Physical level values
!!$    logical, optional, intent(in) :: in_log             !Compute levels in ln() [.false.]


!!$  integer function levels_withref_prof(self,ip1_list,levels,sfc_field,in_log) result(status)
!!$     use utils, only: get_allocate
!!$     type(vgrid_descriptor), intent(in) :: self                  !Vertical descriptor instance
!!$     integer, dimension(:), intent(in) :: ip1_list               !Key of prototype field
!!$     real, dimension(:), pointer :: levels                       !Physical level values
!!$     real, optional, intent(in) :: sfc_field                     !Surface field reference for coordinate [none]
!!$     logical, optional, intent(in) :: in_log                     !Compute levels in ln() [.false.]        


!!$  integer function levels_withref_prof_8(self,ip1_list,levels,sfc_field,in_log) result(status)
!!$     type(vgrid_descriptor), intent(in) :: self                  !Vertical descriptor instance
!!$     integer, dimension(:), intent(in) :: ip1_list               !Key of prototype field
!!$     real*8, dimension(:), pointer :: levels                       !Physical level values
!!$     real*8, optional, intent(in) :: sfc_field                     !Surface field reference for coordinate [none]
!!$     logical, optional, intent(in) :: in_log                     !Compute levels in ln() [.false.]


!!$  integer function dpidpis_withref_prof(self,ip1_list,dpidpis,sfc_field) result(status)
!!$     use utils, only: get_allocate,up,get_error
!!$     type(vgrid_descriptor), intent(in) :: self                  !Vertical descriptor instance
!!$     integer, dimension(:), intent(in) :: ip1_list               !Key of prototype field
!!$     real, dimension(:), pointer :: dpidpis                      !Derivative values
!!$     real, optional, intent(in) :: sfc_field                     !Surface field reference for coordinate [none]


!!$  integer function dpidpis_withref_prof_8(self,ip1_list,dpidpis,sfc_field) result(status)
!!$     use utils, only: get_allocate,up,get_error
!!$     type(vgrid_descriptor), intent(in) :: self                  !Vertical descriptor instance
!!$     integer, dimension(:), intent(in) :: ip1_list               !Key of prototype field
!!$     real*8, dimension(:), pointer :: dpidpis                      !Derivative values
!!$     real*8, optional, intent(in) :: sfc_field                     !Surface field reference for coordinate [none]


!!$  integer function diag_withref_prof_8(self,ip1_list,levels,sfc_field,in_log,dpidpis) result(status)
!!$     use utils, only: get_allocate
!!$     type(vgrid_descriptor), intent(in) :: self                  !Vertical descriptor instance
!!$     integer, dimension(:), intent(in) :: ip1_list               !Key of prototype field
!!$     real*8, dimension(:), pointer :: levels                       !Physical level values
!!$     real*8, optional, intent(in) :: sfc_field                     !Surface field reference for coordinate [none]
!!$     logical, optional, intent(in) :: in_log                     !Compute levels in ln() [.false.]          
!!$     logical, optional, intent(in) :: dpidpis                    !Compute partial derivative of hydrostatic pressure (pi) with
!!$                                                                 !   respect to surface hydrostatic pressure(pis) [.false.]




   !==== Private functions =================================================


   subroutine priv_idx2name(name_S,idx)
      character(len=*), intent(out) :: name_S
      integer, intent(in) :: idx
      character(len=5),parameter :: PREFIX = 'cvgd/'
      write(name_S,'*') idx
      name_S = PREFIX//adjustl(name_S)
   end subroutine priv_idx2name


   function priv_store_vgrid(idx,self) result(status) 
      integer, intent(in) :: idx
      type(vgrid_descriptor) :: self
      integer :: status
      character(len=32) :: name_S
      integer,target :: dummy(1)
      integer,pointer :: pdummy(:) 
      call priv_idx2name(name_S,iself)
      dummy(1) = 0 ; pdummy => dummy
      status = vgrid_wb_put(name_S,self,pdummy)
   end function priv_store_vgrid


   function priv_get_vgrid(idx,self) result(status) 
      integer, intent(in) :: idx
      type(vgrid_descriptor) :: self
      integer :: status
      character(len=32) :: name_S
      call priv_idx2name(name_S,iself)
      status = vgrid_wb_get(name_S,self)
   end function priv_get_vgrid

end module c_vgrid
