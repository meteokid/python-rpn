!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     This routine is the driver for computing the addresses and weights
!     for interpolating between two grids on a sphere.
!
!-----------------------------------------------------------------------
!
!     CVS:$Id: scrip.f,v 1.6 2001/08/21 21:06:44 pwjones Exp $
!
!     Copyright (c) 1997, 1998 the Regents of the University of
!       California.
!
!     This software and ancillary information (herein called software)
!     called SCRIP is made available under the terms described here.
!     The software has been approved for release with associated
!     LA-CC Number 98-45.
!
!     Unless otherwise indicated, this software has been authored
!     by an employee or employees of the University of California,
!     operator of the Los Alamos National Laboratory under Contract
!     No. W-7405-ENG-36 with the U.S. Department of Energy.  The U.S.
!     Government has rights to use, reproduce, and distribute this
!     software.  The public may copy and use this software without
!     charge, provided that this Notice and any statement of authorship
!     are reproduced on all copies.  Neither the Government nor the
!     University makes any warranty, express or implied, or assumes
!     any liability or responsibility for the use of this software.
!
!     If software is modified to produce derivative works, such modified
!     software should be clearly marked, so as not to confuse it with
!     the version available from Los Alamos National Laboratory.
!
!***********************************************************************

      subroutine scrip_addr_wts()

!-----------------------------------------------------------------------

      use kinds_mod                  ! module defining data types
      use constants                  ! module for common constants
      use grids                      ! module with grid information
      use remap_vars                 ! common remapping variables
      use remap_conservative         ! routines for conservative remap
      use remap_distance_weight      ! routines for dist-weight remap
      use remap_bilinear             ! routines for bilinear interp
      use remap_bicubic              ! routines for bicubic  interp
      use remap_write                ! routines for remap outputrogram main
      implicit none

      integer (kind=int_kind) :: n     ! dummy counter
      integer :: k

 
!-----------------------------------------------------------------------

      call grid_init()
      call init_remap_vars()

      select case(map_type)
      case(map_type_conserv)
        call remap_conserv()
      case(map_type_bilinear)
        call remap_bilin()
      case(map_type_bicubic)
        call remap_bicub()
      case default !map_type_distwgt
        call remap_distwgt()
      end select

!-----------------------------------------------------------------------
!
!     reduce size of remapping arrays
!
!-----------------------------------------------------------------------

      if (num_links_map1 /= max_links_map1) then
        call resize_remap_vars(1, num_links_map1-max_links_map1)
      endif
      if ((num_maps > 1) .and. (num_links_map2 /= max_links_map2)) then
        call resize_remap_vars(2, num_links_map2-max_links_map2)
      endif

      call sort_add(grid2_add_map1, grid1_add_map1, wts_map1)

      if (num_maps > 1) then
        call sort_add(grid1_add_map2, grid2_add_map2, wts_map2)
      endif

!-----------------------------------------------------------------------
      return
      end subroutine scrip_addr_wts

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
