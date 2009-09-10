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

      subroutine scrip_addr_wts(
     $     F_grid1_add_map1, F_grid2_add_map1, F_wts_map1, 
     $     F_num_wts,F_num_links,
     $     F_num_srch_bins, F_map_method, 
     $     F_normalize_opt, F_restrict_type,
     $     F_grid1_size,F_grid1_dims,F_grid1_corners,
     $     F_grid1_center_lat,F_grid1_center_lon,
     $     F_grid1_corner_lat,F_grid1_corner_lon,
     $     F_grid2_size,F_grid2_dims,F_grid2_corners,
     $     F_grid2_center_lat,F_grid2_center_lon,
     $     F_grid2_corner_lat,F_grid2_corner_lon)

!-----------------------------------------------------------------------

      use kinds_mod                  ! module defining data types
      use constants                  ! module for common constants
      use grids                      ! module with grid information
      use remap_vars                 ! common remapping variables
      use remap_conservative         ! routines for conservative remap
      use remap_distance_weight      ! routines for dist-weight remap
      use remap_bilinear             ! routines for bilinear interp
      use remap_bicubic              ! routines for bicubic  interp
      use remap_write                ! routines for remap output

      implicit none

      !- Output Args
      integer (kind=int_kind) :: F_num_wts, F_num_links
      integer (kind=int_kind), dimension(:), pointer ::
     &      F_grid1_add_map1, ! grid1 address for each link in mapping 1
     &      F_grid2_add_map1  ! grid2 address for each link in mapping 1

      real (kind=dbl_kind), dimension(:,:), pointer ::
     &      F_wts_map1        ! map weights for each link (num_wts,max_links)


      !- Input Args
      integer (kind=int_kind) :: F_num_srch_bins

      character (char_len) ::
     &           F_map_method,   ! choice for mapping method
     &           F_normalize_opt,! option for normalizing weights
     &           F_restrict_type ! type of bins to use

      integer (kind=int_kind), dimension(2) ::
     &             F_grid1_dims, F_grid2_dims  ! size of each grid dimension
      integer (kind=int_kind) ::
     &             F_grid1_size,F_grid2_size,
     &             F_grid1_corners, F_grid2_corners ! number of corners
                                                    ! for each grid cell
      integer (kind=int_kind) :: F_luse_grid_centers

      real (kind=dbl_kind), dimension(F_grid1_size), target ::
     &             F_grid1_center_lat,  ! lat/lon coordinates for
     &             F_grid1_center_lon   ! each grid center in radians

      real (kind=dbl_kind), dimension(F_grid2_size), target ::
     &             F_grid2_center_lat,
     &             F_grid2_center_lon

      real (kind=dbl_kind), 
     &     dimension(F_grid1_corners,F_grid1_size), target  ::
     &             F_grid1_corner_lat,  ! lat/lon coordinates for
     &             F_grid1_corner_lon   ! each grid corner in radians

      real (kind=dbl_kind), 
     &     dimension(F_grid2_corners,F_grid2_size), target  ::
     &             F_grid2_corner_lat,
     &             F_grid2_corner_lon

!-----------------------------------------------------------------------
!
!     input namelist variables
!
!-----------------------------------------------------------------------

      character (char_len) ::
     &           map_method,   ! choice for mapping method
     &           normalize_opt,! option for normalizing weights
     &           output_opt    ! option for output conventions

      integer (kind=int_kind) ::
     &           nmap          ! number of mappings to compute (1 or 2)

!-----------------------------------------------------------------------
!
!     local variables
!
!-----------------------------------------------------------------------

      integer (kind=int_kind) :: n,     ! dummy counter
     &                           iunit  ! unit number for namelist file

!-----------------------------------------------------------------------
!
!     read input namelist
!
!-----------------------------------------------------------------------

      luse_grid1_area = .false.
      luse_grid2_area = .false.
      num_maps      = 1 !2
      map_type      = 1
      map_method    = 'distwgt'
      normalize_opt = 'fracarea'
      output_opt    = 'scrip'
      restrict_type = 'latitude'
      num_srch_bins = 900

c$$$      if (F_num_maps>0)          num_maps      = F_num_maps
      if (F_num_srch_bins>0)     num_srch_bins = F_num_srch_bins
      if (F_map_method   .ne.'') map_method    = F_map_method
      if (F_normalize_opt.ne.'') normalize_opt = F_normalize_opt
c$$$      if (F_output_opt   .ne.'') output_opt    = F_output_opt
      if (F_restrict_type.ne.'') restrict_type = F_restrict_type

      select case(map_method)
      case ('conservative')
        map_type = map_type_conserv
        luse_grid_centers = .false.
      case ('bilinear')
        map_type = map_type_bilinear
        luse_grid_centers = .true.
      case ('bicubic')
        map_type = map_type_bicubic
        luse_grid_centers = .true.
      case default !'distwgt'
        map_type = map_type_distwgt
        luse_grid_centers = .true.
      end select

      select case(normalize_opt(1:4))
      case ('none')
        norm_opt = norm_opt_none
      case ('dest')
        norm_opt = norm_opt_dstarea
      case default !'frac'
        norm_opt = norm_opt_frcarea
      end select

!-----------------------------------------------------------------------
!
!     initialize grid information for both grids
!
!-----------------------------------------------------------------------

c$$$      call grid_init(grid1_file, grid2_file)
      call scrip_grid_init(
     $     F_grid1_dims,F_grid1_corners,
     $     F_grid1_center_lat,F_grid1_center_lon,
     $     F_grid1_corner_lat,F_grid1_corner_lon,
     $     F_grid2_dims,F_grid2_corners,
     $     F_grid2_center_lat,F_grid2_center_lon,
     $     F_grid2_corner_lat,F_grid2_corner_lon)

!-----------------------------------------------------------------------
!
!     initialize some remapping variables.
!
!-----------------------------------------------------------------------

      call init_remap_vars()

!-----------------------------------------------------------------------
!
!     call appropriate interpolation setup routine based on type of
!     remapping requested.
!
!-----------------------------------------------------------------------

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
!     reduce size of remapping arrays and then write remapping info
!     to a file.
!
!-----------------------------------------------------------------------

      if (num_links_map1 /= max_links_map1) then
        call resize_remap_vars(1, num_links_map1-max_links_map1)
      endif
      if ((num_maps > 1) .and. (num_links_map2 /= max_links_map2)) then
        call resize_remap_vars(2, num_links_map2-max_links_map2)
      endif

c$$$      call write_remap(map1_name, map2_name,
c$$$     &                 interp_file1, interp_file2, output_opt)

      call sort_add(grid2_add_map1, grid1_add_map1, wts_map1)
      !num_links = SIZE(grid2_add_map1)  !grid2_add_map1(num_links_map1)
      !num_wts   = SIZE(wts_map1, DIM=1) !wts_map1(num_wts, max_links_map1)

      F_grid1_add_map1 => grid1_add_map1
      F_grid2_add_map1 => grid2_add_map1
      F_wts_map1       => wts_map1
      F_num_links = SIZE(grid2_add_map1)
      F_num_wts   = SIZE(wts_map1, DIM=1)

c$$$      if (num_maps > 1) then
c$$$        call sort_add(grid1_add_map2, grid2_add_map2, wts_map2)
c$$$        !num_links = SIZE(grid1_add_map2)
c$$$        !num_wts   = SIZE(wts_map2, DIM=1)
c$$$      endif

      call scrip_grid_finalize()
c$$$      call scrip_remap_vars_finalize(num_maps)

      !TODO: may have to deallocate all working arrays

!-----------------------------------------------------------------------
      return
      end subroutine scrip_addr_wts

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
