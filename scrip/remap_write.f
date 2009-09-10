!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     This module contains routines for writing the remapping data to 
!     a file.  Before writing the data for each mapping, the links are 
!     sorted by destination grid address.
!
!-----------------------------------------------------------------------
!
!     CVS:$Id: remap_write.f,v 1.7 2001/08/21 21:06:42 pwjones Exp $
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

      module remap_write

!-----------------------------------------------------------------------

      use kinds_mod     ! defines common data types
      use constants     ! defines common scalar constants
      use grids         ! module containing grid information
      use remap_vars    ! module containing remap information
c$$$      use netcdf_mod    ! module with netCDF stuff

      implicit none

!-----------------------------------------------------------------------
!
!     module variables
!
!-----------------------------------------------------------------------

      character(char_len), private :: 
     &   map_method       ! character string for map_type
     &,  normalize_opt    ! character string for normalization option
     &,  history          ! character string for history information
     &,  convention       ! character string for output convention

      character(8), private :: 
     &   cdate            ! character date string

      integer (kind=int_kind), dimension(:), allocatable, private ::
     &   src_mask_int     ! integer masks to determine
     &,  dst_mask_int     ! cells that participate in map

!-----------------------------------------------------------------------
!
!     various netCDF identifiers used by output routines
!
!-----------------------------------------------------------------------

c$$$      integer (kind=int_kind), private ::
c$$$     &   ncstat               ! error flag for netCDF calls 
c$$$     &,  nc_file_id           ! id for netCDF file
c$$$     &,  nc_srcgrdsize_id     ! id for source grid size
c$$$     &,  nc_dstgrdsize_id     ! id for destination grid size
c$$$     &,  nc_srcgrdcorn_id     ! id for number of source grid corners
c$$$     &,  nc_dstgrdcorn_id     ! id for number of dest grid corners
c$$$     &,  nc_srcgrdrank_id     ! id for source grid rank
c$$$     &,  nc_dstgrdrank_id     ! id for dest grid rank
c$$$     &,  nc_numlinks_id       ! id for number of links in mapping
c$$$     &,  nc_numwgts_id        ! id for number of weights for mapping
c$$$     &,  nc_srcgrddims_id     ! id for source grid dimensions
c$$$     &,  nc_dstgrddims_id     ! id for dest grid dimensions
c$$$     &,  nc_srcgrdcntrlat_id  ! id for source grid center latitude
c$$$     &,  nc_dstgrdcntrlat_id  ! id for dest grid center latitude
c$$$     &,  nc_srcgrdcntrlon_id  ! id for source grid center longitude
c$$$     &,  nc_dstgrdcntrlon_id  ! id for dest grid center longitude
c$$$     &,  nc_srcgrdimask_id    ! id for source grid mask
c$$$     &,  nc_dstgrdimask_id    ! id for dest grid mask
c$$$     &,  nc_srcgrdcrnrlat_id  ! id for latitude of source grid corners
c$$$     &,  nc_srcgrdcrnrlon_id  ! id for longitude of source grid corners
c$$$     &,  nc_dstgrdcrnrlat_id  ! id for latitude of dest grid corners
c$$$     &,  nc_dstgrdcrnrlon_id  ! id for longitude of dest grid corners
c$$$     &,  nc_srcgrdarea_id     ! id for area of source grid cells
c$$$     &,  nc_dstgrdarea_id     ! id for area of dest grid cells
c$$$     &,  nc_srcgrdfrac_id     ! id for area fraction on source grid
c$$$     &,  nc_dstgrdfrac_id     ! id for area fraction on dest grid
c$$$     &,  nc_srcadd_id         ! id for map source address
c$$$     &,  nc_dstadd_id         ! id for map destination address
c$$$     &,  nc_rmpmatrix_id      ! id for remapping matrix
c$$$
c$$$      integer (kind=int_kind), dimension(2), private ::
c$$$     &   nc_dims2_id  ! netCDF ids for 2d array dims

!***********************************************************************

      contains

!***********************************************************************

      subroutine write_remap(map1_name, map2_name, 
     &                       interp_file1, interp_file2, output_opt)

!-----------------------------------------------------------------------
!
!     calls correct output routine based on output format choice
!
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!
!     input variables
!
!-----------------------------------------------------------------------

      character(char_len), intent(in) ::
     &            map1_name,    ! name for mapping grid1 to grid2
     &            map2_name,    ! name for mapping grid2 to grid1
     &            interp_file1, ! filename for map1 remap data
     &            interp_file2, ! filename for map2 remap data
     &            output_opt    ! option for output conventions

!-----------------------------------------------------------------------
!
!     local variables
!
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!
!     define some common variables to be used in all routines
!
!-----------------------------------------------------------------------

      select case(norm_opt)
      case (norm_opt_none)
        normalize_opt = 'none'
      case (norm_opt_frcarea)
        normalize_opt = 'fracarea'
      case (norm_opt_dstarea)
        normalize_opt = 'destarea'
      end select

      select case(map_type)
      case(map_type_conserv)
        map_method = 'Conservative remapping'
      case(map_type_bilinear)
        map_method = 'Bilinear remapping'
      case(map_type_distwgt)
        map_method = 'Distance weighted avg of nearest neighbors'
      case(map_type_bicubic)
        map_method = 'Bicubic remapping'
      case default
        stop 'Invalid Map Type'
      end select

      call date_and_time(date=cdate)
      write (history,1000) cdate(5:6),cdate(7:8),cdate(1:4)
 1000 format('Created: ',a2,'-',a2,'-',a4)

!-----------------------------------------------------------------------
!
!     sort address and weight arrays
!
!-----------------------------------------------------------------------

c$$$      call sort_add(grid2_add_map1, grid1_add_map1, wts_map1)
c$$$      if (num_maps > 1) then
c$$$        call sort_add(grid1_add_map2, grid2_add_map2, wts_map2)
c$$$      endif

!-----------------------------------------------------------------------
!
!     call appropriate output routine
!
!-----------------------------------------------------------------------

c$$$      select case(output_opt)
c$$$      case ('scrip')
c$$$        call write_remap_scrip(map1_name, interp_file1, 1)
c$$$      case ('ncar-csm')
c$$$        call write_remap_csm  (map1_name, interp_file1, 1)
c$$$      case default
c$$$        stop 'unknown output file convention'
c$$$      end select

!-----------------------------------------------------------------------
!
!     call appropriate output routine for second mapping if required
!
!-----------------------------------------------------------------------

c$$$      if (num_maps > 1) then
c$$$        select case(output_opt)
c$$$        case ('scrip')
c$$$          call write_remap_scrip(map2_name, interp_file2, 2)
c$$$        case ('ncar-csm')
c$$$          call write_remap_csm  (map2_name, interp_file2, 2)
c$$$        case default
c$$$          stop 'unknown output file convention'
c$$$        end select
c$$$      endif

!-----------------------------------------------------------------------

      end subroutine write_remap

!***********************************************************************

c$$$      subroutine write_remap_scrip(map_name, interp_file, direction)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     writes remap data to a netCDF file using SCRIP conventions
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     input variables
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      character(char_len), intent(in) ::
c$$$     &            map_name     ! name for mapping 
c$$$     &,           interp_file  ! filename for remap data
c$$$
c$$$      integer (kind=int_kind), intent(in) ::
c$$$     &  direction              ! direction of map (1=grid1 to grid2
c$$$                               !                   2=grid2 to grid1)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     local variables
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      character(char_len) ::
c$$$     &  grid1_ctmp        ! character temp for grid1 names
c$$$     &, grid2_ctmp        ! character temp for grid2 names
c$$$
c$$$      integer (kind=int_kind) ::
c$$$     &  itmp1             ! integer temp
c$$$     &, itmp2             ! integer temp
c$$$     &, itmp3             ! integer temp
c$$$     &, itmp4             ! integer temp
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     create netCDF file for mapping and define some global attributes
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      ncstat = nf_create (interp_file, NF_CLOBBER, nc_file_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** map name
c$$$      !***
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'title',
c$$$     &                          len_trim(map_name), map_name)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** normalization option
c$$$      !***
c$$$      ncstat = nf_put_att_text(nc_file_id, NF_GLOBAL, 'normalization',
c$$$     &                         len_trim(normalize_opt), normalize_opt)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** map method
c$$$      !***
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'map_method',
c$$$     &                          len_trim(map_method), map_method)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** history
c$$$      !***
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'history',
c$$$     &                          len_trim(history), history)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** file convention
c$$$      !***
c$$$      convention = 'SCRIP'
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'conventions',
c$$$     &                          len_trim(convention), convention)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** source and destination grid names
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        grid1_ctmp = 'source_grid'
c$$$        grid2_ctmp = 'dest_grid'
c$$$      else
c$$$        grid1_ctmp = 'dest_grid'
c$$$        grid2_ctmp = 'source_grid'
c$$$      endif
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, trim(grid1_ctmp),
c$$$     &                          len_trim(grid1_name), grid1_name)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, trim(grid2_ctmp),
c$$$     &                          len_trim(grid2_name), grid2_name)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     prepare netCDF dimension info
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      !***
c$$$      !*** define grid size dimensions
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_size
c$$$        itmp2 = grid2_size
c$$$      else
c$$$        itmp1 = grid2_size
c$$$        itmp2 = grid1_size
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'src_grid_size', itmp1, 
c$$$     &                     nc_srcgrdsize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'dst_grid_size', itmp2, 
c$$$     &                     nc_dstgrdsize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid corner dimension
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_corners
c$$$        itmp2 = grid2_corners
c$$$      else
c$$$        itmp1 = grid2_corners
c$$$        itmp2 = grid1_corners
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'src_grid_corners', 
c$$$     &                     itmp1, nc_srcgrdcorn_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'dst_grid_corners', 
c$$$     &                     itmp2, nc_dstgrdcorn_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid rank dimension
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_rank
c$$$        itmp2 = grid2_rank
c$$$      else
c$$$        itmp1 = grid2_rank
c$$$        itmp2 = grid1_rank
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'src_grid_rank', 
c$$$     &                     itmp1, nc_srcgrdrank_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'dst_grid_rank', 
c$$$     &                     itmp2, nc_dstgrdrank_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define map size dimensions
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = num_links_map1
c$$$      else
c$$$        itmp1 = num_links_map2
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'num_links', 
c$$$     &                     itmp1, nc_numlinks_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'num_wgts', 
c$$$     &                     num_wts, nc_numwgts_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid dimensions
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_dims', NF_INT,
c$$$     &                     1, nc_srcgrdrank_id, nc_srcgrddims_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_dims', NF_INT,
c$$$     &                     1, nc_dstgrdrank_id, nc_dstgrddims_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     define all arrays for netCDF descriptors
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      !***
c$$$      !*** define grid center latitude array
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_center_lat', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdcntrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_center_lat', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdcntrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid center longitude array
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_center_lon', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdcntrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_center_lon', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdcntrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid corner lat/lon arrays
c$$$      !***
c$$$
c$$$      nc_dims2_id(1) = nc_srcgrdcorn_id
c$$$      nc_dims2_id(2) = nc_srcgrdsize_id
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_corner_lat', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_srcgrdcrnrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_corner_lon', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_srcgrdcrnrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      nc_dims2_id(1) = nc_dstgrdcorn_id
c$$$      nc_dims2_id(2) = nc_dstgrdsize_id
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_corner_lat', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_dstgrdcrnrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_corner_lon', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_dstgrdcrnrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define units for all coordinate arrays
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        grid1_ctmp = grid1_units
c$$$        grid2_ctmp = grid2_units
c$$$      else
c$$$        grid1_ctmp = grid2_units
c$$$        grid2_ctmp = grid1_units
c$$$      endif
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcntrlat_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcntrlat_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcntrlon_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcntrlon_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcrnrlat_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcrnrlon_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcrnrlat_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcrnrlon_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid mask
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_imask', NF_INT,
c$$$     &                     1, nc_srcgrdsize_id, nc_srcgrdimask_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdimask_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_imask', NF_INT,
c$$$     &                     1, nc_dstgrdsize_id, nc_dstgrdimask_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdimask_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid area arrays
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_area', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdarea_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdarea_id, 
c$$$     &                          'units', 14, 'square radians')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_area', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdarea_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdarea_id, 
c$$$     &                          'units', 14, 'square radians')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid fraction arrays
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_frac', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdfrac_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdfrac_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_frac', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdfrac_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdfrac_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define mapping arrays
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_address', 
c$$$     &                     NF_INT, 1, nc_numlinks_id, 
c$$$     &                     nc_srcadd_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_address', 
c$$$     &                     NF_INT, 1, nc_numlinks_id, 
c$$$     &                     nc_dstadd_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      nc_dims2_id(1) = nc_numwgts_id
c$$$      nc_dims2_id(2) = nc_numlinks_id
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'remap_matrix', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_rmpmatrix_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** end definition stage
c$$$      !***
c$$$
c$$$      ncstat = nf_enddef(nc_file_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     compute integer masks
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      if (direction == 1) then
c$$$        allocate (src_mask_int(grid1_size),
c$$$     &            dst_mask_int(grid2_size))
c$$$
c$$$        where (grid2_mask)
c$$$          dst_mask_int = 1
c$$$        elsewhere
c$$$          dst_mask_int = 0
c$$$        endwhere
c$$$
c$$$        where (grid1_mask)
c$$$          src_mask_int = 1
c$$$        elsewhere
c$$$          src_mask_int = 0
c$$$        endwhere
c$$$      else
c$$$        allocate (src_mask_int(grid2_size),
c$$$     &            dst_mask_int(grid1_size))
c$$$
c$$$        where (grid1_mask)
c$$$          dst_mask_int = 1
c$$$        elsewhere
c$$$          dst_mask_int = 0
c$$$        endwhere
c$$$
c$$$        where (grid2_mask)
c$$$          src_mask_int = 1
c$$$        elsewhere
c$$$          src_mask_int = 0
c$$$        endwhere
c$$$      endif
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     change units of lat/lon coordinates if input units different
c$$$!     from radians
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      if (grid1_units(1:7) == 'degrees' .and. direction == 1) then
c$$$        grid1_center_lat = grid1_center_lat/deg2rad
c$$$        grid1_center_lon = grid1_center_lon/deg2rad
c$$$        grid1_corner_lat = grid1_corner_lat/deg2rad
c$$$        grid1_corner_lon = grid1_corner_lon/deg2rad
c$$$      endif
c$$$
c$$$      if (grid2_units(1:7) == 'degrees' .and. direction == 1) then
c$$$        grid2_center_lat = grid2_center_lat/deg2rad
c$$$        grid2_center_lon = grid2_center_lon/deg2rad
c$$$        grid2_corner_lat = grid2_corner_lat/deg2rad
c$$$        grid2_corner_lon = grid2_corner_lon/deg2rad
c$$$      endif
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     write mapping data
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_srcgrddims_id
c$$$        itmp2 = nc_dstgrddims_id
c$$$      else
c$$$        itmp2 = nc_srcgrddims_id
c$$$        itmp1 = nc_dstgrddims_id
c$$$      endif
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, itmp1, grid1_dims)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, itmp2, grid2_dims)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, nc_srcgrdimask_id, 
c$$$     &                        src_mask_int)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, nc_dstgrdimask_id,
c$$$     &                        dst_mask_int)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      deallocate(src_mask_int, dst_mask_int)
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_srcgrdcntrlat_id
c$$$        itmp2 = nc_srcgrdcntrlon_id
c$$$        itmp3 = nc_srcgrdcrnrlat_id
c$$$        itmp4 = nc_srcgrdcrnrlon_id
c$$$      else
c$$$        itmp1 = nc_dstgrdcntrlat_id
c$$$        itmp2 = nc_dstgrdcntrlon_id
c$$$        itmp3 = nc_dstgrdcrnrlat_id
c$$$        itmp4 = nc_dstgrdcrnrlon_id
c$$$      endif
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp1, grid1_center_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp2, grid1_center_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp3, grid1_corner_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp4, grid1_corner_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_dstgrdcntrlat_id
c$$$        itmp2 = nc_dstgrdcntrlon_id
c$$$        itmp3 = nc_dstgrdcrnrlat_id
c$$$        itmp4 = nc_dstgrdcrnrlon_id
c$$$      else
c$$$        itmp1 = nc_srcgrdcntrlat_id
c$$$        itmp2 = nc_srcgrdcntrlon_id
c$$$        itmp3 = nc_srcgrdcrnrlat_id
c$$$        itmp4 = nc_srcgrdcrnrlon_id
c$$$      endif
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp1, grid2_center_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp2, grid2_center_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp3, grid2_corner_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp4, grid2_corner_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_srcgrdarea_id
c$$$        itmp2 = nc_srcgrdfrac_id
c$$$        itmp3 = nc_dstgrdarea_id
c$$$        itmp4 = nc_dstgrdfrac_id
c$$$      else
c$$$        itmp1 = nc_dstgrdarea_id
c$$$        itmp2 = nc_dstgrdfrac_id
c$$$        itmp3 = nc_srcgrdarea_id
c$$$        itmp4 = nc_srcgrdfrac_id
c$$$      endif
c$$$
c$$$      if (luse_grid1_area) then
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp1, grid1_area_in)
c$$$      else
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp1, grid1_area)
c$$$      endif
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp2, grid1_frac)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (luse_grid2_area) then
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp3, grid2_area_in)
c$$$      else
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp3, grid2_area)
c$$$      endif
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp4, grid2_frac)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (direction == 1) then
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_srcadd_id, 
c$$$     &                          grid1_add_map1)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_dstadd_id, 
c$$$     &                          grid2_add_map1)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix_id, 
c$$$     &                             wts_map1)
c$$$        call netcdf_error_handler(ncstat)
c$$$      else
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_srcadd_id, 
c$$$     &                          grid2_add_map2)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_dstadd_id, 
c$$$     &                          grid1_add_map2)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix_id, 
c$$$     &                             wts_map2)
c$$$        call netcdf_error_handler(ncstat)
c$$$      endif
c$$$
c$$$      ncstat = nf_close(nc_file_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      end subroutine write_remap_scrip

!***********************************************************************

c$$$      subroutine write_remap_csm(map_name, interp_file, direction)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     writes remap data to a netCDF file using NCAR-CSM conventions
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     input variables
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      character(char_len), intent(in) ::
c$$$     &            map_name     ! name for mapping 
c$$$     &,           interp_file  ! filename for remap data
c$$$
c$$$      integer (kind=int_kind), intent(in) ::
c$$$     &  direction              ! direction of map (1=grid1 to grid2
c$$$                               !                   2=grid2 to grid1)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     local variables
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      character(char_len) ::
c$$$     &  grid1_ctmp        ! character temp for grid1 names
c$$$     &, grid2_ctmp        ! character temp for grid2 names
c$$$
c$$$      integer (kind=int_kind) ::
c$$$     &  itmp1             ! integer temp
c$$$     &, itmp2             ! integer temp
c$$$     &, itmp3             ! integer temp
c$$$     &, itmp4             ! integer temp
c$$$     &, nc_numwgts1_id    ! extra netCDF id for additional weights
c$$$     &, nc_src_isize_id   ! extra netCDF id for ni_a
c$$$     &, nc_src_jsize_id   ! extra netCDF id for nj_a
c$$$     &, nc_dst_isize_id   ! extra netCDF id for ni_b
c$$$     &, nc_dst_jsize_id   ! extra netCDF id for nj_b
c$$$     &, nc_rmpmatrix2_id  ! extra netCDF id for high-order remap matrix
c$$$
c$$$      real (kind=dbl_kind), dimension(:),allocatable ::
c$$$     &  wts1              ! CSM wants single array for 1st-order wts
c$$$
c$$$      real (kind=dbl_kind), dimension(:,:),allocatable ::
c$$$     &  wts2              ! write remaining weights in different array
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     create netCDF file for mapping and define some global attributes
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      ncstat = nf_create (interp_file, NF_CLOBBER, nc_file_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** map name
c$$$      !***
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'title',
c$$$     &                          len_trim(map_name), map_name)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** normalization option
c$$$      !***
c$$$      ncstat = nf_put_att_text(nc_file_id, NF_GLOBAL, 'normalization',
c$$$     &                         len_trim(normalize_opt), normalize_opt)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** map method
c$$$      !***
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'map_method',
c$$$     &                          len_trim(map_method), map_method)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** history
c$$$      !***
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'history',
c$$$     &                          len_trim(history), history)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** file convention
c$$$      !***
c$$$      convention = 'NCAR-CSM'
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, 'conventions',
c$$$     &                          len_trim(convention), convention)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** source and destination grid names
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        grid1_ctmp = 'domain_a'
c$$$        grid2_ctmp = 'domain_b'
c$$$      else
c$$$        grid1_ctmp = 'domain_b'
c$$$        grid2_ctmp = 'domain_a'
c$$$      endif
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, trim(grid1_ctmp),
c$$$     &                          len_trim(grid1_name), grid1_name)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, NF_GLOBAL, trim(grid2_ctmp),
c$$$     &                          len_trim(grid2_name), grid2_name)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     prepare netCDF dimension info
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      !***
c$$$      !*** define grid size dimensions
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_size
c$$$        itmp2 = grid2_size
c$$$      else
c$$$        itmp1 = grid2_size
c$$$        itmp2 = grid1_size
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'n_a', itmp1, nc_srcgrdsize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'n_b', itmp2, nc_dstgrdsize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid corner dimension
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_corners
c$$$        itmp2 = grid2_corners
c$$$      else
c$$$        itmp1 = grid2_corners
c$$$        itmp2 = grid1_corners
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'nv_a', itmp1, nc_srcgrdcorn_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'nv_b', itmp2, nc_dstgrdcorn_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid rank dimension
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_rank
c$$$        itmp2 = grid2_rank
c$$$      else
c$$$        itmp1 = grid2_rank
c$$$        itmp2 = grid1_rank
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'src_grid_rank', 
c$$$     &                     itmp1, nc_srcgrdrank_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'dst_grid_rank', 
c$$$     &                     itmp2, nc_dstgrdrank_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define first two dims as if 2-d cartesian domain
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = grid1_dims(1)
c$$$        if (grid1_rank > 1) then
c$$$          itmp2 = grid1_dims(2)
c$$$        else
c$$$          itmp2 = 0
c$$$        endif
c$$$        itmp3 = grid2_dims(1)
c$$$        if (grid2_rank > 1) then
c$$$          itmp4 = grid2_dims(2)
c$$$        else
c$$$          itmp4 = 0
c$$$        endif
c$$$      else
c$$$        itmp1 = grid2_dims(1)
c$$$        if (grid2_rank > 1) then
c$$$          itmp2 = grid2_dims(2)
c$$$        else
c$$$          itmp2 = 0
c$$$        endif
c$$$        itmp3 = grid1_dims(1)
c$$$        if (grid1_rank > 1) then
c$$$          itmp4 = grid1_dims(2)
c$$$        else
c$$$          itmp4 = 0
c$$$        endif
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'ni_a', itmp1, nc_src_isize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'nj_a', itmp2, nc_src_jsize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'ni_b', itmp3, nc_dst_isize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'nj_b', itmp4, nc_dst_jsize_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define map size dimensions
c$$$      !***
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = num_links_map1
c$$$      else
c$$$        itmp1 = num_links_map2
c$$$      endif
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'n_s', itmp1, nc_numlinks_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_dim (nc_file_id, 'num_wgts', 
c$$$     &                     num_wts, nc_numwgts_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (num_wts > 1) then
c$$$        ncstat = nf_def_dim (nc_file_id, 'num_wgts1', 
c$$$     &                       num_wts-1, nc_numwgts1_id)
c$$$        call netcdf_error_handler(ncstat)
c$$$      endif
c$$$
c$$$      !***
c$$$      !*** define grid dimensions
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'src_grid_dims', NF_INT,
c$$$     &                     1, nc_srcgrdrank_id, nc_srcgrddims_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'dst_grid_dims', NF_INT,
c$$$     &                     1, nc_dstgrdrank_id, nc_dstgrddims_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     define all arrays for netCDF descriptors
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      !***
c$$$      !*** define grid center latitude array
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'yc_a',
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdcntrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'yc_b', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdcntrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid center longitude array
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'xc_a', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdcntrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'xc_b', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdcntrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid corner lat/lon arrays
c$$$      !***
c$$$
c$$$      nc_dims2_id(1) = nc_srcgrdcorn_id
c$$$      nc_dims2_id(2) = nc_srcgrdsize_id
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'yv_a', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_srcgrdcrnrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'xv_a', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_srcgrdcrnrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      nc_dims2_id(1) = nc_dstgrdcorn_id
c$$$      nc_dims2_id(2) = nc_dstgrdsize_id
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'yv_b', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_dstgrdcrnrlat_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'xv_b', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_dstgrdcrnrlon_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** CSM wants all in degrees
c$$$      !***
c$$$
c$$$      grid1_units = 'degrees'
c$$$      grid2_units = 'degrees'
c$$$
c$$$      if (direction == 1) then
c$$$        grid1_ctmp = grid1_units
c$$$        grid2_ctmp = grid2_units
c$$$      else
c$$$        grid1_ctmp = grid2_units
c$$$        grid2_ctmp = grid1_units
c$$$      endif
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcntrlat_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcntrlat_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcntrlon_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcntrlon_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcrnrlat_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdcrnrlon_id, 
c$$$     &                          'units', 7, grid1_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcrnrlat_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdcrnrlon_id, 
c$$$     &                          'units', 7, grid2_ctmp)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid mask
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'mask_a', NF_INT,
c$$$     &                     1, nc_srcgrdsize_id, nc_srcgrdimask_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdimask_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'mask_b', NF_INT,
c$$$     &                     1, nc_dstgrdsize_id, nc_dstgrdimask_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdimask_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid area arrays
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'area_a', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdarea_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdarea_id, 
c$$$     &                          'units', 14, 'square radians')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'area_b', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdarea_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdarea_id, 
c$$$     &                          'units', 14, 'square radians')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define grid fraction arrays
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'frac_a', 
c$$$     &                     NF_DOUBLE, 1, nc_srcgrdsize_id, 
c$$$     &                     nc_srcgrdfrac_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_srcgrdfrac_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'frac_b', 
c$$$     &                     NF_DOUBLE, 1, nc_dstgrdsize_id, 
c$$$     &                     nc_dstgrdfrac_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_att_text (nc_file_id, nc_dstgrdfrac_id, 
c$$$     &                          'units', 8, 'unitless')
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      !***
c$$$      !*** define mapping arrays
c$$$      !***
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'col', 
c$$$     &                     NF_INT, 1, nc_numlinks_id, 
c$$$     &                     nc_srcadd_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'row', 
c$$$     &                     NF_INT, 1, nc_numlinks_id, 
c$$$     &                     nc_dstadd_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_def_var (nc_file_id, 'S', 
c$$$     &                     NF_DOUBLE, 1, nc_numlinks_id, 
c$$$     &                     nc_rmpmatrix_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (num_wts > 1) then
c$$$        nc_dims2_id(1) = nc_numwgts1_id
c$$$        nc_dims2_id(2) = nc_numlinks_id
c$$$
c$$$        ncstat = nf_def_var (nc_file_id, 'S2', 
c$$$     &                     NF_DOUBLE, 2, nc_dims2_id, 
c$$$     &                     nc_rmpmatrix2_id)
c$$$        call netcdf_error_handler(ncstat)
c$$$      endif
c$$$
c$$$      !***
c$$$      !*** end definition stage
c$$$      !***
c$$$
c$$$      ncstat = nf_enddef(nc_file_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     compute integer masks
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      if (direction == 1) then
c$$$        allocate (src_mask_int(grid1_size),
c$$$     &            dst_mask_int(grid2_size))
c$$$
c$$$        where (grid2_mask)
c$$$          dst_mask_int = 1
c$$$        elsewhere
c$$$          dst_mask_int = 0
c$$$        endwhere
c$$$
c$$$        where (grid1_mask)
c$$$          src_mask_int = 1
c$$$        elsewhere
c$$$          src_mask_int = 0
c$$$        endwhere
c$$$      else
c$$$        allocate (src_mask_int(grid2_size),
c$$$     &            dst_mask_int(grid1_size))
c$$$
c$$$        where (grid1_mask)
c$$$          dst_mask_int = 1
c$$$        elsewhere
c$$$          dst_mask_int = 0
c$$$        endwhere
c$$$
c$$$        where (grid2_mask)
c$$$          src_mask_int = 1
c$$$        elsewhere
c$$$          src_mask_int = 0
c$$$        endwhere
c$$$      endif
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     change units of lat/lon coordinates if input units different
c$$$!     from radians. if this is the second mapping, the conversion has
c$$$!     alread been done.
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      if (grid1_units(1:7) == 'degrees' .and. direction == 1) then
c$$$        grid1_center_lat = grid1_center_lat/deg2rad
c$$$        grid1_center_lon = grid1_center_lon/deg2rad
c$$$        grid1_corner_lat = grid1_corner_lat/deg2rad
c$$$        grid1_corner_lon = grid1_corner_lon/deg2rad
c$$$      endif
c$$$
c$$$      if (grid2_units(1:7) == 'degrees' .and. direction == 1) then
c$$$        grid2_center_lat = grid2_center_lat/deg2rad
c$$$        grid2_center_lon = grid2_center_lon/deg2rad
c$$$        grid2_corner_lat = grid2_corner_lat/deg2rad
c$$$        grid2_corner_lon = grid2_corner_lon/deg2rad
c$$$      endif
c$$$
c$$$!-----------------------------------------------------------------------
c$$$!
c$$$!     write mapping data
c$$$!
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_srcgrddims_id
c$$$        itmp2 = nc_dstgrddims_id
c$$$      else
c$$$        itmp2 = nc_srcgrddims_id
c$$$        itmp1 = nc_dstgrddims_id
c$$$      endif
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, itmp1, grid1_dims)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, itmp2, grid2_dims)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, nc_srcgrdimask_id, 
c$$$     &                        src_mask_int)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_int(nc_file_id, nc_dstgrdimask_id,
c$$$     &                        dst_mask_int)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      deallocate(src_mask_int, dst_mask_int)
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_srcgrdcntrlat_id
c$$$        itmp2 = nc_srcgrdcntrlon_id
c$$$        itmp3 = nc_srcgrdcrnrlat_id
c$$$        itmp4 = nc_srcgrdcrnrlon_id
c$$$      else
c$$$        itmp1 = nc_dstgrdcntrlat_id
c$$$        itmp2 = nc_dstgrdcntrlon_id
c$$$        itmp3 = nc_dstgrdcrnrlat_id
c$$$        itmp4 = nc_dstgrdcrnrlon_id
c$$$      endif
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp1, grid1_center_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp2, grid1_center_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp3, grid1_corner_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp4, grid1_corner_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_dstgrdcntrlat_id
c$$$        itmp2 = nc_dstgrdcntrlon_id
c$$$        itmp3 = nc_dstgrdcrnrlat_id
c$$$        itmp4 = nc_dstgrdcrnrlon_id
c$$$      else
c$$$        itmp1 = nc_srcgrdcntrlat_id
c$$$        itmp2 = nc_srcgrdcntrlon_id
c$$$        itmp3 = nc_srcgrdcrnrlat_id
c$$$        itmp4 = nc_srcgrdcrnrlon_id
c$$$      endif
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp1, grid2_center_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp2, grid2_center_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp3, grid2_corner_lat)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp4, grid2_corner_lon)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (direction == 1) then
c$$$        itmp1 = nc_srcgrdarea_id
c$$$        itmp2 = nc_srcgrdfrac_id
c$$$        itmp3 = nc_dstgrdarea_id
c$$$        itmp4 = nc_dstgrdfrac_id
c$$$      else
c$$$        itmp1 = nc_dstgrdarea_id
c$$$        itmp2 = nc_dstgrdfrac_id
c$$$        itmp3 = nc_srcgrdarea_id
c$$$        itmp4 = nc_srcgrdfrac_id
c$$$      endif
c$$$
c$$$      if (luse_grid1_area) then
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp1, grid1_area_in)
c$$$      else
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp1, grid1_area)
c$$$      endif
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp2, grid1_frac)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (luse_grid2_area) then
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp3, grid2_area)
c$$$      else
c$$$        ncstat = nf_put_var_double(nc_file_id, itmp3, grid2_area)
c$$$      endif
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      ncstat = nf_put_var_double(nc_file_id, itmp4, grid2_frac)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$      if (direction == 1) then
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_srcadd_id, 
c$$$     &                          grid1_add_map1)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_dstadd_id, 
c$$$     &                          grid2_add_map1)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        if (num_wts == 1) then
c$$$          ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix_id, 
c$$$     &                               wts_map1)
c$$$          call netcdf_error_handler(ncstat)
c$$$        else
c$$$          allocate(wts1(num_links_map1),wts2(num_wts-1,num_links_map1))
c$$$
c$$$          wts1 = wts_map1(1,:)
c$$$          wts2 = wts_map1(2:,:)
c$$$
c$$$          ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix_id, 
c$$$     &                               wts1)
c$$$          call netcdf_error_handler(ncstat)
c$$$          ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix2_id, 
c$$$     &                               wts2)
c$$$          call netcdf_error_handler(ncstat)
c$$$          deallocate(wts1,wts2)
c$$$        endif
c$$$      else
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_srcadd_id, 
c$$$     &                          grid2_add_map2)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        ncstat = nf_put_var_int(nc_file_id, nc_dstadd_id, 
c$$$     &                          grid1_add_map2)
c$$$        call netcdf_error_handler(ncstat)
c$$$
c$$$        if (num_wts == 1) then
c$$$          ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix_id, 
c$$$     &                               wts_map2)
c$$$          call netcdf_error_handler(ncstat)
c$$$        else
c$$$          allocate(wts1(num_links_map2),wts2(num_wts-1,num_links_map2))
c$$$
c$$$          wts1 = wts_map2(1,:)
c$$$          wts2 = wts_map2(2:,:)
c$$$
c$$$          ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix_id, 
c$$$     &                               wts1)
c$$$          call netcdf_error_handler(ncstat)
c$$$          ncstat = nf_put_var_double(nc_file_id, nc_rmpmatrix2_id, 
c$$$     &                               wts2)
c$$$          call netcdf_error_handler(ncstat)
c$$$          deallocate(wts1,wts2)
c$$$        endif
c$$$      endif
c$$$
c$$$      ncstat = nf_close(nc_file_id)
c$$$      call netcdf_error_handler(ncstat)
c$$$
c$$$!-----------------------------------------------------------------------
c$$$
c$$$      end subroutine write_remap_csm

!***********************************************************************

      subroutine sort_add(add1, add2, weights)

!-----------------------------------------------------------------------
!
!     this routine sorts address and weight arrays based on the
!     destination address with the source address as a secondary
!     sorting criterion.  the method is a standard heap sort.
!
!-----------------------------------------------------------------------

      use kinds_mod     ! defines common data types
      use constants     ! defines common scalar constants

      implicit none

!-----------------------------------------------------------------------
!
!     Input and Output arrays
!
!-----------------------------------------------------------------------

      integer (kind=int_kind), intent(inout), dimension(:) ::
     &        add1,       ! destination address array (num_links)
     &        add2        ! source      address array

      real (kind=dbl_kind), intent(inout), dimension(:,:) ::
     &        weights     ! remapping weights (num_wts, num_links)

!-----------------------------------------------------------------------
!
!     local variables
!
!-----------------------------------------------------------------------

      integer (kind=int_kind) ::
     &          num_links,          ! num of links for this mapping
     &          num_wts,            ! num of weights for this mapping
     &          add1_tmp, add2_tmp, ! temp for addresses during swap
     &          nwgt,
     &          lvl, final_lvl,     ! level indexes for heap sort levels
     &          chk_lvl1, chk_lvl2, max_lvl

      real (kind=dbl_kind), dimension(SIZE(weights,DIM=1)) ::
     &          wgttmp              ! temp for holding wts during swap

!-----------------------------------------------------------------------
!
!     determine total number of links to sort and number of weights
!
!-----------------------------------------------------------------------

      num_links = SIZE(add1)
      num_wts   = SIZE(weights, DIM=1)

!-----------------------------------------------------------------------
!
!     start at the lowest level (N/2) of the tree and sift lower 
!     values to the bottom of the tree, promoting the larger numbers
!
!-----------------------------------------------------------------------

      do lvl=num_links/2,1,-1

        final_lvl = lvl
        add1_tmp = add1(lvl)
        add2_tmp = add2(lvl)
        wgttmp(:) = weights(:,lvl)

        !***
        !*** loop until proper level is found for this link, or reach
        !*** bottom
        !***

        sift_loop1: do

          !***
          !*** find the largest of the two daughters
          !***

          chk_lvl1 = 2*final_lvl
          chk_lvl2 = 2*final_lvl+1
          if (chk_lvl1 .EQ. num_links) chk_lvl2 = chk_lvl1

          if ((add1(chk_lvl1) >  add1(chk_lvl2)) .OR.
     &       ((add1(chk_lvl1) == add1(chk_lvl2)) .AND.
     &        (add2(chk_lvl1) >  add2(chk_lvl2)))) then
            max_lvl = chk_lvl1
          else 
            max_lvl = chk_lvl2
          endif

          !***
          !*** if the parent is greater than both daughters,
          !*** the correct level has been found
          !***

          if ((add1_tmp .GT. add1(max_lvl)) .OR.
     &       ((add1_tmp .EQ. add1(max_lvl)) .AND.
     &        (add2_tmp .GT. add2(max_lvl)))) then
            add1(final_lvl) = add1_tmp
            add2(final_lvl) = add2_tmp
            weights(:,final_lvl) = wgttmp(:)
            exit sift_loop1

          !***
          !*** otherwise, promote the largest daughter and push
          !*** down one level in the tree.  if haven't reached
          !*** the end of the tree, repeat the process.  otherwise
          !*** store last values and exit the loop
          !***

          else 
            add1(final_lvl) = add1(max_lvl)
            add2(final_lvl) = add2(max_lvl)
            weights(:,final_lvl) = weights(:,max_lvl)

            final_lvl = max_lvl
            if (2*final_lvl > num_links) then
              add1(final_lvl) = add1_tmp
              add2(final_lvl) = add2_tmp
              weights(:,final_lvl) = wgttmp(:)
              exit sift_loop1
            endif
          endif
        end do sift_loop1
      end do

!-----------------------------------------------------------------------
!
!     now that the heap has been sorted, strip off the top (largest)
!     value and promote the values below
!
!-----------------------------------------------------------------------

      do lvl=num_links,3,-1

        !***
        !*** move the top value and insert it into the correct place
        !***

        add1_tmp = add1(lvl)
        add1(lvl) = add1(1)

        add2_tmp = add2(lvl)
        add2(lvl) = add2(1)

        wgttmp(:) = weights(:,lvl)
        weights(:,lvl) = weights(:,1)

        !***
        !*** as above this loop sifts the tmp values down until proper 
        !*** level is reached
        !***

        final_lvl = 1

        sift_loop2: do

          !***
          !*** find the largest of the two daughters
          !***

          chk_lvl1 = 2*final_lvl
          chk_lvl2 = 2*final_lvl+1
          if (chk_lvl2 >= lvl) chk_lvl2 = chk_lvl1

          if ((add1(chk_lvl1) >  add1(chk_lvl2)) .OR.
     &       ((add1(chk_lvl1) == add1(chk_lvl2)) .AND.
     &        (add2(chk_lvl1) >  add2(chk_lvl2)))) then
            max_lvl = chk_lvl1
          else 
            max_lvl = chk_lvl2
          endif

          !***
          !*** if the parent is greater than both daughters,
          !*** the correct level has been found
          !***

          if ((add1_tmp >  add1(max_lvl)) .OR.
     &       ((add1_tmp == add1(max_lvl)) .AND.
     &        (add2_tmp >  add2(max_lvl)))) then
            add1(final_lvl) = add1_tmp
            add2(final_lvl) = add2_tmp
            weights(:,final_lvl) = wgttmp(:)
            exit sift_loop2

          !***
          !*** otherwise, promote the largest daughter and push
          !*** down one level in the tree.  if haven't reached
          !*** the end of the tree, repeat the process.  otherwise
          !*** store last values and exit the loop
          !***

          else 
            add1(final_lvl) = add1(max_lvl)
            add2(final_lvl) = add2(max_lvl)
            weights(:,final_lvl) = weights(:,max_lvl)

            final_lvl = max_lvl
            if (2*final_lvl >= lvl) then
              add1(final_lvl) = add1_tmp
              add2(final_lvl) = add2_tmp
              weights(:,final_lvl) = wgttmp(:)
              exit sift_loop2
            endif
          endif
        end do sift_loop2
      end do

      !***
      !*** swap the last two entries
      !***


      add1_tmp = add1(2)
      add1(2)  = add1(1)
      add1(1)  = add1_tmp

      add2_tmp = add2(2)
      add2(2)  = add2(1)
      add2(1)  = add2_tmp

      wgttmp (:)   = weights(:,2)
      weights(:,2) = weights(:,1)
      weights(:,1) = wgttmp (:)

!-----------------------------------------------------------------------

      end subroutine sort_add

!***********************************************************************

      end module remap_write

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
