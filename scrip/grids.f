!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     This module reads in and initializes two grids for remapping.
!     NOTE: grid1 must be the master grid -- the grid that determines
!           which cells participate (e.g. land mask) and the fractional
!           area of grid2 cells that participate in the remapping.
!
!-----------------------------------------------------------------------
!
!     CVS:$Id: grids.f,v 1.6 2001/08/21 21:06:41 pwjones Exp $
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

      module grids

!-----------------------------------------------------------------------

      use kinds_mod    ! defines data types
      use constants    ! common constants
c$$$      use iounits      ! I/O unit manager
c$$$      use netcdf_mod   ! netCDF stuff

      implicit none

!-----------------------------------------------------------------------
!
!     variables that describe each grid
!
!-----------------------------------------------------------------------

      integer (kind=int_kind), save ::
     &             grid1_size, grid2_size, ! total points on each grid
     &             grid1_rank, grid2_rank, ! rank of each grid
     &             grid1_corners, grid2_corners ! number of corners
                                                ! for each grid cell

      integer (kind=int_kind), dimension(:), allocatable, save ::
     &             grid1_dims, grid2_dims  ! size of each grid dimension

      character(char_len), save ::
     &             grid1_name, grid2_name  ! name for each grid

      character (char_len), save ::
     &             grid1_units, ! units for grid coords (degs/radians)
     &             grid2_units  ! units for grid coords

      real (kind=dbl_kind), parameter ::
     &      deg2rad = pi/180.   ! conversion for deg to rads

!-----------------------------------------------------------------------
!
!     grid coordinates and masks
!
!-----------------------------------------------------------------------

      logical (kind=log_kind), dimension(:), pointer, save ::
     &             grid1_mask,        ! flag which cells participate
     &             grid2_mask         ! flag which cells participate

      real (kind=dbl_kind), dimension(:), pointer, save ::
     &             grid1_center_lat,  ! lat/lon coordinates for
     &             grid1_center_lon,  ! each grid center in radians
     &             grid2_center_lat,
     &             grid2_center_lon,
     &             grid1_area,        ! tot area of each grid1 cell
     &             grid2_area,        ! tot area of each grid2 cell
     &             grid1_area_in,     ! area of grid1 cell from file
     &             grid2_area_in,     ! area of grid2 cell from file
     &             grid1_frac,        ! fractional area of grid cells
     &             grid2_frac         ! participating in remapping

      real (kind=dbl_kind), dimension(:,:), pointer, save ::
     &             grid1_corner_lat,  ! lat/lon coordinates for
     &             grid1_corner_lon,  ! each grid corner in radians
     &             grid2_corner_lat,
     &             grid2_corner_lon

      logical (kind=log_kind), save ::
     &             luse_grid_centers ! use centers for bounding boxes
     &,            luse_grid1_area   ! use area from grid file
     &,            luse_grid2_area   ! use area from grid file

      real (kind=dbl_kind), dimension(:,:), allocatable, save ::
     &             grid1_bound_box,  ! lat/lon bounding box for use
     &             grid2_bound_box   ! in restricting grid searches

!-----------------------------------------------------------------------
!
!     bins for restricting searches
!
!-----------------------------------------------------------------------

      character (char_len), save ::
     &        restrict_type  ! type of bins to use

      integer (kind=int_kind), save ::
     &        num_srch_bins  ! num of bins for restricted srch

      integer (kind=int_kind), dimension(:,:), allocatable, save ::
     &        bin_addr1, ! min,max adds for grid1 cells in this lat bin
     &        bin_addr2  ! min,max adds for grid2 cells in this lat bin

      real(kind=dbl_kind), dimension(:,:), allocatable, save ::
     &        bin_lats   ! min,max latitude for each search bin
     &,       bin_lons   ! min,max longitude for each search bin

!***********************************************************************

      contains

!***********************************************************************

c$$$      subroutine grid_init(grid1_file, grid2_file)
      subroutine grid_init()

!-----------------------------------------------------------------------
!
!     this routine reads grid info from grid files and makes any
!     necessary changes (e.g. for 0,2pi longitude range)
!
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!
!     input variables
!
!-----------------------------------------------------------------------

c$$$      character(char_len), intent(in) ::
c$$$     &             grid1_file, grid2_file  ! grid data files

!-----------------------------------------------------------------------
!
!     local variables
!
!-----------------------------------------------------------------------

      integer (kind=int_kind) ::
     &  n      ! loop counter
     &, nele   ! element loop counter
     &, iunit  ! unit number for opening files
     &, i,j    ! logical 2d addresses
     &, ip1,jp1
     &, n_add, e_add, ne_add
     &, nx, ny

c$$$      integer (kind=int_kind) ::
c$$$     &         ncstat,           ! netCDF status variable
c$$$     &         nc_grid1_id,       ! netCDF grid file id
c$$$     &         nc_grid2_id,       ! netCDF grid file id
c$$$     &         nc_grid1size_id,   ! netCDF grid size dim id
c$$$     &         nc_grid2size_id,   ! netCDF grid size dim id
c$$$     &         nc_grid1corn_id,   ! netCDF grid corner dim id
c$$$     &         nc_grid2corn_id,   ! netCDF grid corner dim id
c$$$     &         nc_grid1rank_id,   ! netCDF grid rank dim id
c$$$     &         nc_grid2rank_id,   ! netCDF grid rank dim id
c$$$     &         nc_grid1area_id,   ! netCDF grid rank dim id
c$$$     &         nc_grid2area_id,   ! netCDF grid rank dim id
c$$$     &         nc_grid1dims_id,   ! netCDF grid dimension size id
c$$$     &         nc_grid2dims_id,   ! netCDF grid dimension size id
c$$$     &         nc_grd1imask_id,   ! netCDF grid imask var id
c$$$     &         nc_grd2imask_id,   ! netCDF grid imask var id
c$$$     &         nc_grd1crnrlat_id, ! netCDF grid corner lat var id
c$$$     &         nc_grd2crnrlat_id, ! netCDF grid corner lat var id
c$$$     &         nc_grd1crnrlon_id, ! netCDF grid corner lon var id
c$$$     &         nc_grd2crnrlon_id, ! netCDF grid corner lon var id
c$$$     &         nc_grd1cntrlat_id, ! netCDF grid center lat var id
c$$$     &         nc_grd2cntrlat_id, ! netCDF grid center lat var id
c$$$     &         nc_grd1cntrlon_id, ! netCDF grid center lon var id
c$$$     &         nc_grd2cntrlon_id  ! netCDF grid center lon var id

c$$$      integer (kind=int_kind), dimension(:), allocatable ::
c$$$     &                            imask ! integer mask read from file

      real (kind=dbl_kind) ::
     &  dlat,dlon           ! lat/lon intervals for search bins

      real (kind=dbl_kind), dimension(4) ::
     &  tmp_lats, tmp_lons  ! temps for computing bounding boxes

!-----------------------------------------------------------------------
!
!     open grid files and read grid size/name data
!
!-----------------------------------------------------------------------

c$$$      TODO: grid1_size,grid2_size,grid1_rank,grid2_rank,grid1_corners,grid2_corners,grid1_name,grid2_name

c$$$      ncstat = nf_open(grid1_file, NF_NOWRITE, nc_grid1_id)
c$$$
c$$$      ncstat = nf_open(grid2_file, NF_NOWRITE, nc_grid2_id)
c$$$
c$$$      ncstat = nf_inq_dimid(nc_grid1_id, 'grid_size', nc_grid1size_id)
c$$$      ncstat = nf_inq_dimlen(nc_grid1_id, nc_grid1size_id, grid1_size)
c$$$
c$$$      ncstat = nf_inq_dimid(nc_grid2_id, 'grid_size', nc_grid2size_id)
c$$$      ncstat = nf_inq_dimlen(nc_grid2_id, nc_grid2size_id, grid2_size)
c$$$
c$$$      ncstat = nf_inq_dimid(nc_grid1_id, 'grid_rank', nc_grid1rank_id)
c$$$      ncstat = nf_inq_dimlen(nc_grid1_id, nc_grid1rank_id, grid1_rank)
c$$$
c$$$      ncstat = nf_inq_dimid(nc_grid2_id, 'grid_rank', nc_grid2rank_id)
c$$$      ncstat = nf_inq_dimlen(nc_grid2_id, nc_grid2rank_id, grid2_rank)
c$$$
c$$$      ncstat = nf_inq_dimid(nc_grid1_id,'grid_corners',nc_grid1corn_id)
c$$$      ncstat = nf_inq_dimlen(nc_grid1_id,nc_grid1corn_id,grid1_corners)
c$$$
c$$$      ncstat = nf_inq_dimid(nc_grid2_id,'grid_corners',nc_grid2corn_id)
c$$$      ncstat = nf_inq_dimlen(nc_grid2_id,nc_grid2corn_id,grid2_corners)

c$$$      allocate( grid1_dims(grid1_rank),
c$$$     &          grid2_dims(grid2_rank))

c$$$      ncstat = nf_get_att_text(nc_grid1_id, nf_global, 'title',
c$$$     &                         grid1_name)
c$$$
c$$$      ncstat = nf_get_att_text(nc_grid2_id, nf_global, 'title',
c$$$     &                         grid2_name)

!-----------------------------------------------------------------------
!
!     allocate grid coordinates/masks and read data
!
!-----------------------------------------------------------------------

c$$$      allocate( grid1_mask      (grid1_size),
c$$$     &          grid2_mask      (grid2_size),
c$$$     &          grid1_center_lat(grid1_size),
c$$$     &          grid1_center_lon(grid1_size),
c$$$     &          grid2_center_lat(grid2_size),
c$$$     &          grid2_center_lon(grid2_size),
c$$$     &          grid1_area      (grid1_size),
c$$$     &          grid2_area      (grid2_size),
c$$$     &          grid1_frac      (grid1_size),
c$$$     &          grid2_frac      (grid2_size),
c$$$     &          grid1_corner_lat(grid1_corners, grid1_size),
c$$$     &          grid1_corner_lon(grid1_corners, grid1_size),
c$$$     &          grid2_corner_lat(grid2_corners, grid2_size),
c$$$     &          grid2_corner_lon(grid2_corners, grid2_size),
c$$$     &          grid1_bound_box (4            , grid1_size),
c$$$     &          grid2_bound_box (4            , grid2_size))

      allocate(
     &          grid1_area      (grid1_size),
     &          grid2_area      (grid2_size),
     &          grid1_frac      (grid1_size),
     &          grid2_frac      (grid2_size),
     &          grid1_bound_box (4            , grid1_size),
     &          grid2_bound_box (4            , grid2_size))

c$$$      allocate(imask(grid1_size))

c$$$      TODO: grid1_dims,imask,grid1_center_lat,grid1_center_lon,grid1_corner_lat,grid1_corner_lon

c$$$      ncstat = nf_inq_varid(nc_grid1_id, 'grid_dims', nc_grid1dims_id)
c$$$      ncstat = nf_inq_varid(nc_grid1_id, 'grid_imask', nc_grd1imask_id)
c$$$      ncstat = nf_inq_varid(nc_grid1_id, 'grid_center_lat',
c$$$     &                                   nc_grd1cntrlat_id)
c$$$      ncstat = nf_inq_varid(nc_grid1_id, 'grid_center_lon',
c$$$     &                                   nc_grd1cntrlon_id)
c$$$      ncstat = nf_inq_varid(nc_grid1_id, 'grid_corner_lat',
c$$$     &                                   nc_grd1crnrlat_id)
c$$$      ncstat = nf_inq_varid(nc_grid1_id, 'grid_corner_lon',
c$$$     &                                   nc_grd1crnrlon_id)
c$$$
c$$$      ncstat = nf_get_var_int(nc_grid1_id, nc_grid1dims_id, grid1_dims)
c$$$
c$$$      ncstat = nf_get_var_int(nc_grid1_id, nc_grd1imask_id, imask)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid1_id, nc_grd1cntrlat_id,
c$$$     &                                       grid1_center_lat)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid1_id, nc_grd1cntrlon_id,
c$$$     &                                       grid1_center_lon)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid1_id, nc_grd1crnrlat_id,
c$$$     &                                       grid1_corner_lat)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid1_id, nc_grd1crnrlon_id,
c$$$     &                                       grid1_corner_lon)

c$$$      if (luse_grid1_area) then
c$$$        allocate (grid1_area_in(grid1_size))
c$$$      TODO: grid1_area_in

c$$$        ncstat = nf_inq_varid(nc_grid1_id, 'grid_area', nc_grid1area_id)
c$$$        ncstat = nf_get_var_double(nc_grid1_id, nc_grid1area_id,
c$$$     &                                          grid1_area_in)
c$$$      endif

      grid1_area = zero
      grid1_frac = zero

!-----------------------------------------------------------------------
!
!     initialize logical mask and convert lat/lon units if required
!
!-----------------------------------------------------------------------

c$$$      !TODO: replace this
c$$$      where (imask == 1)
c$$$        grid1_mask = .true.
c$$$      elsewhere
c$$$        grid1_mask = .false.
c$$$      endwhere
c$$$      deallocate(imask)

c$$$      grid1_units = ' '
c$$$      TODO: grid1_units (for centers)

c$$$      ncstat = nf_get_att_text(nc_grid1_id, nc_grd1cntrlat_id, 'units',
c$$$     &                         grid1_units)

c$$$      select case (grid1_units(1:7))
c$$$      case ('degrees')
c$$$
c$$$        grid1_center_lat = grid1_center_lat*deg2rad
c$$$        grid1_center_lon = grid1_center_lon*deg2rad
c$$$
c$$$      case ('radians')
c$$$
c$$$        !*** no conversion necessary
c$$$
c$$$      case default
c$$$
c$$$        print *,'unknown units supplied for grid1 center lat/lon: '
c$$$        print *,'proceeding assuming radians'
c$$$
c$$$      end select

c$$$      grid1_units = ' '
c$$$      TODO: grid1_units (for corners)

c$$$      ncstat = nf_get_att_text(nc_grid1_id, nc_grd1crnrlat_id, 'units',
c$$$     &                         grid1_units)

c$$$      select case (grid1_units(1:7))
c$$$      case ('degrees')
c$$$
c$$$        grid1_corner_lat = grid1_corner_lat*deg2rad
c$$$        grid1_corner_lon = grid1_corner_lon*deg2rad
c$$$
c$$$      case ('radians')
c$$$
c$$$        !*** no conversion necessary
c$$$
c$$$      case default
c$$$
c$$$        print *,'unknown units supplied for grid1 corner lat/lon: '
c$$$        print *,'proceeding assuming radians'
c$$$
c$$$      end select

c$$$      ncstat = nf_close(nc_grid1_id)

!-----------------------------------------------------------------------
!
!     read data for grid 2
!
!-----------------------------------------------------------------------

c$$$      allocate(imask(grid2_size))

c$$$      TODO: grid2_dims,imask,grid2_center_lat,grid2_center_lon,grid2_corner_lat,grid2_corner_lon (!!!imask is read again - used above - need 2 diff names!!!)

c$$$      ncstat = nf_inq_varid(nc_grid2_id, 'grid_dims', nc_grid2dims_id)
c$$$      ncstat = nf_inq_varid(nc_grid2_id, 'grid_imask', nc_grd2imask_id)
c$$$      ncstat = nf_inq_varid(nc_grid2_id, 'grid_center_lat',
c$$$     &                                   nc_grd2cntrlat_id)
c$$$      ncstat = nf_inq_varid(nc_grid2_id, 'grid_center_lon',
c$$$     &                                   nc_grd2cntrlon_id)
c$$$      ncstat = nf_inq_varid(nc_grid2_id, 'grid_corner_lat',
c$$$     &                                   nc_grd2crnrlat_id)
c$$$      ncstat = nf_inq_varid(nc_grid2_id, 'grid_corner_lon',
c$$$     &                                   nc_grd2crnrlon_id)
c$$$
c$$$      ncstat = nf_get_var_int(nc_grid2_id, nc_grid2dims_id, grid2_dims)
c$$$
c$$$      ncstat = nf_get_var_int(nc_grid2_id, nc_grd2imask_id, imask)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid2_id, nc_grd2cntrlat_id,
c$$$     &                                       grid2_center_lat)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid2_id, nc_grd2cntrlon_id,
c$$$     &                                       grid2_center_lon)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid2_id, nc_grd2crnrlat_id,
c$$$     &                                       grid2_corner_lat)
c$$$
c$$$      ncstat = nf_get_var_double(nc_grid2_id, nc_grd2crnrlon_id,
c$$$     &                                       grid2_corner_lon)

c$$$      if (luse_grid2_area) then
c$$$        allocate (grid2_area_in(grid2_size))
c$$$      TODO: grid2_area_in

c$$$        ncstat = nf_inq_varid(nc_grid2_id, 'grid_area', nc_grid2area_id)
c$$$        ncstat = nf_get_var_double(nc_grid2_id, nc_grid2area_id,
c$$$     &                                          grid2_area_in)
c$$$      endif

      grid2_area = zero
      grid2_frac = zero

!-----------------------------------------------------------------------
!
!     initialize logical mask and convert lat/lon units if required
!
!-----------------------------------------------------------------------

c$$$      where (imask == 1)
c$$$        grid2_mask = .true.
c$$$      elsewhere
c$$$        grid2_mask = .false.
c$$$      endwhere
c$$$      deallocate(imask)

c$$$      grid2_units = ' '
c$$$      TODO: grid2_units (for centers)

c$$$      ncstat = nf_get_att_text(nc_grid2_id, nc_grd2cntrlat_id, 'units',
c$$$     &                         grid2_units)

c$$$      select case (grid2_units(1:7))
c$$$      case ('degrees')
c$$$
c$$$        grid2_center_lat = grid2_center_lat*deg2rad
c$$$        grid2_center_lon = grid2_center_lon*deg2rad
c$$$
c$$$      case ('radians')
c$$$
c$$$        !*** no conversion necessary
c$$$
c$$$      case default
c$$$
c$$$        print *,'unknown units supplied for grid2 center lat/lon: '
c$$$        print *,'proceeding assuming radians'
c$$$
c$$$      end select

c$$$      grid2_units = ' '
c$$$      TODO: grid2_units (for corners)

c$$$      ncstat = nf_get_att_text(nc_grid2_id, nc_grd2crnrlat_id, 'units',
c$$$     &                         grid2_units)

c$$$      select case (grid2_units(1:7))
c$$$      case ('degrees')
c$$$
c$$$        grid2_corner_lat = grid2_corner_lat*deg2rad
c$$$        grid2_corner_lon = grid2_corner_lon*deg2rad
c$$$
c$$$      case ('radians')
c$$$
c$$$        !*** no conversion necessary
c$$$
c$$$      case default
c$$$
c$$$        print *,'no units supplied for grid2 corner lat/lon: '
c$$$        print *,'proceeding assuming radians'
c$$$
c$$$      end select

c$$$      ncstat = nf_close(nc_grid2_id)


!-----------------------------------------------------------------------
!
!     convert longitudes to 0,2pi interval
!
!-----------------------------------------------------------------------

      where (grid1_center_lon .gt. pi2)  grid1_center_lon =
     &                                   grid1_center_lon - pi2
      where (grid1_center_lon .lt. zero) grid1_center_lon =
     &                                   grid1_center_lon + pi2
      where (grid2_center_lon .gt. pi2)  grid2_center_lon =
     &                                   grid2_center_lon - pi2
      where (grid2_center_lon .lt. zero) grid2_center_lon =
     &                                   grid2_center_lon + pi2
      where (grid1_corner_lon .gt. pi2)  grid1_corner_lon =
     &                                   grid1_corner_lon - pi2
      where (grid1_corner_lon .lt. zero) grid1_corner_lon =
     &                                   grid1_corner_lon + pi2
      where (grid2_corner_lon .gt. pi2)  grid2_corner_lon =
     &                                   grid2_corner_lon - pi2
      where (grid2_corner_lon .lt. zero) grid2_corner_lon =
     &                                   grid2_corner_lon + pi2

!-----------------------------------------------------------------------
!
!     make sure input latitude range is within the machine values
!     for +/- pi/2
!
!-----------------------------------------------------------------------

      where (grid1_center_lat >  pih) grid1_center_lat =  pih
      where (grid1_corner_lat >  pih) grid1_corner_lat =  pih
      where (grid1_center_lat < -pih) grid1_center_lat = -pih
      where (grid1_corner_lat < -pih) grid1_corner_lat = -pih

      where (grid2_center_lat >  pih) grid2_center_lat =  pih
      where (grid2_corner_lat >  pih) grid2_corner_lat =  pih
      where (grid2_center_lat < -pih) grid2_center_lat = -pih
      where (grid2_corner_lat < -pih) grid2_corner_lat = -pih

!-----------------------------------------------------------------------
!
!     compute bounding boxes for restricting future grid searches
!
!-----------------------------------------------------------------------

      if (.not. luse_grid_centers) then
        grid1_bound_box(1,:) = minval(grid1_corner_lat, DIM=1)
        grid1_bound_box(2,:) = maxval(grid1_corner_lat, DIM=1)
        grid1_bound_box(3,:) = minval(grid1_corner_lon, DIM=1)
        grid1_bound_box(4,:) = maxval(grid1_corner_lon, DIM=1)

        grid2_bound_box(1,:) = minval(grid2_corner_lat, DIM=1)
        grid2_bound_box(2,:) = maxval(grid2_corner_lat, DIM=1)
        grid2_bound_box(3,:) = minval(grid2_corner_lon, DIM=1)
        grid2_bound_box(4,:) = maxval(grid2_corner_lon, DIM=1)

      else

        nx = grid1_dims(1)
        ny = grid1_dims(2)

        do n=1,grid1_size

          !*** find N,S and NE points to this grid point

          j = (n - 1)/nx +1
          i = n - (j-1)*nx

          if (i < nx) then
            ip1 = i + 1
          else
            !*** assume cyclic
            ip1 = 1
            !*** but if it is not, correct
            e_add = (j - 1)*nx + ip1
            if (abs(grid1_center_lat(e_add) -
     &              grid1_center_lat(n   )) > pih) then
              ip1 = i
            endif
          endif

          if (j < ny) then
            jp1 = j+1
          else
            !*** assume cyclic
            jp1 = 1
            !*** but if it is not, correct
            n_add = (jp1 - 1)*nx + i
            if (abs(grid1_center_lat(n_add) -
     &              grid1_center_lat(n   )) > pih) then
              jp1 = j
            endif
          endif

          n_add = (jp1 - 1)*nx + i
          e_add = (j - 1)*nx + ip1
          ne_add = (jp1 - 1)*nx + ip1

          !*** find N,S and NE lat/lon coords and check bounding box

          tmp_lats(1) = grid1_center_lat(n)
          tmp_lats(2) = grid1_center_lat(e_add)
          tmp_lats(3) = grid1_center_lat(ne_add)
          tmp_lats(4) = grid1_center_lat(n_add)

          tmp_lons(1) = grid1_center_lon(n)
          tmp_lons(2) = grid1_center_lon(e_add)
          tmp_lons(3) = grid1_center_lon(ne_add)
          tmp_lons(4) = grid1_center_lon(n_add)

          grid1_bound_box(1,n) = minval(tmp_lats)
          grid1_bound_box(2,n) = maxval(tmp_lats)
          grid1_bound_box(3,n) = minval(tmp_lons)
          grid1_bound_box(4,n) = maxval(tmp_lons)
        end do

        nx = grid2_dims(1)
        ny = grid2_dims(2)

        do n=1,grid2_size

          !*** find N,S and NE points to this grid point

          j = (n - 1)/nx +1
          i = n - (j-1)*nx

          if (i < nx) then
            ip1 = i + 1
          else
            !*** assume cyclic
            ip1 = 1
            !*** but if it is not, correct
            e_add = (j - 1)*nx + ip1
            if (abs(grid2_center_lat(e_add) -
     &              grid2_center_lat(n   )) > pih) then
              ip1 = i
            endif
          endif

          if (j < ny) then
            jp1 = j+1
          else
            !*** assume cyclic
            jp1 = 1
            !*** but if it is not, correct
            n_add = (jp1 - 1)*nx + i
            if (abs(grid2_center_lat(n_add) -
     &              grid2_center_lat(n   )) > pih) then
              jp1 = j
            endif
          endif

          n_add = (jp1 - 1)*nx + i
          e_add = (j - 1)*nx + ip1
          ne_add = (jp1 - 1)*nx + ip1

          !*** find N,S and NE lat/lon coords and check bounding box

          tmp_lats(1) = grid2_center_lat(n)
          tmp_lats(2) = grid2_center_lat(e_add)
          tmp_lats(3) = grid2_center_lat(ne_add)
          tmp_lats(4) = grid2_center_lat(n_add)

          tmp_lons(1) = grid2_center_lon(n)
          tmp_lons(2) = grid2_center_lon(e_add)
          tmp_lons(3) = grid2_center_lon(ne_add)
          tmp_lons(4) = grid2_center_lon(n_add)

          grid2_bound_box(1,n) = minval(tmp_lats)
          grid2_bound_box(2,n) = maxval(tmp_lats)
          grid2_bound_box(3,n) = minval(tmp_lons)
          grid2_bound_box(4,n) = maxval(tmp_lons)
        end do

      endif

      where (abs(grid1_bound_box(4,:) - grid1_bound_box(3,:)) > pi)
        grid1_bound_box(3,:) = zero
        grid1_bound_box(4,:) = pi2
      end where

      where (abs(grid2_bound_box(4,:) - grid2_bound_box(3,:)) > pi)
        grid2_bound_box(3,:) = zero
        grid2_bound_box(4,:) = pi2
      end where

      !***
      !*** try to check for cells that overlap poles
      !***

      where (grid1_center_lat > grid1_bound_box(2,:))
     &  grid1_bound_box(2,:) = pih

      where (grid1_center_lat < grid1_bound_box(1,:))
     &  grid1_bound_box(1,:) = -pih

      where (grid2_center_lat > grid2_bound_box(2,:))
     &  grid2_bound_box(2,:) = pih

      where (grid2_center_lat < grid2_bound_box(1,:))
     &  grid2_bound_box(1,:) = -pih

!-----------------------------------------------------------------------
!
!     set up and assign address ranges to search bins in order to
!     further restrict later searches
!
!-----------------------------------------------------------------------

      select case (restrict_type)

      case ('latitude')
c$$$        write(stdout,*) 'Using latitude bins to restrict search.'

        allocate(bin_addr1(2,num_srch_bins))
        allocate(bin_addr2(2,num_srch_bins))
        allocate(bin_lats (2,num_srch_bins))
        allocate(bin_lons (2,num_srch_bins))

        dlat = pi/num_srch_bins

        do n=1,num_srch_bins
          bin_lats(1,n) = (n-1)*dlat - pih
          bin_lats(2,n) =     n*dlat - pih
          bin_lons(1,n) = zero
          bin_lons(2,n) = pi2
          bin_addr1(1,n) = grid1_size + 1
          bin_addr1(2,n) = 0
          bin_addr2(1,n) = grid2_size + 1
          bin_addr2(2,n) = 0
        end do

        do nele=1,grid1_size
          do n=1,num_srch_bins
            if (grid1_bound_box(1,nele) <= bin_lats(2,n) .and.
     &          grid1_bound_box(2,nele) >= bin_lats(1,n)) then
              bin_addr1(1,n) = min(nele,bin_addr1(1,n))
              bin_addr1(2,n) = max(nele,bin_addr1(2,n))
            endif
          end do
        end do

        do nele=1,grid2_size
          do n=1,num_srch_bins
            if (grid2_bound_box(1,nele) <= bin_lats(2,n) .and.
     &          grid2_bound_box(2,nele) >= bin_lats(1,n)) then
              bin_addr2(1,n) = min(nele,bin_addr2(1,n))
              bin_addr2(2,n) = max(nele,bin_addr2(2,n))
            endif
          end do
        end do

      case default
c$$$      case ('latlon')
c$$$        write(stdout,*) 'Using lat/lon boxes to restrict search.'

        dlat = pi /num_srch_bins
        dlon = pi2/num_srch_bins

        allocate(bin_addr1(2,num_srch_bins*num_srch_bins))
        allocate(bin_addr2(2,num_srch_bins*num_srch_bins))
        allocate(bin_lats (2,num_srch_bins*num_srch_bins))
        allocate(bin_lons (2,num_srch_bins*num_srch_bins))

        n = 0
        do j=1,num_srch_bins
        do i=1,num_srch_bins
          n = n + 1

          bin_lats(1,n) = (j-1)*dlat - pih
          bin_lats(2,n) =     j*dlat - pih
          bin_lons(1,n) = (i-1)*dlon
          bin_lons(2,n) =     i*dlon
          bin_addr1(1,n) = grid1_size + 1
          bin_addr1(2,n) = 0
          bin_addr2(1,n) = grid2_size + 1
          bin_addr2(2,n) = 0
        end do
        end do

        num_srch_bins = num_srch_bins**2

        do nele=1,grid1_size
          do n=1,num_srch_bins
            if (grid1_bound_box(1,nele) <= bin_lats(2,n) .and.
     &          grid1_bound_box(2,nele) >= bin_lats(1,n) .and.
     &          grid1_bound_box(3,nele) <= bin_lons(2,n) .and.
     &          grid1_bound_box(4,nele) >= bin_lons(1,n)) then
              bin_addr1(1,n) = min(nele,bin_addr1(1,n))
              bin_addr1(2,n) = max(nele,bin_addr1(2,n))
            endif
          end do
        end do

        do nele=1,grid2_size
          do n=1,num_srch_bins
            if (grid2_bound_box(1,nele) <= bin_lats(2,n) .and.
     &          grid2_bound_box(2,nele) >= bin_lats(1,n) .and.
     &          grid2_bound_box(3,nele) <= bin_lons(2,n) .and.
     &          grid2_bound_box(4,nele) >= bin_lons(1,n)) then
              bin_addr2(1,n) = min(nele,bin_addr2(1,n))
              bin_addr2(2,n) = max(nele,bin_addr2(2,n))
            endif
          end do
        end do

c$$$      case default
c$$$        stop 'unknown search restriction method'
      end select

!-----------------------------------------------------------------------

      end subroutine grid_init

!***********************************************************************

      end module grids

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!/**
      subroutine scrip_grid_init(
     $     F_grid1_size,F_grid1_dims,F_grid1_corners,
     $     F_grid1_center_lat,F_grid1_center_lon,
     $     F_grid1_corner_lat,F_grid1_corner_lon,
     $     F_grid2_size,F_grid2_dims,F_grid2_corners,
     $     F_grid2_center_lat,F_grid2_center_lon,
     $     F_grid2_corner_lat,F_grid2_corner_lon)
      use kinds_mod    ! defines data types
      use grids
      implicit none

      integer (kind=int_kind), dimension(2) ::
     &             F_grid1_dims, F_grid2_dims  ! size of each grid dimension

      integer (kind=int_kind) ::
     &             F_grid1_size,F_grid2_size,
     &             F_grid1_corners, F_grid2_corners ! number of corners
                                                    ! for each grid cell
c$$$      logical (kind=log_kind), dimension(:), target ::
c$$$     &             F_grid1_mask,        ! flag which cells participate
c$$$     &             F_grid2_mask         ! flag which cells participate

      real (kind=dbl_kind), dimension(F_grid1_size), target ::
     &             F_grid1_center_lat,  ! lat/lon coordinates for
     &             F_grid1_center_lon   ! each grid center in radians

      real (kind=dbl_kind), dimension(F_grid2_size), target ::
     &             F_grid2_center_lat,
     &             F_grid2_center_lon

c$$$      real (kind=dbl_kind), dimension(:), target ::
c$$$     &             F_grid1_area_in,     ! area of grid1 cell from file
c$$$     &             F_grid2_area_in      ! area of grid2 cell from file

      real (kind=dbl_kind), 
     &     dimension(F_grid1_corners,F_grid1_size), target  ::
     &             F_grid1_corner_lat,  ! lat/lon coordinates for
     &             F_grid1_corner_lon   ! each grid corner in radians

      real (kind=dbl_kind), 
     &     dimension(F_grid2_corners,F_grid2_size), target  ::
     &             F_grid2_corner_lat,
     &             F_grid2_corner_lon

!**/
      !--------------------------------------------------
      !These are init before in scrip.f
c$$$      restrict_type     = 'latlon' ! type of bins to use: latitude,latlon
c$$$      num_srch_bins     = 900      ! num of bins for restricted srch
c$$$      luse_grid_centers = (F_luse_grid_centers>0)

      grid1_rank = 1
      if (F_grid1_dims(2) > 0) grid1_rank = 2
      allocate(grid1_dims(grid1_rank))
      grid1_dims(1)    = F_grid1_dims(1)
      if (grid1_rank>1) grid1_dims(2)    = F_grid1_dims(2)
      grid1_corners = F_grid1_corners
      luse_grid1_area = .false.
c$$$      if (present(F_grid1_area_in)) then
c$$$         luse_grid1_area = .true.
c$$$         grid1_area_in => F_grid1_area_in
c$$$      endif
c$$$      if (present(F_grid1_mask)) then
c$$$         grid1_mask => F_grid1_mask
c$$$      else
         allocate(grid1_mask(grid1_size))
         grid1_mask = .true.
c$$$      endif
      grid1_center_lat => F_grid1_center_lat
      grid1_center_lon => F_grid1_center_lon
      grid1_corner_lat => F_grid1_corner_lat
      grid1_corner_lon => F_grid1_corner_lon


      grid2_rank = 1
      if (F_grid2_dims(2) > 0) grid2_rank = 2
      allocate(grid2_dims(grid2_rank))
      grid2_dims(1)    = F_grid2_dims(1)
      if (grid2_rank>1) grid2_dims(2)    = F_grid2_dims(2)
      grid2_corners = F_grid2_corners
      luse_grid1_area = .false.
c$$$      if (present(F_grid2_area_in)) then
c$$$         luse_grid2_area = .true.
c$$$         grid2_area_in => F_grid2_area_in
c$$$      endif
c$$$      if (present(F_grid2_mask)) then
c$$$         grid2_mask => F_grid2_mask
c$$$      else
         allocate(grid2_mask(grid2_size))
         grid2_mask = .true.
c$$$      endif
      grid2_center_lat => F_grid2_center_lat
      grid2_center_lon => F_grid2_center_lon
      grid2_corner_lat => F_grid2_corner_lat
      grid2_corner_lon => F_grid2_corner_lon

      call grid_init()
      !--------------------------------------------------
      return
      end subroutine scrip_grid_init

!/**
      subroutine scrip_grid_finalize()
      use kinds_mod    ! defines data types
      use grids
      implicit none
!**/
      !--------------------------------------------------
      deallocate(
     $     grid1_dims,grid1_mask,
     $     grid1_area,grid1_frac,grid1_bound_box,
     $     grid2_dims,grid2_mask,
     $     grid2_area,grid2_frac,grid2_bound_box,
     $     bin_addr1,bin_addr2,bin_lats,bin_lons
     $     )
      !--------------------------------------------------
      return
      end subroutine scrip_grid_finalize
