!COMP_ARCH=intel-2016.1.156 ; -add=-C -g -traceback -ftrapuv
!---------------------------------- LICENCE BEGIN -------------------------------
! SCM - Library of kernel routines for the RPN single column model
! Copyright (C) 1990-2017 - Division de Recherche en Prevision Numerique
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

subroutine fst2ptxt()
  ! Generate a profile model input text file from an input FST file.
  use prof_mod, only: prof_set_geom
  use vgrid_descriptors, only: vgrid_descriptor,vgd_new,vgd_get,vgd_levels,VGD_OK
  use hgrid_wb, only: hgrid_wb_get
  use vertical_interpolation, only: vertint2

  implicit none  

  ! Local parameters
  integer, parameter :: LONG_CHAR=1024
  integer, parameter :: MAX_TRACERS=1000
  integer, parameter :: MAX_LEVELS=10000
  integer, parameter :: NI_ZOOM=3,NJ_ZOOM=3
  real, parameter :: KTTOMS=0.514444
  character(len=2), parameter :: TESTFLD='UU'

  ! Local derived types
  type field
     integer :: ni=0,nj=0,nk=0,gid=-1
     integer, dimension(4) :: ig=(/-1,-1,-1,-1/)
     character(len=LONG_CHAR) :: in='',out='',short='',lev='N',grtyp=''
     real, pointer, dimension(:,:) :: v2d=>null()
     real, pointer, dimension(:,:,:) :: v3d=>null()
     real :: mult=1.,add=0.
     logical :: found=.false.,required=.true.
  end type field

  ! Local variables
  integer :: i,j,k,fdin,fdout,err,nrec,nkeys,dateo,deet,npas,nbits,datyp, &
       ip1,ip2,ip3,ip4,ni,nj,nk,ig1,ig2,ig3,ig4,swa,lng,dltf,ubc, &
       ex1,ex2,ex3,nnames,ntr,datev,nfld,itt,iuu,ivv,iww,ihu,itr,ip0,igz, &
       gid_prof,gid_zoom,i_zoom,j_zoom,key,ndates,t,iuu_dyn,ivv_dyn,itt_dyn, &
       ihu_dyn,itst,base_datev,pex,pey,pe_local,pe_total
  integer, dimension(:), allocatable :: keyList,keysm,keyst,im,it,dateList
  integer, dimension(:), pointer :: ip1t,ip1m
  real :: tcdk,omega,pi,grav,di,dj,d_zoom,mult
  real, dimension(1) :: pt,i_map,j_map,ax,ay,lon,lat
  real, dimension(:), allocatable :: geo,advm,advt
  real, dimension(:), pointer :: profpm_col=>null(),profpt_col=>null()
  real, dimension(1,1,1) :: p0,me,test_point
  real, dimension(NI_ZOOM,NJ_ZOOM) :: lat_zoom,lon_zoom,fcor,fcor_clipped
  real, dimension(:,:), allocatable :: lat_map,lon_map,psfc,sfc_gz,test_grid
  real, dimension(:,:,:), allocatable :: profpm,profpt,uuz,vvz,ttz,&
       gzz,uut_p,vvt_p,ttt_p,gzm_p,uum_p,vvm_p,pmz,ptz,profpm_p,profpt_p
  real, dimension(:,:,:,:), allocatable :: trz,trt_p
  real, dimension(:,:,:), pointer :: presm=>null(),prest=>null()
  real(kind=8), dimension(2) :: ppoint
  character(len=1) :: typvar,grtyp,delim
  character(len=4) :: nomvar
  character(len=12) :: etiket
  character(len=15) :: date
  character(len=LONG_CHAR) :: infile,mode,outfile,interp,maxfld,cmax_levels, &
       datafmt,levfmt,vinterp,header_entries,trstring,fldname,fldlist
  character(len=LONG_CHAR), dimension(MAX_TRACERS) :: trlist=''
  character(len=LONG_CHAR), dimension(:), allocatable :: args
  logical :: init,found
  type(field), dimension(MAX_TRACERS+10) :: fld
  type(vgrid_descriptor) :: vcoord

  ! External functions
  integer, external :: iargc,fnom,fclos,fstouv,fstfrm,fstprm,fstinl,fstluk,fstinf, &
       ezqkdef,ezsetopt,ezdefset,ezsint,ezuvint,gdll,gdxyfll,ezgdef_fmem,ezsetval, &
       ip1_all,utils_topology

  ! Initialize RPN COMM libraries
  pex = 0; pey = 0
  call RPN_COMM_init(utils_topology,pe_local,pe_total,pex,pey)

  ! Local settings
  vinterp='cubic'

  ! Get positional command line arguments
  i = iargc()
  if (i < 5) call handle_error(-1,'fst2ptxt','USAGE: fst2ptxt FILE [init/forcings] LAT LON INTERP TRACER_LIST')
  allocate(args(i),stat=err)
  call handle_error(err,'fst2ptxt','Allocating space for args')
  do i=1,size(args)
     call getarg(i,args(i))
  enddo
  infile = args(1)
  mode = args(2)
  read(args(3),*) ppoint(1)
  read(args(4),*) ppoint(2)
  interp = args(5)
  trstring = args(6)
  ntr = 0; delim = ','
  do while (index(trstring,delim) > 0 .and. ntr < MAX_TRACERS)
     ntr = ntr+1
     trlist(ntr) = trstring(1:index(trstring,delim)-1)
     trstring = trstring(index(trstring,delim)+1:)
  enddo
  if (len_trim(trstring) > 0) then
     ntr = ntr+1
     trlist(ntr) = trstring
  endif
  deallocate(args,stat=err)
  call handle_error(err,'fst2ptxt','Freeing args')

  ! Get external constants
  call constnt(tcdk,err,'TCDK',0)
  call handle_error(err,'fst2ptxt','Retrieving constant TCDK')
  call constnt(omega,err,'OMEGA',0)
  call handle_error(err,'fst2ptxt','Retrieving constant OMEGA')
  call constnt(pi,err,'PI',0)
  call handle_error(err,'fst2ptxt','Retrieving constant PI')
  call constnt(grav,err,'GRAV',0)
  call handle_error(err,'fst2ptxt','Retrieving constant GRAV')

  ! Open input file
  fdin = 0
  err = fnom(fdin,trim(infile),'STD+R/O',0)
  call handle_error(err,'fst2ptxt','Acquiring lock for '//trim(infile))
  nrec = fstouv(fdin,'RND')
  call handle_error_l(nrec>=0,'fst2ptxt','Opening '//trim(infile))

  ! Check whether to generate an initial state or forcing file
  select case (trim(mode))
  case ('init')
     init = .true.
  case ('forcings')
     init = .false.
  case DEFAULT
     call handle_error(-1,'fst2ptxt','Unknown mode '//trim(mode)//' specified on the command line')
  end select

  ! Set up specifics of fields to use for calculations
  i = 1
  fld(i)%in='P0'; ip0 = i
  fld(i)%out='PRE_P0'
  fld(i)%lev='S'
  fld(i)%mult=100.
  i = i+1
  fld(i)%in='TT'; itt = i
  fld(i)%out='PW_TT:M'
  fld(i)%lev='T'
  fld(i)%add=tcdk
  i = i+1
  fld(i)%in='UU'; iuu = i
  fld(i)%out='PW_UU:M'
  fld(i)%lev='M'
  fld(i)%mult=KTTOMS
  i = i+1
  fld(i)%in='VV'; ivv = i
  fld(i)%out='PW_VV:M'
  fld(i)%lev='M'
  fld(i)%mult=KTTOMS
  i = i+1
  fld(i)%in='WT1|WW'; iww = i
  fld(i)%out='PRE_'
  fld(i)%lev='T'
  i = i+1
  fld(i)%in='GZ'; igz = i
  fld(i)%lev='M'
  fld(i)%mult=10.
  i = i+1
  fld(i)%in='TTND|TADV'; itt_dyn = i
  fld(i)%out='DYN_TT'
  fld(i)%lev='T'
  fld(i)%required=.false.
  i = i+1
  fld(i)%in='QADV'; ihu_dyn = i
  fld(i)%out='DYN_HU'
  fld(i)%lev='T'
  fld(i)%required=.false.
  i = i+1
  fld(i)%in='UTND|UADV'; iuu_dyn = i
  fld(i)%out='DYN_UU'
  fld(i)%lev='M'
  fld(i)%required=.false.
  i = i+1
  fld(i)%in='VTND|VADV'; ivv_dyn = i
  fld(i)%out='DYN_VV'
  fld(i)%lev='M'
  fld(i)%required=.false.
  do k=1,ntr
     i = i+1;
     if (k == 1) itr = i
     fld(i)%in=trim(trlist(k)); if (trim(fld(i)%in) == 'HU') ihu = i
     trstring = trlist(k)
     ! Remove T1 extension if present
     if (index(trstring,'T1') > 0) then
        if (len_trim(trstring)-index(trstring,'T1') == 1) trstring = trstring(:index(trstring,'T1')-1)
     endif
     fld(i)%out='TR^'//trim(trstring)//':M'
     fld(i)%short=trstring
     fld(i)%lev='T'
  enddo
  nfld = i

  ! Obtain input field dimensions for each record and valid dates for the file
  allocate(dateList(nrec),keyList(nrec),stat=err)
  call handle_error(err,'fst2ptxt','Allocating dateList/keyList')
  ndates = 0
  do k=1,nfld
     fldname = trim(fld(k)%in)//'|'; fldlist = fldname
     nkeys = 0
     do while (len_trim(fldname) > 0 .and. nkeys <= 0 .and. index(fldlist,'|') > 0)
        fldname = fldlist(1:index(fldlist,'|')-1)
        fldlist = fldlist(index(fldlist,'|')+1:)
        err = fstinl(fdin,ni,nj,nk,-1,'',-1,-1,-1,'',fldname,keyList,nkeys,size(keyList))
     enddo
     call handle_error(err,'fst2ptxt','Getting key list for '//trim(fld(k)%in))
     fld(k)%in = fldname
     if (fld(k)%required .and. nkeys <= 0) call handle_error(-1,'fst2ptxt','Cannot find required field '//trim(fld(k)%in))
     if (nkeys > 0) fld(k)%found = .true.
     if (k == iww) fld(k)%out=trim(fld(k)%out)//fldname
     do i=1,nkeys
        err = fstprm(keyList(i),dateo,deet,npas,ni,nj,nk,nbits,datyp, &
             ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4, &
             swa,lng,dltf,ubc,ex1,ex2,ex3)
        call incdatr(datev,dateo,dble(deet*npas)/3600.d0)
        if (all(fld(k)%ig < 0)) then
           fld(k)%ig = (/ig1,ig2,ig3,ig4/)
           fld(k)%ni = ni; fld(k)%nj = nj
           fld(k)%grtyp = grtyp
           base_datev = datev
        else
           call handle_error_l((/ig1,ig2,ig3,ig4/)==fld(k)%ig,'fst2ptxt','More than one grid found for '//trim(fld(k)%in))
        endif
        if (datev == base_datev) fld(k)%nk = fld(k)%nk+1
        if (trim(nomvar) == trim(TESTFLD)) then
           itst = k
           found = .false.; j = 1
           do while (j <= ndates)
              if (datev == dateList(j)) found = .true.
              j = j+1
           enddo
           if (.not.found) then 
              ndates = ndates+1
              dateList(ndates) = datev
           endif
        endif
     enddo
  enddo
  datev = dateList(1)

  ! Get vertical coordinate information
  err = vgd_new(vcoord,fdin)
  call handle_error_l(err==VGD_OK,'fst2ptxt','Cannot build struct vcoord')
  i = 1
  do while (trim(fld(i)%lev) /= 'M' .and. i < nfld)
     i = i+1
  enddo
  call handle_error_l(trim(fld(i)%lev)=='M','fst2ptxt','No momentum-level fields found')
  err = fstinl(fdin,ni,nj,nk,datev,'',-1,-1,-1,'',trim(fld(i)%in),keyList,nkeys,size(keyList))
  call handle_error(err,'fst2ptxt','Retrieving momentum keys using '//trim(fld(i)%in))
  allocate(keysm(nkeys),stat=err)
  call handle_error(err,'fst2ptxt','Allocating keysm')
  call handle_error_l(nkeys==fld(i)%nk,'fst2ptxt','Impossible mismatch for momentum levels')
  keysm = keyList(1:nkeys)
  allocate(ip1m(nkeys),stat=err)
  call handle_error(err,'fst2ptxt','Allocating ip1m')
  do i=1,nkeys
     err = fstprm(keyList(i),dateo,deet,npas,ni,nj,nk,nbits,datyp, &
          ip1m(i),ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4, &
          swa,lng,dltf,ubc,ex1,ex2,ex3)
  enddo
  i = 1
  do while (trim(fld(i)%lev) /= 'T' .and. i < nfld)
     i = i+1
  enddo
  call handle_error_l(trim(fld(i)%lev)=='T','fst2ptxt','No thermo-level fields found')
  err = fstinl(fdin,ni,nj,nk,datev,'',-1,-1,-1,'',trim(fld(i)%in),keyList,nkeys,size(keyList))
  call handle_error(err,'fst2ptxt','Retrieving thermo keys using '//trim(fld(i)%in))
  allocate(keyst(nkeys),stat=err)
  call handle_error(err,'fst2ptxt','Allocating keyst')
  call handle_error_l(nkeys==fld(i)%nk .or. nkeys==fld(i)%nk+1,'fst2ptxt','Impossible mismatch for thermo levels')
  keyst = keyList(1:nkeys)
  allocate(ip1t(nkeys),stat=err)
  call handle_error(err,'fst2ptxt','Allocating ip1t')
  do i=1,nkeys
     err = fstprm(keyList(i),dateo,deet,npas,ni,nj,nk,nbits,datyp, &
          ip1t(i),ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4, &
          swa,lng,dltf,ubc,ex1,ex2,ex3)
  enddo

  ! Allocate space in 'fld' struct for data
  do i=1,nfld
     select case (trim(fld(i)%lev))
     case ('S')
        allocate(fld(i)%v2d(fld(i)%ni,fld(i)%nj),stat=err)
        call handle_error(err,'fst2ptxt','Allocating '//trim(fld(i)%in))
     case ('M')
        allocate(fld(i)%v3d(fld(i)%ni,fld(i)%nj,size(ip1m)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating momentum-level '//trim(fld(i)%in))
     case ('T')
        allocate(fld(i)%v3d(fld(i)%ni,fld(i)%nj,size(ip1t)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating thermodynamic-level '//trim(fld(i)%in))
     case DEFAULT
        call handle_error(-1,'fst2ptxt','Unknown level indicator '//trim(fld(i)%lev)//' encountered')
     end select
  enddo

  ! Set up grids for horizontal interpolation
  fld(:)%gid = -1
  do i=1,nfld
     if (fld(i)%found) then
        fld(i)%gid = ezqkdef(fld(i)%ni,fld(i)%nj,fld(i)%grtyp,fld(i)%ig(1),fld(i)%ig(2),fld(i)%ig(3),fld(i)%ig(4),fdin)
     endif
  enddo
  if (ntr > 0) then
     call handle_error_l(all(fld(itr:itr+ntr-1)%gid==fld(itr)%gid),'fst2ptxt','Tracers must all share the same input grid')
  endif
  call prof_set_geom(ppoint)
  call handle_error(hgrid_wb_get('grid/scm',gid_prof),'fst2ptxt','Cannot define horizontal grid')
  err = gdll(gid_prof,lat,lon)

  ! Check that all source grids contain the destination grid point
  err = ezsetopt('EXTRAP_DEGREE','VALUE')
  call handle_error(err,'fst2ptxt','EZSETOPT extrap degree request VALUE')
  err = ezsetval('EXTRAP_VALUE',-9999.)
  call handle_error(err,'fst2ptxt','EZSETVAL value set to -9999')
  do i=1,nfld
     if (.not.fld(i)%found) cycle
     err = ezdefset(gid_prof,fld(i)%gid)
     call handle_error(err,'fst2ptxt','EZDEFSET for map->profile test interpolation for '//trim(fld(i)%in))
     allocate(test_grid(fld(i)%ni,fld(i)%nj),stat=err)
     call handle_error(err,'fst2ptxt','Allocating test_grid')
     test_grid = 1.
     err = ezsint(test_point,test_grid)
     call handle_error(err,'fst2ptxt','EZSINT interpolation for '//trim(fld(i)%in)//' test grid')
     call handle_error_l(test_point>0.,'fst2ptxt','Requested lat/lon is outside input grid for '//trim(fld(i)%in)// &
          ' in '//trim(infile))
     deallocate(test_grid,stat=err)
     call handle_error(err,'fst2ptxt','Freeing test_grid')
  enddo
  err = ezsetopt('EXTRAP_DEGREE','ABORT')
  call handle_error(err,'fst2ptxt','EZSETOPT extrap degree request ABORT')

  ! Compute surface elevation as the lowest-level geopotential height
  if (.not.allocated(sfc_gz)) allocate(sfc_gz(fld(igz)%ni,fld(igz)%nj),stat=err)
  call handle_error(err,'fst2ptxt','Allocating sfc_gz')
  mult = 1.
  key = fstinf(fdin,ni,nj,nk,datev,'',-1,-1,-1,'','ME')
  if (key < 0) then
     mult = 10.
     key = fstinf(fdin,ni,nj,nk,datev,'',ip1_all(1.,5),-1,-1,'','GZ')
  endif
  call handle_error(key,'fst2ptxt','Cannot find surface elevation information (ME or GZ)')
  err = fstluk(sfc_gz,key,ni,nj,nk)
  call handle_error(err,'fst2ptxt','Reading 2D record for elevation information')
  err = ezdefset(gid_prof,fld(igz)%gid)
  call handle_error(err,'fst2ptxt','EZDEFSET for map->profile interpolation for elevation')
  err = ezsetopt('INTERP_DEGREE',trim(interp))
  call handle_error(err,'fst2ptxt','EZSETOPT interp degree request '//trim(interp))
  err = ezsint(me,sfc_gz)
  call handle_error(err,'fst2ptxt','EZSINT interpolation for ME')
  me = me*mult
  deallocate(sfc_gz,stat=err)
  call handle_error(err,'fst2ptxt','Freeing sfc_gz')

  ! Process all valid dates found in the file
  dates: do t=1,ndates

     ! Read in pressure field to define vertical coordinate levels
     if (.not.allocated(psfc)) allocate(psfc(fld(ip0)%ni,fld(ip0)%nj),stat=err)
     call handle_error(err,'fst2ptxt','Allocating psfc')
     key = fstinf(fdin,ni,nj,nk,dateList(t),'',-1,-1,-1,'','P0')
     call handle_error(err,'fst2ptxt','Reading 2D record for surface pressure (P0)')
     if (key >= 0) then
        err = fstluk(psfc,key,ni,nj,nk)
        call handle_error(err,'fst2ptxt','Reading 2D record for surface pressure (P0)')
     else
        call handle_error(err,'fst2ptxt','Cannot find required P0')
     endif
     err = ezdefset(gid_prof,fld(ip0)%gid)
     call handle_error(err,'fst2ptxt','EZDEFSET for map->profile interpolation for P0')
     err = ezsetopt('INTERP_DEGREE',trim(interp))
     call handle_error(err,'fst2ptxt','EZSETOPT interp degree request '//trim(interp))
     psfc = psfc*100.
     err = ezsint(p0(:,:,1),psfc)
     call handle_error(err,'fst2ptxt','EZSINT interpolation error for surface pressure')

     ! Get pressure value for each set of coordinates
     err = vgd_levels(vcoord,ip1m,profpm_col,sfc_field=p0(1,1,1),in_log=.false.)
     call handle_error_l(err==VGD_OK,'fst2ptxt','Cannot retrieve momentum level profile')
     err = vgd_levels(vcoord,ip1t,profpt_col,sfc_field=p0(1,1,1),in_log=.false.)
     call handle_error_l(err==VGD_OK,'fst2ptxt','Cannot retrieve thermo level profile')
     err = vgd_levels(vcoord,ip1m,presm,sfc_field=psfc,in_log=.false.)
     call handle_error_l(err==VGD_OK,'fst2ptxt','Cannot retrieve momentum levels')
     err = vgd_levels(vcoord,ip1t,prest,sfc_field=psfc,in_log=.false.)
     call handle_error_l(err==VGD_OK,'fst2ptxt','Cannot retrieve thermo levels')
     allocate(profpm(1,1,size(profpm_col)),profpt(1,1,size(profpt_col)),stat=err)
     call handle_error(err,'fst2ptxt','Allocating profpm/profpt')
     profpm(1,1,:) = profpm_col
     profpt(1,1,:) = profpt_col
     deallocate(profpm_col,profpt_col,stat=err)
     call handle_error(err,'fst2ptxt','Freeing profpm_col/profpt_col')
     nullify(profpm_col,profpt_col)

     ! Sort levels so that the hightest altitudes are first (increasing pressure)
     if (.not.allocated(im)) then
        allocate(im(size(ip1m)),it(size(ip1t)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating im/it')
     endif
     err = sort(profpm(1,1,:),im)
     profpm(1,1,:) = profpm(1,1,im)
     call handle_error(err,'fst2ptxt','Sorting momentum levels')
     err = sort(profpt(1,1,:),it)
     profpt(1,1,:) = profpt(1,1,it)
     call handle_error(err,'fst2ptxt','Sorting thermodynamic levels')

     ! Read input fields
     fld(:)%found = .false.
     do i=1,nfld
        if (associated(fld(i)%v2d)) then
           key = fstinf(fdin,ni,nj,nk,dateList(t),'',-1,-1,-1,'',trim(fld(i)%in))
           if (key >= 0) then
              err = fstluk(fld(i)%v2d,key,ni,nj,nk)
              call handle_error(err,'fst2ptxt','Reading 2D record for '//trim(fld(i)%in))
              fld(i)%found = .true.
           else
              if (fld(i)%required) call handle_error(err,'fst2ptxt','Cannot find required (2D) '// &
                   trim(fld(i)%in)//' in '//trim(infile))
           endif
           if (fld(i)%found) fld(i)%v2d = fld(i)%v2d*fld(i)%mult + fld(i)%add
        else
           do k=1,size(fld(i)%v3d,dim=3)
              select case (trim(fld(i)%lev))
              case ('M')
                 ip1 = ip1m(im(k))
              case ('T')
                 ip1 = ip1t(it(k))
              case DEFAULT
                 call handle_error(-1,'fst2ptxt','Invalid level type '//trim(fld(i)%lev))
              end select
              key = fstinf(fdin,ni,nj,nk,dateList(t),'',ip1,-1,-1,'',trim(fld(i)%in))
              if (key >= 0) then
                 err = fstluk(fld(i)%v3d(:,:,k),key,ni,nj,nk)
                 call handle_error(err,'fst2ptxt','Reading 3D record for '//trim(fld(i)%in))
                 fld(i)%found = .true.
              else
                 if (fld(i)%required) call handle_error(err,'fst2ptxt','Cannot find required (3D) '// &
                      trim(fld(i)%in)//' in '//trim(infile))
                 fld(i)%v3d(:,:,k) = 0.
              endif
           enddo
           if (fld(i)%found) fld(i)%v3d = fld(i)%v3d*fld(i)%mult + fld(i)%add
        endif
     enddo

     ! Set up correct timestamp
     call datf2p(date,dateList(t))

     ! Open output file
     fdout = 0; outfile = trim(mode)//'_'//trim(date)//'.ptxt'
     err = fnom(fdout,trim(outfile),'SEQ',0)
     call handle_error(err,'fst2ptxt','Acquiring lock for '//trim(outfile))
     open(unit=fdout,file=trim(outfile))

     ! Set write formatting
     levfmt = 'i10'
     write(cmax_levels,'('//trim(levfmt)//')') 2*MAX_LEVELS
     datafmt = '(a,x,'//trim(levfmt)//',2(x,a),'//trim(cmax_levels)//'(x,es14.6))'
     
     ! Write generic header
     !   type, version, header_length
     !   mode, kind, lat, lon, elev, hinterp, date
     write(fdout,'(a,x,i1,x,i1)') 'PTXT',1,4
     write(fdout,'(a,x,i1,3(x,f14.8),2(x,a))') 'linear',5,lat,lon,me,trim(interp),trim(date)

     ! Write initializing data or compute/write derived fields
     initialization: if (init) then

        ! Write type-specific header
        !   n_sfc, sfc_varnames
        !   n_vars, atm_varnames
        write(maxfld,'(i10)') 4+ntr
        write(fdout,'(i1,x,a)') 1,trim(fld(ip0)%out)
        write(fdout,'(a,x,'//trim(maxfld)//'(x,a))') trim(maxfld),trim(fld(iuu)%out),trim(fld(ivv)%out), &
             trim(fld(itt)%out),trim(fld(iww)%out),(trim(fld(i)%out),i=itr,itr+ntr-1)

        ! Write individual fields
        !   varname,nlev,stag,interp,levels,values (sfc,atm)

        ! Set up horizontal interpolation rule
        err = ezsetopt('INTERP_DEGREE',trim(interp))
        call handle_error(err,'fst2ptxt','EZSETOPT interp degree request '//trim(interp))

        ! Write prescribed (pre-interpolated) P0
        write(fdout,datafmt) trim(fld(ip0)%out),1,trim(fld(ip0)%lev),'none',p0,p0

        ! Vector interpolate and write initial UU/VV (rotated to geographic west/south winds respectively)
        call interpv_write(fdout,fld(iuu),fld(ivv),profpm,vinterp,datafmt)

        ! Interpolate and write initial TT
        call interp_write(fdout,fld(itt),profpt,vinterp,datafmt)

        ! Interpolate and write prescribed vertical motion
        call interp_write(fdout,fld(iww),profpt,vinterp,datafmt)

        ! Interpolate and write initial tracers
        do j=itr,itr+ntr-1
           call interp_write(fdout,fld(j),profpt,vinterp,datafmt)
        enddo

     else

        ! Interpolate to a lat/lon grid centered on the profile point (zoom) in order
        ! to avoid pole and wrapping problems on global grids - instead, the forcing
        ! calculations are always made with finite differences on the limited-area grid
        call handle_error_l(mod(NI_ZOOM,2)>0 .and. mod(NJ_ZOOM,2)>0,'fst2ptxt','Zoom dimensions must be odd')
        err = ezsetopt('INTERP_DEGREE','CUBIQUE')
        call handle_error(err,'fst2ptxt','EZSETOPT interp degree request CUBIQUE')
        allocate(lat_map(fld(itst)%ni,fld(itst)%nj),lon_map(fld(itst)%ni,fld(itst)%nj),stat=err)
        call handle_error(err,'fst2ptxt','Allocating lat_map/lon_map')
        err = gdll(fld(itst)%gid,lat_map,lon_map)
        call handle_error(err,'fst2ptxt','Obtaining lat/lon from source grid')
        err = gdxyfll(fld(itst)%gid,i_map,j_map,lat,lon,1)
        call handle_error(err,'fst2ptxt','Determining i,j of profile point')
        i_map = max(min(i_map,real(fld(itst)%ni-1)),2.)
        j_map = max(min(j_map,real(fld(itst)%nj-1)),2.)
        di = path_length(lat_map(nint(i_map(1))+1,nint(j_map(1))),lon_map(nint(i_map(1))+1,nint(j_map(1))), &
             lat_map(nint(i_map(1))-1,nint(j_map(1))),lon_map(nint(i_map(1))-1,nint(j_map(1))))
        dj = path_length(lat_map(nint(i_map(1)),nint(j_map(1))+1),lon_map(nint(i_map(1)),nint(j_map(1))+1), &
             lat_map(nint(i_map(1)),nint(j_map(1))-1),lon_map(nint(i_map(1)),nint(j_map(1))-1))
        d_zoom = max(nint(100.*min(di,dj)/2.)/100.,0.01) !double-up spacing for possible rotation
        deallocate(lat_map,lon_map,stat=err)
        call handle_error(err,'fst2ptxt','Freeing lat_map/lon_map')
        call cxgaig('L',ig1,ig2,ig3,ig4,lat(1)-d_zoom*(NI_ZOOM/2),lon(1)-d_zoom*(NJ_ZOOM/2),d_zoom,d_zoom)
        gid_zoom = ezgdef_fmem(NI_ZOOM,NJ_ZOOM,'L','',ig1,ig2,ig3,ig4,ax,ay)
        err = gdll(gid_zoom,lat_zoom,lon_zoom)
        call handle_error(err,'fst2ptxt','Obtaining lat/lon from zoom grid')
        allocate(ptz(NI_ZOOM,NJ_ZOOM,size(ip1t)),ttz(NI_ZOOM,NJ_ZOOM,size(ip1t)), &
             trz(NI_ZOOM,NJ_ZOOM,size(ip1t),ntr),stat=err)
        call handle_error(err,'fst2ptxt','Allocating zoom thermo-level fields')
        allocate(pmz(NI_ZOOM,NJ_ZOOM,size(ip1m)),gzz(NI_ZOOM,NJ_ZOOM,size(ip1m)), &
             uuz(NI_ZOOM,NJ_ZOOM,size(ip1m)),vvz(NI_ZOOM,NJ_ZOOM,size(ip1m)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating zoom momentum-level fields')
        do k=1,size(ip1m)
           err = ezdefset(gid_zoom,fld(ip0)%gid)
           call handle_error(err,'fst2ptxt','EZDEFSET for map->zoom interpolation for pressure (M)')
           err = ezsint(pmz(:,:,k),presm(:,:,k))
           call handle_error(err,'fst2ptxt','EZSINT for pressure (M) zoom interpolation')
           err = ezdefset(gid_zoom,fld(igz)%gid)
           call handle_error(err,'fst2ptxt','EZDEFSET for map->zoom interpolation for '//trim(fld(igz)%in))
           err = ezsint(gzz(:,:,k),fld(igz)%v3d(:,:,k))
           call handle_error(err,'fst2ptxt','EZSINT for '//trim(fld(igz)%in)//' zoom interpolation')
           err = ezdefset(gid_zoom,fld(iuu)%gid)
           call handle_error(err,'fst2ptxt','EZDEFSET for map->zoom interpolation for winds')
           err = ezuvint(uuz(:,:,k),vvz(:,:,k),fld(iuu)%v3d(:,:,k),fld(ivv)%v3d(:,:,k))
           call handle_error(err,'fst2ptxt','EZUVINT for vector zoom interpolation')
        enddo
        do k=1,size(ip1t)
           err = ezdefset(gid_zoom,fld(ip0)%gid)
           call handle_error(err,'fst2ptxt','EZDEFSET for map->zoom interpolation for pressure (T)')
           err = ezsint(ptz(:,:,k),prest(:,:,k))
           call handle_error(err,'fst2ptxt','EZSINT for pressure (T) zoom interpolation')
           err = ezdefset(gid_zoom,fld(itt)%gid)
           call handle_error(err,'fst2ptxt','EZDEFSET for map->zoom interpolation for '//trim(fld(itt)%in))
           err = ezsint(ttz(:,:,k),fld(itt)%v3d(:,:,k))
           call handle_error(err,'fst2ptxt','EZSINT for '//trim(fld(itt)%in)//' zoom interpolation')
           do i=1,ntr
              err = ezdefset(gid_zoom,fld(itr+i-1)%gid)
              call handle_error(err,'fst2ptxt','EZDEFSET for map->zoom interpolation for '//trim(fld(itr+i-1)%in))
              err = ezsint(trz(:,:,k,i),fld(itr+i-1)%v3d(:,:,k))
              call handle_error(err,'fst2ptxt','EZSINT for '//trim(fld(itr+i-1)%in)//' zoom interpolation')
           enddo
        enddo

        ! Create pressure-level data for all fields at the levels defined in the 
        ! chosen profile.  This will allow subsequent calculations and interpolations
        ! to be performed on quasi-horizontal surfaces.
        allocate(uut_p(NI_ZOOM,NJ_ZOOM,size(ip1t)),vvt_p(NI_ZOOM,NJ_ZOOM,size(ip1t)), &
             ttt_p(NI_ZOOM,NJ_ZOOM,size(ip1t)),profpt_p(NI_ZOOM,NJ_ZOOM,size(ip1t)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating thermo-level pressure fields')
        allocate(uum_p(NI_ZOOM,NJ_ZOOM,size(ip1m)),vvm_p(NI_ZOOM,NJ_ZOOM,size(ip1m)), &
             gzm_p(NI_ZOOM,NJ_ZOOM,size(ip1m)),profpm_p(NI_ZOOM,NJ_ZOOM,size(ip1t)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating momentum-level pressure fields')
        if (ntr > 0) then
           allocate(trt_p(NI_ZOOM,NJ_ZOOM,size(ip1t),ntr),stat=err)
           call handle_error(err,'fst2ptxt','Allocating thermo-level tracer fields')
        endif
        call vertint2(uum_p,profpm,size(profpm,dim=3),uuz,pmz,size(ip1m), &
             1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,varname='UU',inttype='cubic')
        call vertint2(vvm_p,profpm,size(profpm,dim=3),vvz,pmz,size(ip1m), &
             1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,varname='VV',inttype='cubic')
        call vertint2(gzm_p,profpm,size(profpm,dim=3),gzz,pmz,size(ip1m), &
             1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,varname='GZ',inttype='cubic')
        call vertint2(uut_p,profpt,size(profpt,dim=3),uuz,pmz,size(ip1m), &
             1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,varname='UU',inttype='cubic')
        call vertint2(vvt_p,profpt,size(profpt,dim=3),vvz,pmz,size(ip1m), &
             1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,varname='VV',inttype='cubic')
        call vertint2(ttt_p,profpt,size(profpt,dim=3),ttz,ptz,size(ip1t), &
             1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,varname='TT',inttype='cubic')
        do k=1,ntr
           call vertint2(trt_p,profpt,size(profpt,dim=3),trz,ptz,size(ip1t), &
                1,NI_ZOOM,1,NJ_ZOOM,1,NI_ZOOM,1,NJ_ZOOM,inttype='cubic')
        enddo

        ! Precompute fields for calculations
        i_zoom = (NI_ZOOM+1)/2; j_zoom = (NJ_ZOOM+1)/2
        fcor = 2.*omega*sin(lat_zoom*pi/180.)
        fcor_clipped = sign(max(abs(fcor),2.5e-5),fcor) !clip within about 10deg of the equator

        ! Write type-specific header
        !   n_sfc, sfc_varnames
        !   n_vars, atm_varnames
        k = 2*ntr+9
        header_entries = trim(fld(iww)%out)//' GEO_UU GEO_VV ADV_UU ADV_VV ADV_TT'
        do i=itr,itr+ntr-1
           header_entries = trim(header_entries)//' ADV_'//trim(fld(i)%short)
        enddo
        header_entries = trim(header_entries)//' BKG_UU BKG_VV BKG_TT '
        do i=itr,itr+ntr-1
           header_entries = trim(header_entries)//' BKG_'//trim(fld(i)%short)
        enddo
        call add_to_header(fld(iuu_dyn),header_entries,k,vectorPair=fld(ivv_dyn))
        call add_to_header(fld(itt_dyn),header_entries,k)
        call add_to_header(fld(ihu_dyn),header_entries,k)
        write(maxfld,'(i10)') k
        write(fdout,'(i1,x,a)') 1,trim(fld(ip0)%out)
        write(fdout,'(a,x,'//trim(maxfld)//'(x,a))') trim(maxfld),trim(header_entries)
 
        ! Write prescribed (pre-interpolated) P0
        write(fdout,datafmt) trim(fld(ip0)%out),1,trim(fld(ip0)%lev),'none',p0,p0

        ! Interpolate and write prescribed WW
        call interp_write(fdout,fld(iww),profpt,vinterp,datafmt)

        ! Compute and write geostrophic winds
        allocate(geo(size(ip1m)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating geo')
        do k=1,size(geo)
           geo(k) = -grav*der(gzm_p(:,:,k),'y',lat_zoom,lon_zoom)/fcor_clipped(i_zoom,j_zoom)
        enddo
        call write_term(fdout,'GEO_UU',geo,profpm(1,1,:),'M',vinterp,datafmt)
        do k=1,size(geo)
           geo(k) = grav*der(gzm_p(:,:,k),'x',lat_zoom,lon_zoom)/fcor_clipped(i_zoom,j_zoom)
        enddo
        call write_term(fdout,'GEO_VV',geo,profpm(1,1,:),'M',vinterp,datafmt)
        deallocate(geo,stat=err)
        call handle_error(err,'fst2ptxt','Freeing geo')

        ! Compute and write advections (use model-generated advection fields if available, 
        ! but only after removing the vertical component of advection to make it isobaric)
        allocate(advm(size(ip1m)),advt(size(ip1t)),stat=err)
        call handle_error(err,'fst2ptxt','Allocating advm/advt')
        do k=1,size(ip1m)
           advm(k) = - (uum_p(i_zoom,j_zoom,k)*der(uum_p(:,:,k),'x',lat_zoom,lon_zoom) + &
                vvm_p(i_zoom,j_zoom,k)*der(uum_p(:,:,k),'y',lat_zoom,lon_zoom))
        enddo
        call write_term(fdout,'ADV_UU',advm,profpm(1,1,:),'M',vinterp,datafmt)
        do k=1,size(ip1m)
           advm(k) = - (uum_p(i_zoom,j_zoom,k)*der(vvm_p(:,:,k),'x',lat_zoom,lon_zoom) + &
                vvm_p(i_zoom,j_zoom,k)*der(vvm_p(:,:,k),'y',lat_zoom,lon_zoom))
        enddo
        call write_term(fdout,'ADV_VV',advm,profpm(1,1,:),'M',vinterp,datafmt)
        do k=1,size(ip1t)
           advt(k) = - (uut_p(i_zoom,j_zoom,k)*der(ttt_p(:,:,k),'x',lat_zoom,lon_zoom) + &
                vvt_p(i_zoom,j_zoom,k)*der(ttt_p(:,:,k),'y',lat_zoom,lon_zoom))
        enddo
        call write_term(fdout,'ADV_TT',advt,profpt(1,1,:),'T',vinterp,datafmt)
        tracer_advection: do i=1,ntr
           do k=1,size(ip1t)
              advt(k) = - (uut_p(i_zoom,j_zoom,k)*der(trt_p(:,:,k,i),'x',lat_zoom,lon_zoom) + &
                   vvt_p(i_zoom,j_zoom,k)*der(trt_p(:,:,k,i),'y',lat_zoom,lon_zoom))
           enddo
           call write_term(fdout,'ADV_'//trim(fld(itr+i-1)%short),advt,profpt(1,1,:),'T',vinterp,datafmt)
        enddo tracer_advection
        deallocate(advm,advt,stat=err)
        call handle_error(err,'fst2ptxt','Freeing advm/advt')

        ! Write backgrounds
        call write_term(fdout,'BKG_UU',uuz(i_zoom,j_zoom,:),profpm(1,1,:),'M',vinterp,datafmt)     
        call write_term(fdout,'BKG_VV',vvz(i_zoom,j_zoom,:),profpm(1,1,:),'M',vinterp,datafmt)     
        call write_term(fdout,'BKG_TT',ttz(i_zoom,j_zoom,:),profpt(1,1,:),'T',vinterp,datafmt)     
        do i=1,ntr
           call write_term(fdout,'BKG_'//trim(fld(itr+i-1)%short),trz(i_zoom,j_zoom,:,i),profpt(1,1,:), &
                'T',vinterp,datafmt)
        enddo

        ! Write dynamics tendency terms if available
        err = ezsetopt('INTERP_DEGREE',trim(interp))
        call handle_error(err,'fst2ptxt','EZSETOPT interp degree request '//trim(interp))
        if (fld(iuu_dyn)%found .and. fld(ivv_dyn)%found) then
           call interpv_write(fdout,fld(iuu_dyn),fld(ivv_dyn),profpm,vinterp,datafmt)
        endif
        if (fld(itt_dyn)%found) call interp_write(fdout,fld(itt_dyn),profpt,vinterp,datafmt)
        if (fld(ihu_dyn)%found) call interp_write(fdout,fld(ihu_dyn),profpt,vinterp,datafmt)

        ! Heap cleanup
        deallocate(ttt_p,uut_p,vvt_p,trt_p,profpt_p,uum_p,vvm_p,gzm_p,profpm_p,stat=err)
        call handle_error(err,'fst2ptxt','Freeing pressure-coordinate fields')
        deallocate(pmz,uuz,vvz,gzz,ptz,ttz,trz,stat=err)
        call handle_error(err,'fst2ptxt','Freeing zoom fields')

     endif initialization

     ! Close output file
     close(unit=fdout)
     err = fclos(fdout)
     call handle_error(err,'fst2ptxt','Releasing lock for '//trim(outfile))

  enddo dates

  ! Garbage collection
  deallocate(im,it,stat=err)
  call handle_error(err,'fst2ptxt','Freeing im/it')
  deallocate(dateList,keyList,stat=err)
  call handle_error(err,'fst2ptxt','Freeing dateList/keyList')
  deallocate(keysm,keyst,stat=err)
  call handle_error(err,'fst2ptxt','Freeing keysm/keyst')
  deallocate(ip1t,ip1m,stat=err)
  call handle_error(err,'fst2ptxt','Freeing ip1t/ip1m')

  ! Close input file
  err = fstfrm(fdin)
  call handle_error(err,'fst2ptxt','Closing '//trim(infile))
  err = fclos(fdin)
  call handle_error(err,'fst2ptxt','Releasing lock for '//trim(infile))

  ! Shut down parallel libraries
  call RPN_COMM_finalize(err)

contains

  subroutine write_term(fd,name,f,p,lev,int,fmt)
    ! Write a forcing term to the selected file
    implicit none
    integer, intent(in) :: fd
    character(len=*), intent(in) :: name,lev,int,fmt
    real, dimension(:), intent(in) :: f,p
    integer :: i
    write(fd,fmt) trim(name),size(f),trim(lev),trim(int),(p(i),i=1,size(f)),(f(i),i=1,size(f))
  end subroutine write_term

  function isPositional(F_name) result(pos)
    ! Check if the provided name is a positional record
    implicit none
    character(len=*), intent(in) :: F_name
    logical :: pos
    if (trim(F_name) == '^^' .or. trim(F_name) == '>>' .or. &
         trim(F_name) == '!!' .or. trim(F_name) == 'HY') then
       pos = .true.
    else
       pos = .false.
    endif
  end function isPositional

  function sort(vec,indx) result(err)
    ! Sort the vector in ascending order with indeces in indx (could be
    ! replaced with a faster sorting algorithm eventually)
    implicit none
    real, dimension(:) :: vec
    integer, dimension(:) :: indx
    integer :: err,k
    integer, dimension(1) :: mpos
    real :: big
    real, dimension(size(vec)) :: vec_tmp
    err = -1
    vec_tmp = vec
    big = maxval(vec_tmp)
    do k=1,size(vec_tmp)
       mpos = minloc(vec_tmp)
       indx(k) = mpos(1)
       vec_tmp(mpos(1)) = vec_tmp(mpos(1))+big
    enddo
    err = 0
  end function sort

  subroutine interpv_write(fd,fu,fv,p,int,fmt)
    ! Perform vector interpolation and write to output file
    integer, intent(in) :: fd
    type(field), intent(in) :: fu,fv
    real, dimension(:,:,:), intent(in) :: p
    character(len=*), intent(in) :: int,fmt
    integer :: k,err
    real, dimension(1,1,size(p,dim=3)) :: pu,pv,su,sv,spd,dir
    real, dimension(size(fu%v3d,dim=1),size(fu%v3d,dim=2),size(fu%v3d,dim=3)) :: v_on_ugrid
    integer, external :: ezdefset,ezuvint,gdwdfuv,ezsetopt,ezsetval
    if (fu%gid /= fv%gid) then
       call handle_error(ezsetopt('EXTRAP_DEGREE','VALUE'),'interpv_write','Setting extrapolation type')
       call handle_error(ezsetval('EXTRAP_VALUE',0.),'interpv_write','Setting extrapolation value')
       call handle_error(ezdefset(fu%gid,fv%gid),'interpv_write','Grid definition for V scalar interpolation')
       do k=1,size(p,dim=3)
          call handle_error(ezsint(v_on_ugrid(:,:,k),fv%v3d(:,:,k)),'interpv_write','Scalar V-to-Ugrid interpolation')
       enddo
       call handle_error(ezsetopt('EXTRAP_DEGREE','ABORT'),'interpv_write','Resetting extrapolation type')
    else
       v_on_ugrid = fv%v3d
    endif
    call handle_error(ezdefset(gid_prof,fu%gid),'interpv_write', &
         'EZDEFSET for map->vector profile interpolation for '//trim(fu%in)//' and '//trim(fv%in))
    do k=1,size(p,dim=3)
       call handle_error(ezuvint(su(:,:,k),sv(:,:,k),fu%v3d(:,:,k),v_on_ugrid(:,:,k)), &
            'interpv_write','EZUVINT interpolation for '//trim(fu%in)//' and '//trim(fv%in))
    enddo
    call write_fld(fd,fu,su,p,int,fmt)
    call write_fld(fd,fv,sv,p,int,fmt)
  end subroutine interpv_write

  subroutine interp_write(fd,f,p,int,fmt)
    ! Perform scalar interpolation and write to output file
    integer, intent(in) :: fd
    type(field), intent(in) :: f
    real, dimension(:,:,:), intent(in) :: p
    character(len=*), intent(in) :: int,fmt
    integer :: k,err
    real, dimension(1,1,size(p,dim=3)) :: s
    integer, external :: ezdefset,ezsint
    err = ezdefset(gid_prof,f%gid)
    call handle_error(err,'fst2ptxt::interp_write','EZDEFSET for map->profile interpolation for '//trim(f%in))
    do k=1,size(p,dim=3)
       err = ezsint(s(:,:,k),f%v3d(:,:,k))
       call handle_error(err,'fst2ptxt::interp_write','EZSINT interpolation for '//trim(f%in))
    enddo
    call write_fld(fd,f,s,p,int,fmt)
  end subroutine interp_write

  subroutine write_fld(fd,f,s,p,int,fmt)
    ! Write an interpolated field to the output file
    integer, intent(in) :: fd
    type(field), intent(in) :: f
    real, dimension(:,:,:), intent(in) :: s,p
    character(len=*), intent(in) :: int,fmt
    integer :: i
    write(fd,fmt) trim(f%out),size(p,dim=3),trim(f%lev),trim(int), &
         (p(1,1,i),i=1,size(p,dim=3)),(s(1,1,i),i=1,size(p,dim=3))
  end subroutine write_fld

  function path_length(lat1,lon1,lat2,lon2) result(dist)
    ! Compute the distance in degrees of latitude between two points on the 
    ! sphere using the Haversine formula
    real, intent(in) :: lat1,lon1,lat2,lon2
    real :: dist
    real :: dlat,dlon,a_val,dgtord,pi
    call constnt(pi,err,'PI',0)
    call handle_error(err,'fst2ptxt::path_length','Retrieving constant PI')
    dgtord = pi/180.
    dlat = abs(lat2-lat1)*dgtord
    dlon = abs(lon2-lon1)*dgtord
    a_val = sin(dlat/2.)**2 + cos(lat1*dgtord)*cos(lat2*dgtord)*sin(dlon/2)**2
    if (a_val < 0. .or. a_val >= 1.) then
       call handle_error(-1,'fst2ptxt::path_length','Computing spherical distance')
    else
       dist = 2.*atan(sqrt(a_val)/sqrt(1.-a_val))/dgtord
    endif
  end function path_length

  function der(f,dir,lat,lon) result(d)
    ! Compute a derivative at the centre of the region provided
    real, dimension(:,:), intent(in) :: f,lat,lon
    character(len=*), intent(in) :: dir
    real :: d
    integer :: ni,nj,i,j
    real :: rayt,pi,scale
    call constnt(pi,err,'PI',0)
    call handle_error(err,'fst2ptxt::der','Retrieving constant PI')
    call constnt(rayt,err,'RAYT',0)
    call handle_error(err,'fst2ptxt::der','Retrieving constant RAYT')
    scale = rayt*pi/180.
    ni = size(f,dim=1); nj = size(f,dim=2)
    i = (ni+1)/2; j = (nj+1)/2
    select case (trim(dir))
    case ('x')
       d = (f(i+1,j)-f(i-1,j)) / scale / path_length(lat(i+1,j),lon(i+1,j),lat(i-1,j),lon(i-1,j))
    case ('y')
       d = (f(i,j+1)-f(i,j-1)) / scale / path_length(lat(i,j+1),lon(i,j+1),lat(i,j-1),lon(i,j-1))
    case DEFAULT
       call handle_error(-1,'fst2ptxt::der','Invalid derivative direction '//trim(dir))
    end select
  end function der

  subroutine add_to_header(f,header,cnt,vectorPair)
    ! Add an entry to the header if the requested data is found
    type(field), intent(in) :: f
    type(field), intent(in), optional :: vectorPair
    character(len=*), intent(inout) :: header
    integer, intent(inout) :: cnt
    type(field) :: v
    v = f
    if (present(vectorPair)) v = vectorPair
    if (f%found .and. v%found) then
       header = trim(header)//' '//trim(f%out)
       cnt = cnt+1
       if (present(vectorPair)) then
          header = trim(header)//' '//trim(v%out)
          cnt = cnt+1
       endif
    endif
  end subroutine add_to_header

end subroutine fst2ptxt
