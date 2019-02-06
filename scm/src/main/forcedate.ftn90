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

subroutine forcedate()
  ! Extract and set the dateo / deet / npas set for a specific field in a file

  implicit none

  ! Local parameters
  integer, parameter :: LONG_CHAR=1024

  ! Local variables
  integer :: i,err,fdin,fdout,nrec,dateo,deet,npas,nbits,datyp, &
       ip1,ip2,ip3,ip4,ni,nj,nk,ig1,ig2,ig3,ig4,swa,lng,dltf,ubc, &
       ex1,ex2,ex3,fdate,fdeet,fnpas,nkeys,pex,pey,pe_local,pe_total
  integer, dimension(:), allocatable :: keyList
  real, dimension(1) :: work
  real, dimension(:,:), allocatable :: fld
  character(len=1) :: typvar,grtyp
  character(len=4) :: nomvar
  character(len=12) :: etiket
  character(len=LONG_CHAR) :: infile,outfile,fld_name
  character(len=LONG_CHAR), dimension(:), allocatable :: args

  ! External functions
  integer, external :: iargc,fnom,fclos,fstouv,fstfrm,fstinl,fstprm,fstluk,fstecr,utils_topology

  ! Initialize RPN COMM libraries
  pex = 0; pey = 0
  call RPN_COMM_init(utils_topology,pe_local,pe_total,pex,pey)

  ! Get positional command line arguments
  i = iargc()
  if (i < 6) call handle_error(-1,'forcedate','Usage: forcedate INPUT OUTPUT FIELD_NAME DATE DEET NPAS')
  allocate(args(i),stat=err)
  call handle_error(err,'forcedate','Allocating space for args')
  do i=1,size(args)
     call getarg(i,args(i))
  enddo
  infile = args(1)
  outfile = args(2)
  fld_name = args(3)
  read(args(4),'(i)') fdate
  read(args(5),'(i)') fdeet
  read(args(6),'(i)') fnpas
  deallocate(args,stat=err)
  call handle_error(err,'forcedate','Freeing args')

  ! Open input and output files
  fdin = 0
  err = fnom(fdin,trim(infile),'STD',0)
  call handle_error(err,'forcedate','Acquiring lock for '//trim(infile))
  nrec = fstouv(fdin,'RND')
  call handle_error_l(nrec>=0,'forcedate','Opening '//trim(infile))
  fdout = 0
  err = fnom(fdout,trim(outfile),'STD',0)
  call handle_error(err,'forcedate','Acquiring lock for '//trim(outfile))
  i = fstouv(fdout,'RND')
  call handle_error_l(i>=0,'forcedate','Opening '//trim(infile))

  ! Loop over all entries to copy or redefine fields
  allocate(keyList(nrec),stat=err)
  call handle_error(err,'forcedate','Allocating keyList')
  err = fstinl(fdin,ni,nj,nk,-1,'',-1,-1,-1,'','',keyList,nkeys,size(keyList))
  call handle_error(err,'forcedate','Getting key list')
  call handle_error_l(nkeys<=nrec,'forcedate','Insufficient number of keys')
  do i=1,nkeys
     err = fstprm(keyList(i),dateo,deet,npas,ni,nj,nk,nbits,datyp, &
          ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4, &
          swa,lng,dltf,ubc,ex1,ex2,ex3)
     call handle_error(err,'forcedate','Getting record information')
     if (trim(nomvar) == trim(fld_name)) then
        allocate(fld(ni,nj),stat=err)
        call handle_error(err,'forcedate','Allocating fld')
        err = fstluk(fld,keyList(i),ni,nj,nk)
        call handle_error(err,'forcedate','Reading record from '//trim(infile))
        err = fstecr(fld,work,-nbits,fdout,fdate,fdeet,fnpas,ni,nj,nk,ip1, &
             int((fdeet*fnpas)/3600.),ip3,typvar,nomvar,etiket,grtyp,ig1,ig2, &
             ig3,ig4,datyp,.true.)
        call handle_error(err,'forcedate','Writing record to '//trim(outfile))
        deallocate(fld,stat=err)
        call handle_error(err,'forcedate','Freeing fld')
     endif
  enddo

  ! Close input and output files
  err = fstfrm(fdin)
  call handle_error(err,'forcedate','Closing '//trim(infile))
  err = fclos(fdin)
  call handle_error(err,'forcedate','Releasing lock for '//trim(infile))
  err = fstfrm(fdout)
  call handle_error(err,'forcedate','Closing '//trim(outfile))
  err = fclos(fdout)
  call handle_error(err,'forcedate','Releasing lock for '//trim(outfile))

  ! Shut down parallel libraries
  call RPN_COMM_finalize(err)

end subroutine forcedate
