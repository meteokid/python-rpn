*/* RMNLIB - Library of useful routines for C and FORTRAN programming
* * Copyright (C) 1975-2001  Division de Recherche en Prevision Numerique
* *                          Environnement Canada
* *
* * This library is free software; you can redistribute it and/or
* * modify it under the terms of the GNU Lesser General Public
* * License as published by the Free Software Foundation,
* * version 2.1 of the License.
* *
* * This library is distributed in the hope that it will be useful,
* * but WITHOUT ANY WARRANTY; without even the implied warranty of
* * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* * Lesser General Public License for more details.
* *
* * You should have received a copy of the GNU Lesser General Public
* * License along with this library; if not, write to the
* * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
* * Boston, MA 02111-1307, USA.
* */

        INTEGER FUNCTION RPN_COMM_init_yinyang (Userinit,
     $   Pelocal,Petotal,Pex,Pey,Pintracomm,Pintercomm,
     $   Pecolor,Glb_myproc,Glb_numproc,Ngrids)
	use rpn_comm
	implicit none
        integer Pecolor,Pintercomm
        integer tag,Glb_myproc,Glb_numproc
        integer Pelocal,Petotal,Pex,Pey,Pintracomm,Ngrids
        external Userinit
*
*revision Qaddouri/Lee -from RPN_COMM_init(ver 301) for Yinyang- Mar 2011
*
*arguments
*  I	Userinit	User routine that will be called by PE 0 to
*		get the processor grid topology if it is not supplied
*		(Pex .eq. 0 ) or (Pey .eq. 0)
*  O	Pelocal	PE rank (number) of local PE
*  O	Petotal	Number of PEs in job
*  I/O	Pex	Number of PEs along the X axis. If Pex=0 upon entry
*		it will be set to the proper value upon exit
*  I/O	Pey	Number of PEs along the Y axis. If Pey=0 upon entry
*		it will be set to the proper value upon exit
*  I    Ngrids  number of simultaneous grids
*  O    Pintracomm  Intracommunicator for Yin or Yang
*  O    Pintercomm  Intercommunicator for communication
*                   between Yin and Yang
*  O    Pecolor     Color (0 for Yin, 1 for Yan (Yang) domain)
*  O    Glb_myproc  PE rank (number) of global PE (Yin and Yang)
*  O    Glb_numproc PE rank (number) of global PE (Yin and Yang)
*
*notes
*	processor topology common /pe/ will be filled here
*	positions are calculated from 0 (ORIGIN 0)
**
*
*	include 'rpn_comm.h'
	include 'mpif.h'
*
	integer ierr, i, j, count, npe, reste, nslots, key
	logical mpi_started
	integer gr1, gr2, half
	INTEGER newgroup,rowcomm
	integer, allocatable, dimension(:) :: proc_indomm
	integer unit, ndom, lndom, nproc, procmax,procmin
	type(domm), allocatable, dimension(:) :: locdom
	logical ok, allok
	logical RPN_COMM_grank
	integer RPN_COMM_petopo
	character *256 SYSTEM_COMMAND
	character *4096 , dimension(:), allocatable :: directories
	integer ncolors,my_color
	integer,dimension(:,:),allocatable::colors
	integer,dimension(:),allocatable::colortab
        integer diag_mode
*
        RPN_COMM_init_yinyang= 0
	unit = 5
	ok = .true.
	call MPI_INITIALIZED(mpi_started,ierr)
	if (.not. mpi_started ) call MPI_init(ierr)
*	call resetenv   ! mpich1 will no longer work
        print *,'RPN_COMM_init_yinyang called'
	pe_wcomm = MPI_COMM_WORLD
	call MPI_COMM_RANK(pe_wcomm,pe_me,ierr)
	call MPI_COMM_SIZE(pe_wcomm,pe_tot,ierr)
       allocate(colortab(0:pe_tot-1))
       colortab=0
       my_color=-1
	call getenvc("RPN_COMM_DIAG",SYSTEM_COMMAND)
        if( SYSTEM_COMMAND .ne. " " ) then
          read(SYSTEM_COMMAND,*) diag_mode
        else
          diag_mode=2
        endif
*
* if environment variable RPN_COMM_DOM is set, split domain
*
	SYSTEM_COMMAND=""
	call getenvc("RPN_COMM_DOM",SYSTEM_COMMAND)
	if( SYSTEM_COMMAND .ne. " " ) then
	  read(SYSTEM_COMMAND,*) ncolors
	  allocate(directories(0:ncolors))
	  allocate(colors(3,ncolors))
	  read(SYSTEM_COMMAND,*)ncolors,colors
	  do i=1,ncolors
	    do j=colors(1,i),colors(3,i),colors(2,i)
	      colortab(j)=i
	    enddo
	  enddo
	  my_color=colortab(pe_me)
	  call MPI_COMM_SPLIT(MPI_COMM_WORLD,my_color,
     &                        pe_me,pe_wcomm,ierr)
          RPN_COMM_init_yinyang= my_color
	  SYSTEM_COMMAND=""
	  call getenvc("RPN_COMM_DIRS",SYSTEM_COMMAND)
	  read(SYSTEM_COMMAND,*)directories
	  if(diag_mode.ge.2)
     &       print *,"my directory is:",trim(directories(my_color))
	  call RPN_COMM_chdir(trim(directories(my_color)))
!	  open(unit,file='rpn_comm_explicit_map.cfg',
!     %         status='OLD',err=50)
!	  do i=0,pe_me-1
!	    read(unit,*,end=40,err=40)
!	  enddo
!	  read(unit,'(a)',end=40,err=40)SYSTEM_COMMAND
!	NOW, cd to SYSTEM_COMMAND
!40	  continue
!	  close(unit)
!50	  continue
	  call MPI_COMM_RANK(pe_wcomm,pe_me,ierr)
	  call MPI_COMM_SIZE(pe_wcomm,pe_tot,ierr)
        endif
*
* if multiple grid environment, (re)split domain
*
*     print *,'commenting out grouping pe_indomms'
*     print *,'commenting assigning pe_wcomms,pe_indomms'
*     print *,'ngrids=',ngrids
!       pe_wcomms = pe_wcomm
!       pe_indomms = pe_wcomms
!       call MPI_COMM_GROUP(pe_indomms,pe_gr_indomms,ierr)
        if(ngrids .gt. 1) then
          my_color=pe_me/(pe_tot/Ngrids)
!          print *,"My color is:",my_color
          call MPI_COMM_SPLIT(pe_wcomms,my_color,
     &                        pe_me,pe_wcomm,ierr)
          RPN_COMM_init_yinyang= my_color
          call MPI_COMM_RANK(pe_wcomm,pe_me,ierr)
          call MPI_COMM_SIZE(pe_wcomm,pe_tot,ierr)
        endif
*
* domain split done, each domain is now on its own
*
!     call MPI_COMM_GROUP(pe_wcomm,pe_gr_wcomm,ierr)
!     call MPI_COMM_GROUP(pe_wcomms,pe_gr_wcomms,ierr)
*
* Now split for Yin and Yang
*
        Glb_myproc = pe_me  ! me in my domain
        Glb_numproc = pe_tot! number of pes in my domain
*
* Define processors for Yin and Yang domains

       half=Glb_numproc/2
       if (Glb_myproc.lt.half) then
           Pecolor=0
       else
           Pecolor=1
       endif
*      print *,'G_myproc=',pe_me,' Pecolor',Pecolor
*
* Construct 2 intracommunicators
*
       Call MPI_Comm_split(MPI_COMM_WORLD,Pecolor,Glb_myproc,
     $        Pintracomm,ierr)
*      print *,'split world ierr=',ierr,'Pintracomm=',Pintracomm
       if (Pecolor.eq.0) then
           call MPI_Intercomm_create(Pintracomm,0,MPI_COMM_WORLD,
     $                half, tag, Pintercomm,ierr)
*          print *,'Pecolor =',Pecolor,Pintercomm,tag,ierr
       endif
       if (Pecolor.eq.1)then
          call MPI_Intercomm_create(Pintracomm,0,MPI_COMM_WORLD,
     $                0, tag, Pintercomm,ierr)
*          print *,'Pecolor =',Pecolor,Pintercomm,tag,ierr
       endif
* Get new ranks for each sub-domain(Yin and Yang)
*
      call MPI_COMM_RANK(Pintracomm,pe_me,ierr)
      Pelocal = pe_me   ! me in my Yin or Yang domain
      print*,'Pelocal',Pelocal
      call MPI_COMM_SIZE(Pintracomm,pe_tot,ierr)
      Petotal = pe_tot  ! number of Pes in my Yin or Yang domain
      pe_wcomm = Pintracomm
      pe_wcomms = pe_wcomm
      pe_indomms = pe_wcomms
*
* Leave the grouping for last.
*
      call MPI_COMM_GROUP(pe_indomms,pe_gr_indomms,ierr)
      call MPI_COMM_GROUP(pe_wcomm,pe_gr_wcomm,ierr)
      call MPI_COMM_GROUP(pe_wcomms,pe_gr_wcomms,ierr)
*
*     Sub-Domain initialization
*
	ndom=0
	if(.not.allocated(locdom)) allocate(locdom(64))
	domm_size=loc(locdom(2))-loc(locdom(1))
	if(pe_me.eq.0) then
           if(diag_mode.ge.3)
     &        print *,'TRYING TO OPEN CONFIGURATION FILE rpn_comm.cfg'
	   open(unit,file='rpn_comm.cfg',status='OLD',err=100)
! 111	   format(A12,2I3,A1024)
           if(diag_mode.ge.2)
     &       print *,'READING DOMAIN CONFIGURATION FILE',' rpn_comm.cfg'
	   do while(.true.)
	      lndom=ndom+1
	      read(unit,*,end=100) locdom(lndom)%nom ,
     &       locdom(lndom)%npex,locdom(lndom)%npey,locdom(lndom)%path
	      ndom=ndom+1
	   enddo
 100	   continue
	   close(unit)
 110	   continue
	endif
*
	call MPI_BCAST(ndom,1,MPI_INTEGER,0,pe_wcomm,ierr)
	allocate(pe_domains(ndom+1))
	if((pe_me.eq.0).and.(ndom.gt.0)) then
	   pe_domains(1:ndom)=locdom(1:ndom)
	endif
	if(allocated(locdom)) deallocate(locdom)
	call MPI_BCAST(pe_domains,domm_size*ndom,MPI_CHARACTER,0,
     &        pe_wcomm,ierr)
	if(ndom.eq.0) then
	   pe_domains(1)%nom='DOM1'
	   pe_domains(1)%npex=pe_tot
	   pe_domains(1)%npey=1
	   pe_domains(1)%path='./'
	   ndom=1
	endif
	pe_pe0 = -1
	nproc = pe_tot-1
	domm_num=-1
	do i=1,ndom
	   procmax = nproc
	   procmin = procmax - pe_domains(i)%npex*pe_domains(i)%npey +1
	   if((pe_me.le.procmax).and.(pe_me.ge.procmin)) then
	      pe_pe0=procmin
	      domm_num = i
	      pe_dommtot = pe_domains(i)%npex*pe_domains(i)%npey
	   endif
	   if(procmin.lt.0) then
	      if(pe_me.eq.0) then
		 write(rpn_u,*) 
     &               'ERROR RPN_COMM_init: not enough PEs for domain '
     &               ,pe_domains(i)%nom
	      endif
	      ok = .false.
	   endif
	   nproc = procmin -1
	enddo
	call MPI_COMM_split(pe_wcomm,domm_num,pe_me,pe_indomm,ierr)
	if(nproc.ge.0) then
	  if(pe_me.eq.0) then
            write(rpn_u,*) "ERROR RPN_COMM_init: some processors are 
     &                  not in a"
            write(rpn_u,*) "       specific domain, please check 
     &                  rpn_comm.cfg"
            write(rpn_u,*) "       Unused PEs = ", nproc+1
	    write(rpn_u,*) "       You should launch with "
     &                  ,Petotal-(nproc+1)," PEs."
	  endif
          call RPN_COMM_finalize(ierr)
          stop
	endif
        if(pe_domains(domm_num)%nom .eq. 'DUMMY') then
          ok=.true.
	  call RPN_COMM_allreduce(ok, allok,1,'MPI_LOGICAL','MPI_LAND',
     &         'ALL',ierr)
          write(SYSTEM_COMMAND,*)trim(pe_domains(domm_num)%path),
     &                           pe_me-pe_pe0,pe_me
          if(diag_mode.ge.1)
     &      print *,'DUMMY DOMAIN, executing:',trim(SYSTEM_COMMAND)
          call system(trim(SYSTEM_COMMAND))
          call RPN_COMM_finalize(ierr)
          stop
        else
          if(ndom.gt.1) then
            if(diag_mode.ge.2) then
              print *,'DOMAIN=',pe_domains(domm_num)%nom
              print *,'executing at:',trim(pe_domains(domm_num)%path)
            endif
          endif
	  call RPN_COMM_chdir(pe_domains(domm_num)%path)
        endif
	if(pe_me .eq. pe_pe0)then
	  if ( Pex.eq.0 .or. Pey.eq.0  ) then ! get processor topology
	    WORLD_pe(1)=pe_domains(domm_num)%npex
	    WORLD_pe(2)=pe_domains(domm_num)%npey
	    call Userinit(WORLD_pe(1),WORLD_pe(2))
!	    write(rpn_u,*) WORLD_pe(1),WORLD_pe(2)
	    if(WORLD_pe(1)*WORLD_pe(2).ne.pe_dommtot) then
	     ok = .false.
	     write(rpn_u,*) 'RPN_COMM_init: Inconsistency between'
	     write(rpn_u,*) 'userinit Subroutine and total number'
             write(rpn_u,*) 'of PE: please check'
	    endif
	   if(diag_mode.ge.1) write(rpn_u,*)'Requested topology = ',
     &               WORLD_pe(1),' by ',WORLD_pe(2)
	   if(diag_mode.ge.1) write(rpn_u,*)'Domain set for '
     &               ,pe_dommtot,' processes'
          else
            write(rpn_u,*) 'RPN_COMM_init: Forced topology'
	    WORLD_pe(1) = Pex
	    WORLD_pe(2) = Pey
	    if(WORLD_pe(1)*WORLD_pe(2).ne.pe_dommtot) then
	     ok = .false.
	     write(rpn_u,*) 'RPN_COMM_init: Inconsistency between Pex'
	     write(rpn_u,*) 'and Pey args and total number of PE: '
	     write(rpn_u,*) 'please check'
	    endif
	   if(diag_mode.ge.1)
     &       write(rpn_u,*)'Requested topology = ',WORLD_pe(1),' by '
     &             ,WORLD_pe(2)
	  endif
*
	  if(WORLD_pe(1)*WORLD_pe(2) .gt. pe_dommtot) then
	    write(rpn_u,*)' ERROR: not enough PEs for decomposition '
	    write(rpn_u,*)' REQUESTED=',WORLD_pe(1)*WORLD_pe(2),
     &              ' AVAILABLE=',pe_dommtot
	    ok = .false.
	  endif
	endif
*
	call RPN_COMM_allreduce(ok, allok, 1,'MPI_LOGICAL','MPI_LAND',
     &         'ALL',ierr)
	if(.not.allok) then
	   call RPN_COMM_finalize(ierr)
	   stop
	endif
*
*	send WORLD topology to all PEs. That will allow all PEs
*	to compute other PE topology parameters locally.
*       for doing this, we need to define some basic domains
*       communicators.

!	pe_indomm = -1
	pe_gr_indomm = -1

	do j=1,ndom
	   if(domm_num.eq.j) then
	      allocate(proc_indomm( pe_domains(domm_num)%npex*
     &	           pe_domains(domm_num)%npey))
	      do i=1,pe_dommtot
		 proc_indomm(i)=pe_pe0+i-1
	      enddo
	      call MPI_Group_incl(pe_gr_wcomm, pe_dommtot,proc_indomm,
     &   	   pe_gr_indomm,ierr)
!	   call MPI_Comm_create(pe_wcomm,pe_gr_indomm, pe_indomm, ierr)
	     
*
*       for each communicator, store the corresponding group
*
	   endif
	enddo
	call MPI_COMM_rank(pe_indomm,pe_medomm,ierr)
	pe_defcomm = pe_indomm
	pe_defgroup = pe_gr_indomm
	
	call MPI_BCAST(WORLD_pe,2,MPI_INTEGER,0,
     &	               pe_indomm,ierr)
	
	if ( Pex.eq.0 .or. Pey.eq.0  ) then ! return processor topology
	  Pex = WORLD_pe(1)
	  Pey = WORLD_pe(2)
	endif
*
	pe_nx=WORLD_pe(1)
	pe_ny=WORLD_pe(2)
*
*	pe_pe0 is not equal to 0 if there are more than one domain
*	computational grid
*
	count = pe_pe0
*
*	fill tables containing the position along the X axis (pe_xtab)
*	and along the Y axis (pe_ytab) for all processors

	ierr = RPN_COMM_petopo(WORLD_pe(1),WORLD_pe(2))

	BLOC_EXIST   =.false.
        BLOC_SIZEX   = 1
        BLOC_SIZEY   = 1
        BLOC_mybloc  = 0
        BLOC_myblocx = 0
        BLOC_myblocy = 0
        BLOC_me      = pe_me
	BLOC_comm_world = pe_indomm
	BLOC_comm_row = pe_myrow
	BLOC_comm_col = pe_mycol
	BLOC_corner = pe_pe0
	BLOC_master = 0
	if(pe_me.eq.pe_pe0) then
	   BLOC_master=1
	endif
	pe_bloc = pe_indomm

	call MPI_Group_incl(pe_gr_indomm,1,0,pe_gr_blocmaster,ierr) 
	call MPI_Comm_create(pe_indomm,pe_gr_blocmaster, 
     &            pe_blocmaster, ierr)

*       for each communicator, store the corresponding group
*
	   call MPI_COMM_GROUP(pe_myrow,pe_gr_myrow,ierr)
	   call MPI_COMM_GROUP(pe_mycol,pe_gr_mycol,ierr)
	   call MPI_COMM_GROUP(pe_bloc,pe_gr_bloc,ierr)
	return
	end
