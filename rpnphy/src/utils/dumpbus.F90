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
!**s/r dumpini3 - Dump interface bus statistics
!
      Subroutine dumpini3( esize, dsize, fsize, vsize, npe, ni,nj, gni,gnj, DoType_S )
!
      implicit none
#include <arch_specific.hf>
!
      logical    do_entry
      character(len=*) DoType_S
      integer    npe, ni, nj, gni, gnj
      integer    esize , dsize , fsize , vsize
      real       e(1)  , d(1)  , f(1)  , v(1)
      integer    slice , lun   , step, pe
!
!author bernard dugas - rpn - july 2000
!
!revision
! 001      B. Dugas    (Nov 2002) - make code MPI aware
! 002      B. Dugas    (Jun 2005) - character*4 names in dumpwrit2
! 003      B. Dugas    (Dec 2007) - use ALLOCATE rather than HPALLOC
! 004      B. Dugas    (Mar 2008) - REAL(16) calculations under AIX
! 005      B. Dugas    (Sep 2008) - Add entry bus
! 006      B. Dugas    (Oct 2010) - Add DoType_S control argument
!                                 - Add dumpwritd and dumpbusd entry points
!
!language
!       fortran 90
!
!object(dumpini3/dumpbus3/dumpwrit3)
!    Computes and writes horizontaly averaged statitics for
!    all variables of the three main physics interface's buses.
!    Nothing is done before dumpini2 is called. Finally, note that
!    dumpini3 and dumpwrit3 should NOT be macro- or micro-tasked.
!
!arguments (dumpini3)
!    esize    - entry bus size
!    dsize    - dynamic bus size
!    fsize    - permanent bus size
!    vsize    - volatile bus size
!    npe      - total number of processing units (PE)
!    ni       - local first horizontal slice dimension
!    nj       - local second horizontal dimension (i.e. total number of slices)
!    gni      - global first horizontal dimension
!    gnj      - global second horizontal dimension
!    DoType_S - what to do: Possible values are ALL, or combinations
!               of D, P and V to treat the dynamical, permanent or
!               volatile buses, respectively
!
!arguments (dumpbus3)
!    e        - entry     bus
!    d        - dynamic   bus
!    f        - permanent bus 
!    v        - volatile  bus
!    slice    - slice ordinal (assumed smaller than nj)
!    do_entry - treat entry bus (logical key)
!
!arguments (dumpwrit3)
!    lun      - logical I/O unit
!    step     - current timestep
!    pe       - current processing unit
!    do_entry - treat entry bus (logical key)
!
!implicits 
#include "buses.cdk"
!
!local variables
!
      character(len=3) ListeDo
      integer    position,positiong
      integer    i,j,k, ier,err, nvar,nvart, busvar
!
      logical,   save :: do_dynamic=.false.,do_permanent=.false.,do_volatile =.false.
      character, save :: DoneIni*12 = 'ini not done'
      integer,   save :: nsize, G_ni,G_nj, G_npe
      integer,   save :: nslice, nvare,nvard,nvarp,nvarv
#if defined (AIX)
      real(16) :: maxs,mins,sums,sum2,rmss,curv
      real(16),  dimension(:,:,:), allocatable, save :: sampe,sampd,sampp,sampv,gamp
      real(16),  dimension(:,:),   allocatable, save :: lamp
#else
      real(8) :: maxs,mins,sums,sum2,rmss,curv
      real(8),   dimension(:,:,:), allocatable, save :: sampe,sampd,sampp,sampv,gamp
      real(8),   dimension(:,:),   allocatable, save :: lamp
#endif
!     inline function to help with indicies calculations
      integer ik
      ik(i,k) = (k-1)*nsize + i
!
!--------------------------------------------------------------------
      if (DoneIni.eq.'ini not done') then

         ! save dimensions ...

         nsize  = ni
         nslice = nj
         G_npe  = npe
         G_ni   = gni
         G_nj   = gnj
 
         nvare   = esize / nsize
         nvard   = dsize / nsize
         nvarp   = fsize / nsize
         nvarv   = vsize / nsize

         call low2up( DoType_S,ListeDo )

         if (ListeDo == 'ALL' ) then
            do_dynamic=.true. ; do_permanent=.true. ; do_volatile=.true.
         else
            if (index(ListeDo,'D') > 0) do_dynamic   = .true.
            if (index(ListeDo,'P') > 0) do_permanent = .true.
            if (index(ListeDo,'V') > 0) do_volatile  = .true.
         end if

         nvart   = nvare+nvard+nvarp+nvarv

!        ...  and allocate local and global sample space

         allocate( sampe(4,nslice,nvare) , &
                   sampd(4,nslice,nvard) , &
                   sampp(4,nslice,nvarp) , &
                   sampv(4,nslice,nvarv) , &
                   gamp (4,nvart ,G_npe) , &
                   lamp (4,nvart)        )

         DoneIni = 'ini done'

      end if

      return
!--------------------------------------------------------------------
      Entry dumpbus3( e, d, f, v, slice, do_entry )

      if ( DoneIni == 'ini not done' ) return
      if ( .not. (do_entry .or. do_dynamic .or. do_permanent .or. do_volatile) ) return

      if (do_entry) then

         ! dump entry variables bus

         do nvar = 1,nvare

            maxs = e(ik(1,nvar))
            mins = e(ik(1,nvar))
            sums = e(ik(1,nvar))
            sum2 = sums * sums

            do i=2,nsize
               curv = e(ik(i,nvar))
               maxs = max( maxs , curv )
               mins = min( mins , curv )
               sums =      sums + curv
               sum2 =      sum2 + curv * curv
            end do

            sampe(1,slice,nvar) = maxs
            sampe(2,slice,nvar) = mins
            sampe(3,slice,nvar) = sums
            sampe(4,slice,nvar) = sum2

         end do

      end if 
      if (do_dynamic) then

         ! dump dynamic variables bus

         do nvar = 1,nvard

            maxs = d(ik(1,nvar))
            mins = d(ik(1,nvar))
            sums = d(ik(1,nvar))
            sum2 = sums * sums

            do i=2,nsize
               curv = d(ik(i,nvar))
               maxs = max( maxs , curv )
               mins = min( mins , curv )
               sums =      sums + curv
               sum2 =      sum2 + curv * curv
            end do

            sampd(1,slice,nvar) = maxs
            sampd(2,slice,nvar) = mins
            sampd(3,slice,nvar) = sums
            sampd(4,slice,nvar) = sum2
            
         end do

      end if
      if (do_permanent) then

         ! dump permanent variables bus

         do nvar = 1,nvarp

            maxs = f(ik(1,nvar))
            mins = f(ik(1,nvar))
            sums = f(ik(1,nvar))
            sum2 = sums * sums

            do i=2,nsize
               curv = f(ik(i,nvar))
               maxs = max( maxs , curv )
               mins = min( mins , curv )
               sums =      sums + curv
               sum2 =      sum2 + curv * curv
            end do

            sampp(1,slice,nvar) = maxs
            sampp(2,slice,nvar) = mins
            sampp(3,slice,nvar) = sums
            sampp(4,slice,nvar) = sum2

         end do

      end if
      if (do_volatile) then

         ! dump volatile variables bus

         do nvar = 1,nvarv

            maxs = v(ik(1,nvar))
            mins = v(ik(1,nvar))
            sums = v(ik(1,nvar))
            sum2 = sums * sums

            do i=2,nsize
               curv = v(ik(i,nvar))
               maxs = max( maxs , curv )
               mins = min( mins , curv )
               sums =      sums + curv
               sum2 =      sum2 + curv * curv
            end do

            sampv(1,slice,nvar) = maxs
            sampv(2,slice,nvar) = mins
            sampv(3,slice,nvar) = sums
            sampv(4,slice,nvar) = sum2

         end do

      end if

      return
!--------------------------------------------------------------------
      Entry dumpwrit3( lun,step,pe, do_entry )

      if ( DoneIni == 'ini not done' ) return
      if ( .not. (do_entry .or. do_dynamic .or. do_permanent .or. do_volatile) ) return

      nvart = nvare+nvard+nvarp+nvarv

      ! calculate global statistics and print them.
      ! but first, do all we can on the local domain

      if (do_entry) then

         do nvar = 1,nvare

            maxs = sampe(1,1,nvar)
            mins = sampe(2,1,nvar)
            sums = sampe(3,1,nvar)
            sum2 = sampe(4,1,nvar)

            do i=2,nslice
               maxs = max( maxs , sampe(1,i,nvar) )
               mins = min( mins , sampe(2,i,nvar) )
               sums = sums      + sampe(3,i,nvar)
               sum2 = sum2      + sampe(4,i,nvar)
            end do

            lamp(1,nvar) = maxs
            lamp(2,nvar) = mins
            lamp(3,nvar) = sums
            lamp(4,nvar) = sum2

         end do

      else

         lamp(:,1:nvare) = 0.0

      end if
      if (do_dynamic) then

         do nvar = 1,nvard

            maxs = sampd(1,1,nvar)
            mins = sampd(2,1,nvar)
            sums = sampd(3,1,nvar)
            sum2 = sampd(4,1,nvar)

            do i=2,nslice
               maxs = max( maxs , sampd(1,i,nvar) )
               mins = min( mins , sampd(2,i,nvar) )
               sums = sums      + sampd(3,i,nvar)
               sum2 = sum2      + sampd(4,i,nvar)
            end do

            lamp(1,nvar+nvare) = maxs
            lamp(2,nvar+nvare) = mins
            lamp(3,nvar+nvare) = sums
            lamp(4,nvar+nvare) = sum2

         end do

      else

         lamp(:,nvare+1:nvare+nvard) = 0.0

      end if
      if (do_permanent) then

         do nvar = 1,nvarp

            maxs = sampp(1,1,nvar)
            mins = sampp(2,1,nvar)
            sums = sampp(3,1,nvar)
            sum2 = sampp(4,1,nvar)

            do i=2,nslice
               maxs = max( maxs , sampp(1,i,nvar) )
               mins = min( mins , sampp(2,i,nvar) )
               sums = sums      + sampp(3,i,nvar)
               sum2 = sum2      + sampp(4,i,nvar)
            end do

            lamp(1,nvar+nvare+nvard) = maxs
            lamp(2,nvar+nvare+nvard) = mins
            lamp(3,nvar+nvare+nvard) = sums
            lamp(4,nvar+nvare+nvard) = sum2

         end do

      else

         lamp(:,nvare+nvard+1:nvare+nvard+nvarp) = 0.0

      end if
      if (do_volatile) then

         do nvar = 1,nvarv

            maxs = sampv(1,1,nvar)
            mins = sampv(2,1,nvar)
            sums = sampv(3,1,nvar)
            sum2 = sampv(4,1,nvar)

            do i=2,nslice
               maxs = max( maxs , sampv(1,i,nvar) )
               mins = min( mins , sampv(2,i,nvar) )
               sums = sums      + sampv(3,i,nvar)
               sum2 = sum2      + sampv(4,i,nvar)
            end do

            lamp(1,nvar+nvare+nvard+nvarp) = maxs
            lamp(2,nvar+nvare+nvard+nvarp) = mins
            lamp(3,nvar+nvare+nvard+nvarp) = sums
            lamp(4,nvar+nvare+nvard+nvarp) = sum2

         end do

      else

         lamp(:,nvare+nvard+nvarp+1:nvare+nvard+nvarp+nvarv) = 0.0

      end if

      ! gather statistics from all processor on processor 0
#if defined (AIX)
      call RPN_COMM_gather( &
             lamp, 16*nvart, 'MPI_INTEGER', &
             gamp, 16*nvart, 'MPI_INTEGER', 0,'GRID',err )
#else
      call RPN_COMM_gather( &
             lamp,  4*nvart, 'MPI_REAL8', &
             gamp,  4*nvart, 'MPI_REAL8',   0,'GRID',err )
#endif
      if (lun > 0 .and. pe == 0) then

         ! produce global statistics from what was gathered

         do nvar=1,nvart

            maxs = gamp(1,nvar,1)
            mins = gamp(2,nvar,1)
            sums = gamp(3,nvar,1)
            sum2 = gamp(4,nvar,1)

            do i=2,G_npe
               maxs = max( maxs , gamp(1,nvar,i) )
               mins = min( mins , gamp(2,nvar,i) )
               sums = sums      + gamp(3,nvar,i)
               sum2 = sum2      + gamp(4,nvar,i)
            end do

            gamp(1,nvar,1) = maxs
            gamp(2,nvar,1) = mins
            gamp(3,nvar,1) = sums
            gamp(4,nvar,1) = sum2

         end do

         ! finish processing and do the printing for each bus

         if (do_entry) then ! start with entry bus if available

            write(lun,6000) step,'entry'

            busvar = 1

            do nvar=1,nvare

               maxs = gamp(1,nvar,1)
               mins = gamp(2,nvar,1)
               sums = gamp(3,nvar,1)
               sum2 = gamp(4,nvar,1)

               sums = sums / ( G_ni * G_nj )
               sum2 = sum2 / ( G_ni * G_nj )

               if (abs( sum2 - ( sums  * sums ) ) < 1.e-14 * sum2) then
                  sum2 = 0.0
               else
                  sum2 = sum2 - ( sums  * sums )
               end if

               rmss = sqrt( sum2 )

               position  = (nvar-1)*nsize+1
               positiong = (nvar-1)*G_ni+1

               if (entpar(busvar,1) == position) then
                  write(lun,6001) entnm(busvar,1),      &
                                  entnm(busvar,2)(1:4), &
                                  entdc(busvar)(1:40)
                  busvar = busvar+1
               end if

               write(lun,6002) positiong,maxs,mins,sums,rmss

            end do

         end if
         if (do_dynamic) then

            write(lun,6000) step,'dynamic'

            busvar = 1

            do nvar=1,nvard

               maxs = gamp(1,nvar+nvare,1)
               mins = gamp(2,nvar+nvare,1)
               sums = gamp(3,nvar+nvare,1)
               sum2 = gamp(4,nvar+nvare,1)

               sums = sums / ( G_ni * G_nj )
               sum2 = sum2 / ( G_ni * G_nj )

               if (abs( sum2 - ( sums  * sums ) ) < 1.e-14 * sum2) then
                  sum2 = 0.0
               else
                  sum2 = sum2 - ( sums  * sums )
               end if

               rmss = sqrt( sum2 )

               position  = (nvar-1)*nsize+1
               positiong = (nvar-1)*G_ni+1

               if (dynpar(busvar,1) == position) then
                  write(lun,6001) dynnm(busvar,1),      &
                                  dynnm(busvar,2)(1:4), &
                                  dyndc(busvar)(1:40)
                  busvar = busvar+1
               end if

               write(lun,6002) positiong,maxs,mins,sums,rmss

            end do

         end if
         if (do_permanent) then

            write(lun,6000) step,'permanent'

            busvar = 1

            do nvar=1,nvarp

               maxs = gamp(1,nvar+nvare+nvard,1)
               mins = gamp(2,nvar+nvare+nvard,1)
               sums = gamp(3,nvar+nvare+nvard,1)
               sum2 = gamp(4,nvar+nvare+nvard,1)

               sums = sums / ( G_ni * G_nj )
               sum2 = sum2 / ( G_ni * G_nj )

               if (abs( sum2 - ( sums  * sums ) ) < 1.e-14 * sum2) then
                  sum2 = 0.0
               else
                  sum2 = sum2 - ( sums  * sums )
               end if

               rmss = sqrt( sum2 )

               position  = (nvar-1)*nsize+1
               positiong = (nvar-1)*G_ni+1

               if (perpar(busvar,1) == position) then
                  write(lun,6001) pernm(busvar,1),      &
                                  pernm(busvar,2)(1:4), &
                                  perdc(busvar)(1:40)
                  busvar = busvar+1
               end if

               write(lun,6002) positiong,maxs,mins,sums,rmss

            end do

         end if
         if (do_volatile) then

            write(lun,6000) step,'volatile'

            busvar = 1

            do nvar=1,nvarv

               maxs = gamp(1,nvar+nvare+nvard+nvarp,1)
               mins = gamp(2,nvar+nvare+nvard+nvarp,1)
               sums = gamp(3,nvar+nvare+nvard+nvarp,1)
               sum2 = gamp(4,nvar+nvare+nvard+nvarp,1)

               sums = sums / ( G_ni * G_nj )
               sum2 = sum2 / ( G_ni * G_nj )

               if (abs( sum2 - ( sums  * sums ) ) < 1.e-14 * sum2) then
                  sum2 = 0.0
               else
                  sum2 = sum2 - ( sums  * sums )
               end if

               rmss = sqrt( sum2 )

               position  = (nvar-1)*nsize+1
               positiong = (nvar-1)*G_ni+1

               if (volpar(busvar,1) == position) then
                  write(lun,6001) volnm(busvar,1),      &
                                  volnm(busvar,2)(1:4), &
                                  voldc(busvar)(1:40)
                  busvar = busvar+1
               end if

               write(lun,6002) positiong,maxs,mins,sums,rmss

            end do

         end if

         call flush( lun )

      end if         

      return
!--------------------------------------------------------------------
      Entry dumpbusd( d, slice )

      if ( DoneIni == 'ini not done' ) return
      if ( .not. do_dynamic ) return

      ! dump dynamic variable bus

      do nvar = 1,nvard

         maxs = d(ik(1,nvar))
         mins = d(ik(1,nvar))
         sums = d(ik(1,nvar))
         sum2 = sums * sums

         do i=2,nsize
            curv = d(ik(i,nvar))
            maxs = max( maxs , curv )
            mins = min( mins , curv )
            sums =      sums + curv
            sum2 =      sum2 + curv * curv
         end do

         sampd(1,slice,nvar) = maxs
         sampd(2,slice,nvar) = mins
         sampd(3,slice,nvar) = sums
         sampd(4,slice,nvar) = sum2

      end do

      return
!--------------------------------------------------------------------
      Entry dumpwritd( lun,step,pe )

      if ( DoneIni == 'ini not done' ) return
      if ( .not. do_dynamic ) return

      nvart = nvare+nvard+nvarp+nvarv

      ! calculate global statistics for dynamic bus variables and
      ! print them. But first, do all we can on the local domain

      lamp(:,                  1:nvare                  ) = 0.0 ! zero entry
      lamp(:,nvare+nvard+      1:nvare+nvard+nvarp      ) = 0.0 ! zero permanent
      lamp(:,nvare+nvard+nvarp+1:nvare+nvard+nvarp+nvarv) = 0.0 ! zero volatile

      do nvar = 1,nvard

         maxs = sampd(1,1,nvar)
         mins = sampd(2,1,nvar)
         sums = sampd(3,1,nvar)
         sum2 = sampd(4,1,nvar)

         do i=2,nslice
            maxs = max( maxs , sampd(1,i,nvar) )
            mins = min( mins , sampd(2,i,nvar) )
            sums = sums      + sampd(3,i,nvar)
            sum2 = sum2      + sampd(4,i,nvar)
         end do

         lamp(1,nvar+nvare) = maxs
         lamp(2,nvar+nvare) = mins
         lamp(3,nvar+nvare) = sums
         lamp(4,nvar+nvare) = sum2

      end do

      do nvar = 1,nvarp

         maxs = sampp(1,1,nvar)
         mins = sampp(2,1,nvar)
         sums = sampp(3,1,nvar)
         sum2 = sampp(4,1,nvar)

         do i=2,nslice
            maxs = max( maxs , sampp(1,i,nvar) )
            mins = min( mins , sampp(2,i,nvar) )
            sums = sums      + sampp(3,i,nvar)
            sum2 = sum2      + sampp(4,i,nvar)
         end do

         lamp(1,nvar+nvare+nvard) = maxs
         lamp(2,nvar+nvare+nvard) = mins
         lamp(3,nvar+nvare+nvard) = sums
         lamp(4,nvar+nvare+nvard) = sum2

      end do

      ! gather statistics from all processor on processor 0
#if defined (AIX)
      call RPN_COMM_gather( &
             lamp, 16*nvart, 'MPI_INTEGER', &
             gamp, 16*nvart, 'MPI_INTEGER', 0,'GRID',err )
#else
      call RPN_COMM_gather( &
             lamp,  4*nvart, 'MPI_REAL8', &
             gamp,  4*nvart, 'MPI_REAL8',   0,'GRID',err )
#endif

      if (lun > 0 .and. pe == 0) then

         ! produce global statistics from what was gathered

         do nvar=1,nvart

            maxs = gamp(1,nvar,1)
            mins = gamp(2,nvar,1)
            sums = gamp(3,nvar,1)
            sum2 = gamp(4,nvar,1)

            do i=2,G_npe
               maxs = max( maxs , gamp(1,nvar,i) )
               mins = min( mins , gamp(2,nvar,i) )
               sums = sums      + gamp(3,nvar,i)
               sum2 = sum2      + gamp(4,nvar,i)
            end do

            gamp(1,nvar,1) = maxs
            gamp(2,nvar,1) = mins
            gamp(3,nvar,1) = sums
            gamp(4,nvar,1) = sum2

         end do

         ! finish processing and do the printing (but only for the dynamic bus)

         busvar = 1

         do nvar=1,nvard ! print dynamic bus

            maxs = gamp(1,nvar+nvare,1)
            mins = gamp(2,nvar+nvare,1)
            sums = gamp(3,nvar+nvare,1)
            sum2 = gamp(4,nvar+nvare,1)

            sums = sums / ( G_ni * G_nj )
            sum2 = sum2 / ( G_ni * G_nj )

            if (abs( sum2 - ( sums  * sums ) ) < 1.e-14 * sum2) then
               sum2 = 0.0
            else
               sum2 = sum2 - ( sums  * sums )
            end if

            rmss = sqrt( sum2 )

            position  = (nvar-1)*nsize+1
            positiong = (nvar-1)*G_ni+1

            if (dynpar(busvar,1) == position) then
               write(lun,6001) dynnm(busvar,1),      &
                               dynnm(busvar,2)(1:4), &
                               dyndc(busvar)(1:40)
               busvar = busvar+1
            end if

            write(lun,6002) positiong,maxs,mins,sums,rmss

         end do

         call flush( lun )

      end if         

      return
!--------------------------------------------------------------------

 6000 format(/' At timestep ',I8,', ',A,' bus contains...')
 6001 format(/ 1X , A , 1X , '(' , A , ')' , 1X , A / &
               25X,'maximum',17X,'minimum', &
               20X,   'mean',17X,'std dev')
 6002 format( I8,1p4e24.16)

      end
