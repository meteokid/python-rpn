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

!**s/r read_3df - Reads self-nested 3DF pilot files where
!
      subroutine read_3df3 (unf,uun,vvn,zdn,ttn,ssqn,meqn,qqn,wwn,trn  ,&
                            trname_a,F_zd_L, F_w_L, F_q_L, F_vt_L, ntra,&
                            nga,nka_t,nka_m,diag_lvl,ofi,ofj,err)
      implicit none
#include <arch_specific.hf>

      logical F_zd_L, F_w_L, F_q_L, F_vt_L
      integer unf,ntra,nga,nka_t,nka_m,diag_lvl,ofi,ofj,err
      character*4 trname_a(ntra)
      real  uun (nga,nka_m+diag_lvl  ), vvn (nga,nka_m+diag_lvl)     , &
            ttn (nga,nka_t+diag_lvl  ), trn (nga,nka_t+diag_lvl,ntra), &
            zdn (nga,nka_t), wwn (nga,nka_t), qqn (nga,nka_m)        , &
            meqn(nga,nka_t), ssqn(nga,1)

!author
!     M. Desgagne  April 2006 (MC2 read_3df)
!
!revision
! v3_30 - Lee V.             - initial version for GEMDM
! v3_30 - McTaggart-Cowan R. - implement variable orography
! v4_05 - Plante/McTaggart   - read all available tracers
! v4_05 - McTaggart-Cowan R. - initialize trname_a @@NOT@@
! v4_60 - Lee V.             - add key for whether VT is found
!
#include "ifd.cdk"
#include "tr3d.cdk"
#include "lun.cdk"

      character*4 nomvar,tracers(ntra)
      logical, save :: done=.false.
      logical s_L,gz_L,tt_L,uu_L,vv_L,found,found_diag
      integer i,j,k,ntr,n,cumerr,ni1,nj1,nk1,nbits,lindex,indx_hu
      real, dimension(nga,nka_t+diag_lvl) :: tracer_not_needed
      real, dimension(:,:), allocatable :: tr1
!
!-----------------------------------------------------------------------
!
      if (ntra.le.0) then
         err= -1
         if (Lun_out.gt.0) write(Lun_out,*)'No tracers are available including HU'
         return
      endif

      ntr   = 0
      s_L   = .false.
      gz_L  = .false.
      tt_L  = .false.
      uu_L  = .false.
      vv_L  = .false.
      F_zd_L= .false.
      F_w_L = .false.
      F_q_L = .false.
      F_vt_L = .false.
!     Initialize tracer names
      tracers= '!@@NOT@@'
      found_diag= .false.

      cumerr= 0
10    read (unf,end=66,err=44) nomvar,ni1,nj1,nk1,nbits
      if ((.not.done) .and. (Lun_out.gt.0)) &
           write(Lun_out,1001) nomvar,ni1,nj1,nk1,nbits
      if (nomvar.eq.'S   ') then
         s_L=.true.
         call filmup (nomvar,ssqn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      1,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10          
      endif
      if (nomvar.eq.'GZ  ') then
         gz_L=.true.
         call filmup (nomvar,meqn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nk1,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'TT  ') then
         tt_L=.true.
         call filmup (nomvar,ttn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_t,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'VT  ') then
         tt_L=.true.
         F_vt_L= .true.
         call filmup (nomvar,ttn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_t,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'UU  ') then
         uu_L=.true.
         call filmup (nomvar,uun,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_m,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'VV  ') then
         vv_L=.true.
         call filmup (nomvar,vvn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_m,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'W   ') then
         F_w_L=.true.
         call filmup (nomvar,wwn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_t,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'ZD  ') then
         F_zd_L=.true.
         call filmup (nomvar,zdn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_t,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'Q   ') then
         F_q_L=.true.
         call filmup (nomvar,qqn,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      nka_m,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      if (nomvar.eq.'DIAG') then
         found_diag= .true.
         allocate(tr1(nga,3+ntra))
         call filmup (nomvar,tr1,ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                      3+ntra,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
         cumerr= cumerr + err
         goto 10
      endif
      ! check for required tracer
      found = .false.
      lindex= ntra
      do n=1,Tr3d_ntr
         if (trim(nomvar).eq.trim(Tr3d_name_S(n)))then
            ntr= ntr+1
            tracers(ntr)= nomvar
            lindex= ntr
            found = .true.
            if (nomvar == 'HU') indx_hu=ntr
         endif
      enddo
      if (.not.found) print*, '     Variable ',nomvar,' is skipped'
      call filmup (nomvar,trn(1,1,lindex)             , &
                   ifd_niad,ifd_niaf,ifd_njad,ifd_njaf, &
                   nka_t,unf,ofi,ofj,err,ni1,nj1,nk1,nbits )
      cumerr= cumerr + err
      goto 10

66    if ((.not.done) .and. (Lun_out.gt.0)) &
           write(Lun_out,*) 'end of file reached'

      done= .true.
      if (cumerr.gt.0) then
         err= -1
         if (Lun_out.gt.0) &
         write(Lun_out,*) 'Problem reading data from 3df file in subroutine filmup'
         return
      endif

      if (.not.(s_L.and.tt_L.and.uu_L.and.vv_L.and.gz_L)) then
         if (Lun_out.gt.0) then
            write(Lun_out,*) 'ESSENTIAL variables are missing in 3df file:'
            write(Lun_out,*) 'S =',s_L
            write(Lun_out,*) 'TT=',tt_L
            write(Lun_out,*) 'UU=',uu_L
            write(Lun_out,*) 'VV=',vv_L
            write(Lun_out,*) 'GZ=',GZ_L
         endif
         err= -1
         return
      endif

      if (found_diag) then
         ttn(:,nka_t+diag_lvl) = tr1(:,1)
         uun(:,nka_m+diag_lvl) = tr1(:,2)
         vvn(:,nka_m+diag_lvl) = tr1(:,3)
           do 20 n=1,ntr
            do k=1,ntra
               if (trim(trname_a(k)).eq.trim(tracers(n)))then
                  trn(:,nka_t+diag_lvl,n)= tr1(:,3+k)
                  goto 20
               endif
            end do
 20      continue
         deallocate (tr1)
      endif

      trname_a = tracers

      err= 0

      return

 44   err= -1
      if (Lun_out.gt.0) &
      write(Lun_out,*) 'Problem reading unit: ',unf,' in subroutine read_3df'

 1001 format (3X,'READ: nomvar=',a4,' nia=',i5,' nja=',i5,' nka=',i4,' nbits=',i4)
!
!-----------------------------------------------------------------------
!
      return
      end

