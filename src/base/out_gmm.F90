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

!**s/r out_gmm - output GMM fields
!
      subroutine out_gmm2 (levset,set)
      implicit none
#include <arch_specific.hf>

      integer levset,set

!author
!     Lee V.                    - rpn July 2009 (from output VMM)
!
!revision
! v4_80 - Desgagne M.       - initial version
!
! There is one vector for the momentum level : Ver_hyb%m.
!
! There are 1 vector for the thermo levels :
!    Ver_hyb%t : Thermo levels without special levels but top and surface.
!
!                Ver_hyb%m(1:G_nk)                 Ver_hyb%t(1:G_nk+1)
! model top
! ===========1        -                              X
! o o o o o o2        -                              -
! - - - - - -3        X                              -
!
! ===========4        -                              X
!
!    ...             ...                            ...
!
!
! ===========2*k      -                              X
!
! - - - - - -2*k+1    X                              -
!
!    ...             ...                            ...
!
! - - - - - -2*G_nk+1 X                              -
! o o o o o o2*G_nk+2 -                              -
! ===========2*G_nk+3 -                              X
! model surface

#include "gmm.hf"
#include "glb_ld.cdk"
#include "out3.cdk"
#include "grid.cdk"
#include "outd.cdk"
#include "lun.cdk"
#include "ver.cdk"
#include "level.cdk"
#include "schm.cdk"

      integer, external :: gmm_nkeys
      type(gmm_metadata) :: tmp_meta
      character(len=GMM_MAXNAMELENGTH), dimension(:), pointer :: keylist
      character(len=2) class_var(100,3)
      logical periodx_L,write_diag_lev
      integer nkeys,nko,i,ii,gridset,istat,id,cid
      integer, dimension(:), allocatable::indo
      integer, parameter :: numvars = 19
      real, pointer, dimension(:,:,:) :: tr3
      real, pointer, dimension(:,:  ) :: tr2
      real, pointer, dimension(:    ) :: level_type
      real, pointer, dimension(:    ), save :: hybt_w=>null()
!_______________________________________________________________________
!
      if ( Level_typ_S(levset).eq.'P') return

      if ( .not. associated (hybt_w) ) then
         allocate(hybt_w(G_nk))
         hybt_w(1:G_nk)= Ver_hyb%t(1:G_nk)
         if (.not. Schm_lift_ltl_L) hybt_w(G_nk)=1.
      endif

      nkeys= gmm_nkeys()
      allocate (keylist(nkeys))
      nkeys= gmm_keys(keylist)

      periodx_L = .false.
      if (.not.G_lam .and. (Grid_x1(Outd_grid(set))- &
            Grid_x0(Outd_grid(set))+1).eq. G_ni ) periodx_L=.true.

      class_var( 1,1) = 'UT' ; class_var( 1,2) = 'UU' ; class_var( 1,3) = 'MM'
      class_var( 2,1) = 'VT' ; class_var( 2,2) = 'VV' ; class_var( 2,3) = 'MM'
      class_var( 3,1) = 'QT' ; class_var( 3,2) = 'QQ' ; class_var( 3,3) = 'MQ'
      class_var( 4,1) = 'TT' ; class_var( 4,2) = 'QQ' ; class_var( 4,3) = 'TT'
      class_var( 5,1) = 'WT' ; class_var( 5,2) = 'QQ' ; class_var( 5,3) = 'TW'
      class_var( 6,1) = 'ZD' ; class_var( 6,2) = 'QQ' ; class_var( 6,3) = 'TT'
      class_var( 7,1) = 'TR' ; class_var( 7,2) = 'QQ' ; class_var( 7,3) = 'TT'
      class_var( 8,1) = 'ST' ; class_var( 8,2) = 'QQ' ; class_var( 8,3) = 'SF'
      class_var( 9,1) = 'MC' ; class_var( 9,2) = 'QQ' ; class_var( 9,3) = 'TT'
      class_var(10,1) = 'DI' ; class_var(10,2) = 'UU' ; class_var(10,3) = 'MM'
      class_var(11,1) = 'UG' ; class_var(11,2) = 'UU' ; class_var(11,3) = 'MM'
      class_var(12,1) = 'VG' ; class_var(12,2) = 'VV' ; class_var(12,3) = 'MM'
      class_var(13,1) = 'EN' ; class_var(13,2) = 'QQ' ; class_var(13,3) = 'TT'
      class_var(14,1) = 'FI' ; class_var(14,2) = 'QQ' ; class_var(14,3) = 'SF'
      class_var(15,1) = 'UD' ; class_var(15,2) = 'QQ' ; class_var(15,3) = 'MM'
      class_var(16,1) = 'VD' ; class_var(16,2) = 'QQ' ; class_var(16,3) = 'MM'
      class_var(17,1) = 'TD' ; class_var(17,2) = 'QQ' ; class_var(17,3) = 'TT'
      class_var(18,1) = 'UR' ; class_var(18,2) = 'UU' ; class_var(18,3) = 'MM'
      class_var(19,1) = 'VR' ; class_var(19,2) = 'VV' ; class_var(19,3) = 'MM'
!     Setup the indexing for output
      allocate (indo   ( min(Level_max(levset),G_nk) ))
      call out_slev2 ( Level(1,levset), Level_max(levset),G_nk,indo,nko,write_diag_lev)

      do ii=1,Outd_var_max(set)
      do  i=1,nkeys

         if (Outd_varnm_S(ii,set)(1:4).eq.keylist(i)(1:4)) then
            gridset = Outd_grid(set)
            id = -1
            do cid=1,numvars
               if (keylist(i)(1:2) == class_var(cid,1)) id=cid
            end do
            if (id.lt.0) then
               if (Lun_out.gt.0) write(Lun_out,1001) trim(keylist(i))
               cycle
            endif
            level_type => Ver_hyb%t
            if (class_var(id,3) == 'MM') level_type => Ver_hyb%m
            if (class_var(id,3) == 'MQ') level_type => Ver_hyb%m(2:G_nk+1)
            if (class_var(id,3) == 'TW') level_type => hybt_w

            select case (class_var(id,2))
            case('UU')
               call out_href3 ( 'U_point', &
                    Grid_x0 (gridset), Grid_x1 (gridset), 1, &
                    Grid_y0 (gridset), Grid_y1 (gridset), 1 )
            case('VV')
               call out_href3 ( 'V_point', &
                    Grid_x0 (gridset), Grid_x1 (gridset), 1, &
                    Grid_y0 (gridset), Grid_y1 (gridset), 1 )
            case default
               call out_href3 ( 'Mass_point', &
                    Grid_x0 (gridset), Grid_x1 (gridset), 1, &
                    Grid_y0 (gridset), Grid_y1 (gridset), 1 )
            end select

            nullify(tr2,tr3)
            istat = gmm_getmeta(keylist(i),tmp_meta)
            if (tmp_meta%l(3)%high.le.1) then
               istat = gmm_get(trim(keylist(i)),tr2,tmp_meta)
               call out_fstecr3 (tr2, tmp_meta%l(1)%low,tmp_meta%l(1)%high,&
                                      tmp_meta%l(2)%low,tmp_meta%l(2)%high,&
                                       0,keylist(i),Outd_convmult(ii,set) ,&
                                       Outd_convadd(ii,set),Level_kind_ip1,&
                                       -1,1,1,1, Outd_nbit(ii,set),.false. )
            else
               istat = gmm_get(trim(keylist(i)),tr3,tmp_meta)
               call out_fstecr3 (tr3, tmp_meta%l(1)%low,tmp_meta%l(1)%high,&
                                      tmp_meta%l(2)%low,tmp_meta%l(2)%high,&
                               level_type,keylist(i),Outd_convmult(ii,set),&
                                      Outd_convadd(ii,set),Level_kind_ip1 ,&
                               -1,G_nk,indo,nko, Outd_nbit(ii,set),.false. )
            endif

            goto 800
         endif

      end do
 800  end do

      deallocate(indo,keylist)

 1001 format(/' ===> In out_gmm: table class_var is incomplete for variable: ',a/)

! ___________________________________________________________________
      return
      end
