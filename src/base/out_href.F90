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

!*s/r out_href - write horizontal positional parameters

      subroutine out_href3 (F_arakawa_S ,F_x0, F_x1, F_stridex,&
                                         F_y0, F_y1, F_stridey )
      implicit none
#include <arch_specific.hf>

      character* (*), intent(in ) :: F_arakawa_S
      integer,        intent(in ) :: F_x0,F_x1,F_stridex,&
                                     F_y0,F_y1,F_stridey
#include "glb_ld.cdk"
#include "geomn.cdk"
#include "grd.cdk"
#include "hgc.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "lun.cdk"
#include "ptopo.cdk"
#include <rmnlib_basics.hf>

      integer, external :: out_samegrd
      character*1 familly_uencode_S
      logical old_grid_L
      integer i,err,ni,nis,njs,niyy,indx,ix1,ix2,ix3,ix4, &
              sindx,i0,in,j0,jn,vesion_uencode
      real wk
      real, dimension(:), pointer     :: posx,posy
      real, dimension(:), allocatable :: yy
!
!----------------------------------------------------------------------
!
      call out_fstecr2 ( wk,wk,wk,wk,wk,wk,wk,wk,wk,wk,wk,wk,&
                         wk,wk, .true. )

! to be completed if at all usefull
!      old_grid_L= out_samegrd ( F_arakawa_S ,F_x0, F_x1, F_stridex,&
!                                F_y0, F_y1, F_stridey, .false. )

      i0 = max( 1   , F_x0)
      in = min( G_ni, F_x1)
      j0 = max( 1   , F_y0)
      jn = min( G_nj, F_y1)

      if (F_arakawa_S =='Mass_point') then
         posx => Geomn_longs
         posy => Geomn_latgs
         Out_ig3  = 1
      endif
      if (F_arakawa_S =='U_point') then
         posx => Geomn_longu
         posy => Geomn_latgs
         Out_ig3  = 2
         if (G_lam) in = min( G_ni-1, F_x1)
      endif
      if (F_arakawa_S =='V_point') then
         posx => Geomn_longs
         posy => Geomn_latgv
         Out_ig3  = 3
         jn = min( G_nj-1, F_y1)
      endif
      if (F_arakawa_S =='F_point') then
         posx => Geomn_longu
         posy => Geomn_latgv
         Out_ig3  = 4
         if (G_lam) in = min( G_ni-1, F_x1)
         jn = min( G_nj-1, F_y1)
      endif

      Out_ig4 = 0

      call set_igs2 ( Out_ig1, Out_ig2, posx, posy, G_ni,G_nj,&
                      Hgc_ig1ro,Hgc_ig2ro,Hgc_ig3ro,Hgc_ig4ro,&
                      i0,in,F_stridex, j0,jn,F_stridey )

      Out_stride= 1 ! can only be one for now
      Out_gridi0= i0
      Out_gridin= in
      Out_gridj0= j0
      Out_gridjn= jn

!      if (old_grid_L) return

      nis = in - i0 + 1  ;  njs = jn - j0 + 1
      if ( (nis .le. 0) .or. (njs .le. 0) ) then
         if (Lun_out.gt.0) write(Lun_out,9000)
         return
      endif

      nis = (in - i0) / Out_stride + 1
      njs = (jn - j0) / Out_stride + 1

      if ((Out3_iome .ge. 0) .and. (Ptopo_couleur.eq.0)) then

         if ( (Grd_yinyang_L) .and. (.not.Out_reduc_l) ) then

            vesion_uencode    = 1
            familly_uencode_S = 'F'

            niyy=5+2*(10+nis+njs)
            allocate (yy(niyy))

            yy(1 ) = iachar(familly_uencode_S)
            yy(2 ) = vesion_uencode
            yy(3 ) = 2          ! 2 grids (Yin & Yang)
            yy(4 ) = 1          ! the 2 grids have same resolution
            yy(5 ) = 1          ! the 2 grids have same area extension

!YIN
            sindx  = 6
            yy(sindx  ) = nis
            yy(sindx+1) = njs
            yy(sindx+2) = posx(Out_gridi0)
            yy(sindx+3) = posx(Out_gridi0+nis-1)
            yy(sindx+4) = posy(Out_gridj0)
            yy(sindx+5) = posy(Out_gridj0+njs-1)
            yy(sindx+6) = Out_rot(1)
            yy(sindx+7) = Out_rot(2)
            yy(sindx+8) = Out_rot(3)
            yy(sindx+9) = Out_rot(4)
            yy(sindx+10    :sindx+9+nis    )= &
            posx(Out_gridi0:Out_gridi0+nis-1)
            yy(sindx+10+nis:sindx+9+nis+njs)= &
            posy(Out_gridj0:Out_gridj0+njs-1)

!YAN
            sindx  = sindx+10+nis+njs
            yy(sindx  ) = nis
            yy(sindx+1) = njs
            yy(sindx+2) = posx(Out_gridi0)
            yy(sindx+3) = posx(Out_gridi0+nis-1)
            yy(sindx+4) = posy(Out_gridj0)
            yy(sindx+5) = posy(Out_gridj0+njs-1)
            yy(sindx+6) = Out_rot(5)
            yy(sindx+7) = Out_rot(6)
            yy(sindx+8) = Out_rot(7)
            yy(sindx+9) = Out_rot(8)
            yy(sindx+10    :sindx+9+nis    )= &
            posx(Out_gridi0:Out_gridi0+nis-1)
            yy(sindx+10+nis:sindx+9+nis+njs)= &
            posy(Out_gridj0:Out_gridj0+njs-1)

            err= fstecr(yy,yy, -32, Out_unf,Out_dateo,0,0,niyy,1,1  ,&
                        Out_ig1,Out_ig2,Out_ig3,'X','^>',Out_etik_S ,&
                        familly_uencode_S,vesion_uencode,0,0,0      ,&
                        5, .true.)
            deallocate (yy, STAT = err)

         else

            if ( Out_stride .le. 1 ) then
               ni = nis
               if ( (Grd_typ_S(1:2) == 'GU') .and. (nis.eq.G_ni) ) &
               ni = G_ni+1
               ix1= Ptopo_couleur*4+1
               ix2= Ptopo_couleur*4+2
               ix3= Ptopo_couleur*4+3
               ix4= Ptopo_couleur*4+4
               err=fstecr(posx(Out_gridi0),wk,-32,Out_unf,Out_dateo   ,&
                    0,0, ni,1,1, Out_ig1,Out_ig2,Out_ig3,'X', '>>'    ,&
                    Out_etik_S,Out_gridtyp_S,Out_ixg(ix1),Out_ixg(ix2),&
                    Out_ixg(ix3), Out_ixg(ix4), 5, .true.)
               err=fstecr(posy(Out_gridj0),wk,-32,Out_unf,Out_dateo   ,&
                    0,0, 1,njs,1,Out_ig1,Out_ig2,Out_ig3,'X', '^^'    ,&
                    Out_etik_S,Out_gridtyp_S,Out_ixg(ix1),Out_ixg(ix2),&
                    Out_ixg(ix3), Out_ixg(ix4), 5, .true.)
            endif

         endif
      endif

 9000 format(/,'OUT_HREF - no grid to output'/)
!
!----------------------------------------------------------------------
!
      return
      end

      logical function out_samegrd ( F_arakawa_S ,F_x0, F_x1, F_stridex,&
                                     F_y0, F_y1, F_stridey, F_init_L )
      implicit none
#include <arch_specific.hf>

      character* (*), intent(in ) :: F_arakawa_S
      logical,        intent(in ) :: F_init_L
      integer,        intent(in ) :: F_x0,F_x1,F_stridex,&
                                     F_y0,F_y1,F_stridey

      integer, external  :: f_crc32
      integer, parameter :: max_grid= 50
      character*20, save :: id_ara_S(max_grid)
      integer crc
      integer     , save :: id_crc(max_grid), cnt
      real, dimension(6) :: identity_vec
!
!----------------------------------------------------------------------
!
      if ( F_init_L ) then
         id_ara_S= '' ; id_crc= 0 ; cnt= 0 ; out_samegrd= .false.
      else
         identity_vec(1)= F_x0
         identity_vec(2)= F_x1
         identity_vec(3)= F_stridex
         identity_vec(4)= F_y0
         identity_vec(5)= F_y1
         identity_vec(6)= F_stridey

         crc= f_crc32 (0., identity_vec(1:6), 24)
         
         if ( any (id_ara_S == trim(F_arakawa_S)) .and. &
              any (id_crc == crc) ) then
            out_samegrd= .true.
         else
            cnt=cnt+1
            id_ara_S(cnt) = F_arakawa_S
            id_crc  (cnt) = crc
            out_samegrd   = .false.
         endif
      endif
!
!----------------------------------------------------------------------
!
      return
      end
