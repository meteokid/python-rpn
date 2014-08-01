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

!*s/r out_href2 - write horizontal positional parameters
!

!
      subroutine out_href2 (F_arakawa_S)
!(F_xpos,F_ypos)
      implicit none
#include <arch_specific.hf>
!
      character* (*) F_arakawa_S
!      real F_xpos(*), F_ypos(*)
!
!author
!     v.lee - rpn march 2008
!
!revision
! v4_03 - Lee V.            - initial MPI version (from wrhref MC2)
!

#include "glb_ld.cdk"
#include "geomn.cdk"
#include "grd.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "ptopo.cdk"

      integer, external  :: fstinl,fstecr
      integer, parameter :: nlis = 1024
      character*1 familly_uencode_S
      logical flag
      integer i,err,lislon,indx,nrec,n1,n2,n3,nis,njs,niyy,vesion_uencode
      integer liste(nlis), ix1,ix2,ix3,ix4,sindx,ig3
      real xpos(Out_nisg), ypos(Out_njsg),wk
      real, dimension(:), pointer     :: posx,posy
      real, dimension(:), allocatable :: yy
!
!----------------------------------------------------------------------
!
      if (F_arakawa_S =='Mass_point') then
         posx => Geomn_longs
         posy => Geomn_latgs
         ig3  = 1
      endif
      if (F_arakawa_S =='U_point') then
         posx => Geomn_longu
         posy => Geomn_latgs
         ig3  = 2
      endif
      if (F_arakawa_S =='V_point') then
         posx => Geomn_longs
         posy => Geomn_latgv
         ig3  = 3
      endif
      if (F_arakawa_S =='F_point') then
         posx => Geomn_longu
         posy => Geomn_latgv
         ig3  = 4
      endif

      nis = Out_nisg ; njs= Out_njsg
      flag= (Out_blocme.eq.0)
      if (Out3_fullplane_L) then
         nis= Out_gridin - Out_gridi0 + 1
         njs= Out_gridjn - Out_gridj0 + 1
         if ( (Grd_typ_S(1:2)=='GU') .and. (nis.eq.G_ni) ) nis= nis   + 1
         if (Out3_uencode_L) flag= flag .and. (Ptopo_couleur.eq.0)
      else
         flag= flag .and. (Out_nisl.gt.0) .and. (Out_njsl.gt.0)
      endif

      if (flag) then

         if (Out3_uencode_L) then

            vesion_uencode    = 1
            familly_uencode_S = 'F'

            nrec= fstinl (Out_unf,n1,n2,n3,' ',' ',Out_ig1,Out_ig2,ig3,&
                                           ' ','^>',liste,lislon,nlis)
            if ((lislon.lt.1).and.(.not.Out_flipit_L)) then
               niyy=5+2*(10+nis+njs)
               allocate (yy(niyy))

               yy(1 ) = iachar(familly_uencode_S)
               yy(2 ) = vesion_uencode
               yy(3 ) = 2 ! 2 grids (Yin & Yang)
               yy(4 ) = 1 ! the 2 grids have same resolution
               yy(5 ) = 1 ! the 2 grids have same area extension on the sphere

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
               yy(sindx+10    :sindx+9+nis    )= posx(Out_gridi0:Out_gridi0+nis-1)
               yy(sindx+10+nis:sindx+9+nis+njs)= posy(Out_gridj0:Out_gridj0+njs-1)

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
               yy(sindx+10    :sindx+9+nis    )= posx(Out_gridi0:Out_gridi0+nis-1)
               yy(sindx+10+nis:sindx+9+nis+njs)= posy(Out_gridj0:Out_gridj0+njs-1)

               err= fstecr(yy,yy, -32, Out_unf,0,0,0,niyy,1,1       ,&
                           Out_ig1,Out_ig2,ig3,'X','^>','YYG_UE_GEMV4',&
                           familly_uencode_S,vesion_uencode,0,0,0, 5, .true.)
            endif

         else

            nrec= fstinl (Out_unf,n1,n2,n3,' ',' ',Out_ig1,Out_ig2,0, &
                                       ' ','>>',liste,lislon,nlis)
            if ((lislon.lt.1).and.(.not.Out_flipit_L)) then
            if ( Out_stride .le. 1 ) then
               ix1= Ptopo_couleur*4+1
               ix2= Ptopo_couleur*4+2
               ix3= Ptopo_couleur*4+3
               ix4= Ptopo_couleur*4+4
               Out_rgridi0 = max(Out_bloci0,Out_gridi0)
               Out_rgridj0 = max(Out_blocj0,Out_gridj0)
               err=fstecr(posx(Out_gridi0),wk,-32,Out_unf,Out_dateo         ,&
                          0,0, nis,1,1, Out_ig1,Out_ig2,0,'X', '>>'         ,&
                          Out_etik_S,Out_gridtyp_S,Out_ixg(ix1),Out_ixg(ix2),&
                          Out_ixg(ix3), Out_ixg(ix4), 5, .true.)
               err=fstecr(posy(Out_gridj0),wk,-32,Out_unf,Out_dateo         ,&
                          0,0, 1,njs,1,Out_ig1,Out_ig2,0,'X', '^^'          ,&
                          Out_etik_S,Out_gridtyp_S,Out_ixg(ix1),Out_ixg(ix2),&
                          Out_ixg(ix3), Out_ixg(ix4), 5, .true.)
            else
               Out_rgridi0=Out_blocin
               do i=1,Out_nisg
                  indx = Out_gridi0+(i-1)*Out_stride
                  xpos(i) = posx(indx)
                  if (indx.ge.Out_bloci0) &
                  Out_rgridi0= min(Out_rgridi0,max(Out_bloci0,indx))
               end do
               Out_rgridj0=Out_blocjn
               do i=1,Out_njsg
                  indx = Out_gridj0+(i-1)*Out_stride
                  ypos(i) = posy(indx)
                  if (indx.ge.Out_blocj0) &
                  Out_rgridj0= min(Out_rgridj0,max(Out_blocj0,indx))
               end do
               err=fstecr(xpos,wk,-32,Out_unf,Out_dateo,0,0,Out_nisg,1 ,&
                              1, Out_ig1,Out_ig2,0,'X', '>>',Out_etik_S,&
                                  Out_gridtyp_S, Out_ixg(1), Out_ixg(2),&
                                       Out_ixg(3), Out_ixg(4), 5, .true.)
               err=fstecr(ypos,wk,-32,Out_unf,Out_dateo,0,0,1,Out_njsg ,&
                              1, Out_ig1,Out_ig2,0,'X', '^^',Out_etik_S,&
                                  Out_gridtyp_S, Out_ixg(1), Out_ixg(2),&
                                       Out_ixg(3), Out_ixg(4), 5, .true.)
            endif
            endif

         endif
      endif
!
!----------------------------------------------------------------------
!
      return
      end
