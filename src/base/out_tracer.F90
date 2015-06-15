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

!**s/r out_tracer - calculate and output tracer fields
!
      subroutine out_tracer (F_wlnph_ta, Minx,Maxx,Miny,Maxy,F_nk,levset,set)

      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      implicit none
#include <arch_specific.hf>
!
      integer F_nk,Minx,Maxx,Miny,Maxy,levset,set
      real F_wlnph_ta(Minx:Maxx,Miny:Maxy,F_nk)

!author
!     Lee V.                    - rpn May 2004
!
!revision
! v3_20 - Lee V.    - initial MPI version (from bloctr     v3_12)
! v3_30 - Lee V.    - option to clip tracers with Out3_cliph_L
! v4_05 - Lee V.    - adaptation to GMM
! v4_40 - Lee V.    - change in argument call for this routine & prgen

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "out3.cdk"
#include "out.cdk"
#include "level.cdk"
#include "outd.cdk"
#include "tr3d.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "schm.cdk"

      type(vgrid_descriptor) :: vcoord
      integer i,j,k,ii,n,nko,kind,istat
      integer, dimension(:), allocatable::indo
      integer, dimension(:), pointer :: ip1t
      real ,dimension(:), allocatable::prprlvl,rf
      real, dimension(:), pointer :: hybt
      save hybt
      real ,dimension(:,:,:), allocatable:: w4,cible
      real t4(l_minx:l_maxx,l_miny:l_maxy,G_nk+1), &
           t5(l_minx:l_maxx,l_miny:l_maxy,G_nk+1) 
      real, pointer, dimension(:,:,:) :: tr1
      logical :: write_diag_lev,near_sfc_L,outvar_L
!_______________________________________________________________________

!
      if (Level_typ_S(levset) .eq. 'M') then  ! output tracers on model levels

         kind=Level_kind_ip1
!       Setup the indexing for output
         allocate (indo( min(Level_max(levset),Level_thermo) ))
         call out_slev2(Level(1,levset), Level_max(levset), &
                       Level_thermo,indo,nko,near_sfc_L)
         write_diag_lev= near_sfc_L .and. out3_sfcdiag_L

!        Retreieve vertical coordinate description
         if ( .not. associated(hybt) ) then
            nullify(ip1t,hybt)
            istat = vgrid_wb_get('ref-t',vcoord,ip1t)
            deallocate(ip1t); nullify(ip1t)
            if (vgd_get(vcoord,'VCDT - vertical coordinate (t)',hybt) /= VGD_OK) istat = VGD_ERROR
         endif

         do ii=1,Outd_var_max(set)
            outvar_L=.false.
            do n=1,Tr3d_ntr
               if (Outd_var_S(ii,set).eq.Tr3d_name_S(n)) then
                  nullify (tr1)
                  istat = gmm_get('TR/'//trim(Tr3d_name_S(n))//':P',tr1)
                  if (.not.GMM_IS_ERROR(istat)) outvar_L=.true.
                  cycle
               endif
            enddo
            if (outvar_L) then
               if (Out3_cliph_L) then
                  do k=1,G_nk
                     do j=1,l_nj
                     do i=1,l_ni
                        t4(i,j,k) = max ( tr1(i,j,k), 0. )
                     enddo
                     enddo
                  enddo
                  call ecris_fst2(t4,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                             Outd_var_S(ii,set),1.0,0.0,kind,G_nk,indo,nko, &
                             Outd_nbit(ii,set) )
               else
                  call ecris_fst2(tr1,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                              Outd_var_S(ii,set),1.0,0.0,kind,G_nk,indo,nko, &
                              Outd_nbit(ii,set) )
               endif

               if (write_diag_lev)  then
                  t4(:,:,G_nk+1) = tr1(:,:,G_nk)
                  call itf_phy_sfcdiag (t4(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,&
                                            'TR/'//trim(Outd_var_S(ii,set))//':P',istat,.true.)
                  if (istat.eq.0) then
                     if (Out3_cliph_L) t4(:,:,G_nk+1) = max ( t4(:,:,G_nk+1), 0. )
                     call ecris_fst2 ( t4(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy   ,&
                                       hybt(G_nk+2),Outd_var_S(ii,set),1.0,0.0,Level_kind_diag,&
                                        1,1,1,Outd_nbit(ii,set) )
                  endif
               endif
                  
            endif
         enddo
         deallocate(indo)

      else  ! output tracers on pressure levels

         kind=2

!       Setup the indexing for output
         nko=Level_max(levset)
         allocate ( indo(nko), rf(nko) , prprlvl(nko), &
                  w4(l_minx:l_maxx,l_miny:l_maxy,nko), &
                    cible(l_minx:l_maxx,l_miny:l_maxy,nko) )

         do i = 1, nko
            indo(i)=i
            rf(i)= Level(i,levset)
            prprlvl(i) = rf(i) * 100.0
            cible(:,:,i) = log(prprlvl(i))
         enddo

         do ii=1,Outd_var_max(set)
            outvar_L=.false.
            do n=1,Tr3d_ntr
               if (Outd_var_S(ii,set).eq.Tr3d_name_S(n)) then
                  nullify (tr1)
                  istat = gmm_get('TR/'//trim(Tr3d_name_S(n))//':P',tr1)
                  if (.not.GMM_IS_ERROR(istat)) outvar_L=.true.
                  cycle
               endif
            enddo
            if (outvar_L) then
               if (out3_sfcdiag_L) then
                  t5(:,:,G_nk+1) = tr1(:,:,G_nk)
                  call itf_phy_sfcdiag (t5(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,&
                                        'TR/'//trim(Outd_var_S(ii,set))//':P',istat,.true.)
               else
                  istat=-1
               endif
               if (istat.eq.0) then
                  t5(:,:,1:G_nk  ) = tr1(:,:,1:G_nk)
                  call vertint ( w4,cible,nko, t5 ,F_wlnph_ta,G_nk+1       ,&
                                 l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                                 'linear', .false. )
               else
                  call vertint ( w4,cible,nko, tr1,F_wlnph_ta,G_nk         ,&
                                 l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                                 'linear', .false. )                     
               endif

               if (Outd_filtpass(ii,set).gt.0) &
                    call filter2( w4,Outd_filtpass(ii,set), &
                                    Outd_filtcoef(ii,set), &
                            l_minx,l_maxx,l_miny,l_maxy, nko)
               if (Out3_cliph_L) then
                  do k=1,nko
                     do j=1,l_nj
                     do i=1,l_ni
                        w4(i,j,k) = amax1(w4(i,j,k), 0. )
                     enddo
                     enddo
                  enddo
               endif
               call ecris_fst2 ( w4,l_minx,l_maxx,l_miny,l_maxy,rf  , &
                                 Outd_var_S(ii,set),1.0,0.0,kind,nko, &
                                 indo, nko, Outd_nbit(ii,set) )
               
            endif
         enddo
         deallocate(indo,rf,prprlvl,w4,cible)

      endif
! ___________________________________________________________________
!
      return
      end
