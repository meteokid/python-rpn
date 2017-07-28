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

!**s/r out_tracer - output tracer

      subroutine out_tracer (levset, set)
      use dynkernel_options
      use vertical_interpolation, only: vertint2
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      use phy_itf, only: phy_get
      use gmm_pw
      use grid_options
      use gem_options
      use glb_ld
      use lun
      use tr3d
      use out_mod
      use out3
      use levels
      use outp
      use outd
      use ver
      use type_mod
      use gmm_itf_mod
      implicit none
#include <arch_specific.hf>

      integer levset,set

      type(vgrid_descriptor) :: vcoord
      character(len=512) :: fullname
      integer i,j,k,ii,n,nko,knd,istat,indxtr
      integer, dimension(:), allocatable::indo
      integer, dimension(:), pointer :: ip1t
      real ,dimension(:), allocatable::prprlvl,rf
      real, dimension(:), pointer :: hybt
      save hybt
      real,dimension(:,:,:), allocatable:: w4,cible
      real,dimension(l_minx:l_maxx,l_miny:l_maxy,G_nk+1), target :: t4 ,t5
      real, dimension(:,:  ), pointer :: qdiag
      real, dimension(:,:,:), pointer :: tr1,ptr3d
      logical :: write_diag_lev,near_sfc_L,outvar_L
!
!----------------------------------------------------------------------
!
      if (Level_typ_S(levset) .eq. 'M') then  ! output tracers on model levels

         knd=Level_kind_ip1
!        Setup the indexing for output
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
               if (Outd_var_S(ii,set).eq.trim(Tr3d_name_S(n))) then
                  nullify (tr1)
                  if (trim(Dynamics_Kernel_S) == 'DYNAMICS_EXPO_H') then
                     fullname= 'TR/'//trim(Tr3d_name_S(n))//':M'
                  else
                     fullname= 'TR/'//trim(Tr3d_name_S(n))//':P'
                  end if
                  indxtr=n
                  istat = gmm_get(fullname,tr1)
                  if (.not.GMM_IS_ERROR(istat)) outvar_L=.true.
                  goto 55
               endif
            enddo
 55         if (outvar_L) then
               if (Out3_cliph_L) then
                  do k=1,G_nk
                     do j=1,l_nj
                     do i=1,l_ni
                        t4(i,j,k) = max ( tr1(i,j,k), 0. )
                     enddo
                     enddo
                  enddo
                  call out_fstecr3(t4,l_minx,l_maxx,l_miny,l_maxy,hybt,&
                       Outd_var_S(ii,set),Outd_convmult(ii,set)       ,&
                       Outd_convadd(ii,set),knd,-1,G_nk,indo,nko     ,&
                       Outd_nbit(ii,set),.false. )
               else
                  call out_fstecr3 ( tr1,l_minx,l_maxx,l_miny,l_maxy,&
                      hybt, Outd_var_S(ii,set),Outd_convmult(ii,set),&
                      Outd_convadd(ii,set),knd,-1,G_nk,indo,nko    ,&
                      Outd_nbit(ii,set),.false. )
               endif

               if (write_diag_lev)  then
                  if (trim(Tr3d_name_S(indxtr))=='HU') then
                     istat = gmm_get(gmmk_diag_hu_s,qdiag)
                     t4(:,:,G_nk+1) = qdiag
                  else
                     t4(:,:,G_nk+1) = tr1(:,:,G_nk)
                     ptr3d => t4(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,G_nk+1:G_nk+1)
                     istat = phy_get (ptr3d, trim(fullname), F_npath='VO', F_bpath='D',&
                                      F_start=(/-1,-1,l_nk+1/), F_end=(/-1,-1,l_nk+1/),&
                                      F_quiet=.true.)
                  endif
                  if (istat.eq.0) then
                     if (Out3_cliph_L) t4(:,:,G_nk+1)= &
                                 max ( t4(:,:,G_nk+1), 0. )
                     call out_fstecr3 ( t4(l_minx,l_miny,G_nk+1)      ,&
                            l_minx,l_maxx,l_miny,l_maxy               ,&
                            hybt(G_nk+2),Outd_var_S(ii,set)           ,&
                            Outd_convmult(ii,set),Outd_convadd(ii,set),&
                            Level_kind_diag,-1,1,1,1,Outd_nbit(ii,set),&
                            .false. )
                  endif
               endif

            endif
         enddo
         deallocate(indo)

      else  ! output tracers on pressure levels

         knd=2

!        Setup the indexing for output
         nko=Level_max(levset)
         allocate ( indo(nko), rf(nko) , prprlvl(nko), &
                  w4(l_minx:l_maxx,l_miny:l_maxy,nko), &
               cible(l_minx:l_maxx,l_miny:l_maxy,nko))

         istat= gmm_get(gmmk_pw_log_pt_s  , pw_log_pt)

         do i = 1, nko
            indo(i)=i
            rf(i)= Level(i,levset)
            prprlvl(i) = rf(i) * 100.0
            cible(:,:,i) = log(prprlvl(i))
         enddo

         do ii=1,Outd_var_max(set)

            outvar_L=.false.
            do n=1,Tr3d_ntr
               if (Outd_var_S(ii,set).eq.trim(Tr3d_name_S(n))) then
                nullify (tr1)
                if (trim(Dynamics_Kernel_S) == 'DYNAMICS_EXPO_H') then
                     fullname= 'TR/'//trim(Tr3d_name_S(n))//':M'
                  else
                     fullname= 'TR/'//trim(Tr3d_name_S(n))//':P'
                  end if
                  indxtr=n
                  istat = gmm_get(fullname,tr1)
                  if (.not.GMM_IS_ERROR(istat)) outvar_L=.true.
                  goto 66
               endif
            enddo

 66         if (outvar_L) then
               if (out3_sfcdiag_L) then
                  if (trim(Tr3d_name_S(indxtr))=='HU') then
                     istat = gmm_get(gmmk_diag_hu_s,qdiag)
                     t5(:,:,G_nk+1) = qdiag
                  else
                     t5(:,:,G_nk+1) = tr1(:,:,G_nk)
                     ptr3d => t5(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,G_nk+1:G_nk+1)
                     istat = phy_get (ptr3d, trim(fullname), F_npath='VO', F_bpath='D',&
                                      F_start=(/-1,-1,l_nk+1/), F_end=(/-1,-1,l_nk+1/),&
                                      F_quiet=.true.)
                  endif
               else
                  istat=-1
               endif
               if (istat.eq.0) then
                  t5(:,:,1:G_nk  ) = tr1(:,:,1:G_nk)
                  call vertint2 ( w4,cible,nko, t5 ,pw_log_pt,G_nk+1,&
                                  l_minx,l_maxx,l_miny,l_maxy       ,&
                          1,l_ni,1,l_nj, inttype=Out3_vinterp_type_S )
               else
                  call vertint2 ( w4,cible,nko, tr1,pw_log_pt,G_nk,&
                                  l_minx,l_maxx,l_miny,l_maxy     ,&
                          1,l_ni,1,l_nj, inttype=Out3_vinterp_type_S )
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

               call out_fstecr3 ( w4,l_minx,l_maxx,l_miny,l_maxy,rf  , &
                             Outd_var_S(ii,set),Outd_convmult(ii,set), &
                                        Outd_convadd(ii,set),knd,-1 , &
                                        nko, indo, nko               , &
                                        Outd_nbit(ii,set) , .false. )

            endif

         enddo

         deallocate(indo,rf,prprlvl,w4,cible)

      endif
!
!----------------------------------------------------------------------
!
      return
      end
