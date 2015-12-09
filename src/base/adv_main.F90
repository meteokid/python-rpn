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
!
      subroutine adv_main ( F_fnitraj, F_icn             ,&
                         ut0, vt0, zdt0, ut1, vt1, zdt1  ,&
                         orhsu, rhsu, orhsv, rhsv, orhsc ,&
                         rhsc, orhst,  rhst, orhsf, rhsf ,&
                         orhsq, rhsq, orhsw, rhsw, orhsx ,&
                         rhsx, Minx,Maxx,Miny,Maxy, Nk )
      implicit none
#include <arch_specific.hf>

      integer,intent(in) ::  F_fnitraj, F_icn
      integer,intent(in) ::  Minx,Maxx,Miny,Maxy, Nk
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(in) :: ut0, vt0, zdt0 ! winds at time t0
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(in) :: ut1, vt1, zdt1 ! winds at time t1
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(in) :: orhsu, orhsv, orhsc, orhst, &
                                                            orhsf, orhsq, orhsw, orhsx                                              
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(inout) :: rhsu, rhsv, rhsc,  rhst, &
                                                               rhsf, rhsq, rhsw, rhsx
!@objective perform semi-lagrangian advection
!@author RPN-A  Model Infrastructure Group  (based on : adx_main , adx_interp_gmm  )  June 2015

#include "glb_ld.cdk"
#include "grd.cdk"
#include <gmm.hf>    
#include "lam.cdk"
#include "schm.cdk"
#include "lun.cdk"
#include "lctl.cdk"
#include "step.cdk"
#include "adv_pos.cdk"
#include "vth.cdk"
#include "adv_grid.cdk"
#include "adv_tracers.cdk"
      
      logical :: doAdwStat_L
      integer :: i0,j0,in,jn,i0u,inu,j0v,jnv ! advection computational i,j,k domain  (glb_ld.cdk)
      integer :: k0, k0m, k0t, err, jext, ext, i0_e, j0_e, in_e, jn_e, i0u_e, inu_e, j0v_e, jnv_e
      real, dimension(:,:,:), allocatable  :: pxm, pym, pzm ! upstream positions valid at time t1
      real, dimension(:,:,:), allocatable  :: ua,va,wa,wat  ! arrival winds
      real, dimension(:,:,:), allocatable  :: ud,vd,wd      ! de-staggered   departure winds 
      real, dimension(adv_lminx:adv_lmaxx,adv_lminy:adv_lmaxy,l_nk) :: a_ud , a_vd , a_wd
      real, dimension(1,1,1), target :: no_slice
      integer :: num, nm,nu,nv,nt,nmax
      integer, dimension(:),  allocatable :: ii             ! pre-computed index used in the tricubic lagrangian interp loop 
     
!
!     ---------------------------------------------------------------
!      
      if (Lun_debug_L) write (Lun_out,1000)
      doAdwStat_L = .false.
      if (Step_gstat.gt.0) then
         doAdwStat_L = (mod(Lctl_step,Step_gstat) == 0)
      end if
      doAdwStat_L = doAdwStat_L .and. (F_icn == Schm_itcn)
      
      allocate ( pxm(l_ni,l_nj,l_nk), pym(l_ni,l_nj,l_nk), pzm(l_ni,l_nj,l_nk),&
                 ua (l_ni,l_nj,l_nk), va (l_ni,l_nj,l_nk), wa (l_ni,l_nj,l_nk),&
                 wat(l_ni,l_nj,l_nk) )
      allocate ( ud(l_minx:l_maxx,l_miny:l_maxy,l_nk),&
                 vd(l_minx:l_maxx,l_miny:l_maxy,l_nk),&
                 wd(l_minx:l_maxx,l_miny:l_maxy,l_nk) )
      if (.not.associated(pxt)) allocate (pxt(l_ni,l_nj,l_nk), &
                                          pyt(l_ni,l_nj,l_nk), &
                                          pzt(l_ni,l_nj,l_nk), &
                                          pxtn(l_ni,l_nj), &
                                          pytn(l_ni,l_nj), &
                                          pztn(l_ni,l_nj), &
                                          pxmu(l_ni,l_nj,l_nk), &
                                          pymu(l_ni,l_nj,l_nk), &
                                          pzmu(l_ni,l_nj,l_nk), &
                                          pxmv(l_ni,l_nj,l_nk), &
                                          pymv(l_ni,l_nj,l_nk), &
                                          pzmv(l_ni,l_nj,l_nk)  )

      if (.not.associated(pxmu_s).and.Adv_slice_L) allocate (pxmu_s(l_ni,l_nj,l_nk), & 
                                                             pymu_s(l_ni,l_nj,l_nk), &
                                                             pzmu_s(l_ni,l_nj,l_nk), &
                                                             pxmv_s(l_ni,l_nj,l_nk), &
                                                             pymv_s(l_ni,l_nj,l_nk), &
                                                             pzmv_s(l_ni,l_nj,l_nk)  )

      nullify (xth, yth, zth)
      err = gmm_get(gmmk_xth_s , xth)
      err = gmm_get(gmmk_yth_s , yth)
      err = gmm_get(gmmk_zth_s , zth)
      
! Get advection computational i,j,k domain
! ----------------------------------------
      i0 = 1 ; in = l_ni
      j0 = 1 ; jn = l_nj

      jext=1
      if (Grd_yinyang_L) jext=2
      if (l_west)  i0 =        pil_w - jext
      if (l_east)  in = l_ni - pil_e + jext
      if (l_south) j0 =        pil_s - jext
      if (l_north) jn = l_nj - pil_n + jext
      i0u = i0 ;  inu = in
      j0v = j0 ;  jnv = jn

      jext = 2
      if (Grd_yinyang_L) jext = 0
      if (l_west)  i0u= i0 + jext
      if (l_east)  inu= in - jext
      if (l_south) j0v= j0 + jext
      if (l_north) jnv= jn - jext

! Establish scope of extended advection operations
! ------------------------------------------------
      if (Adv_extension_L) then
         call adv_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

         i0u_e = i0_e ;  inu_e = in_e
         j0v_e = j0_e ;  jnv_e = jn_e

         jext = 2
         if (Grd_yinyang_L) jext = 0
         if (l_west)  i0u_e= i0_e + jext
         if (l_east)  inu_e= in_e - jext
         if (l_south) j0v_e= j0_e + jext
         if (l_north) jnv_e= jn_e - jext
      endif

      k0 = Lam_gbpil_t+1
      k0t= k0     
      if(Lam_gbpil_t.gt.0) k0t=k0-1
      k0m=max(k0t-2,1)
    
!
      nm=(in-i0+1)*(jn-j0+1)*(l_nk-k0+1)
      nt=(in-i0+1)*(jn-j0+1)*(l_nk-k0t+1)
      nu=(inu-i0u+1)*(jn-j0+1)*(l_nk-k0+1)
      nv=(in-i0+1)*(jnv-j0v+1)*(l_nk-k0+1)
      nmax=max(nm,nt,nu,nv)
      num=l_ni*l_nj*l_nk

      call timing_start2 (30, 'ADV_TRAJE', 21) ! Compute trajectories

! Process winds in preparation for SL advection: unstagger & interpolate from Thermo to Momentum levels
! -----------------------------------------------------------------------------------------------------
      call adv_prepareWinds ( ud, vd, wd, ua, va, wa, wat    , &
                              ut0, vt0 , zdt0, ut1, vt1, zdt1, &
                              l_minx, l_maxx, l_miny, l_maxy , &
                              l_ni , l_nj , l_nk )

! Extend the grid from model to adection with filled halos
! --------------------------------------------------------
      call adv_extend_grid (a_ud,a_vd, a_wd, ud, vd, wd            , &
                            adv_lminx,adv_lmaxx,adv_lminy,adv_lmaxy, &
                            l_minx,l_maxx,l_miny,l_maxy, l_nk)


! Calculate upstream positions at t1 using angular displacement & trapezoidal rule
! --------------------------------------------------------------------------------
      if (.NOT.Adv_extension_L) then
      call adv_trapeze (F_fnitraj, pxm , pym , pzm        ,&
                        a_ud, a_vd, a_wd, ua, va ,wa , wat,&
                        xth, yth, zth, i0, in, j0, jn, i0u,&
                        inu, j0v, jnv, k0, k0m, k0t       ,&
                        adv_lminx, adv_lmaxx, adv_lminy, adv_lmaxy, l_ni, l_nj, l_nk )
      else
      call adv_trapeze (F_fnitraj, pxm , pym , pzm        ,&
                        a_ud, a_vd, a_wd, ua, va ,wa , wat,&
                        xth, yth, zth, i0_e, in_e, j0_e, jn_e, i0u_e,&
                        inu_e, j0v_e, jnv_e, k0, k0m, k0t ,&
                        adv_lminx, adv_lmaxx, adv_lminy, adv_lmaxy, l_ni, l_nj, l_nk )
      endif
      call timing_stop (30) 
      call timing_start2 (31, 'ADV_INLAG', 21)

      allocate (ii(nmax*4))

! RHS Interpolation: momentum levels  
! ----------------------------------
      call adv_get_indices(ii, pxmu, pymu, pzmu, num, nu,  i0u, inu, j0, jn, k0, l_nk, 'm') 
      call adv_cubic('RHSU_S', rhsu, orhsu, pxmu, pymu, pzmu, &
                      no_slice, no_slice, no_slice, no_slice, no_slice, no_slice,&
                      l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy,&
                      nu, ii, i0u, inu, j0, jn, k0, 'm',0 ,0 )    

      call adv_get_indices(ii, pxmv, pymv, pzmv, num, nv, i0, in, j0v, jnv, k0, l_nk, 'm') 
      call adv_cubic('RHSV_S', rhsv, orhsv, pxmv, pymv, pzmv, & 
                      no_slice, no_slice, no_slice, no_slice, no_slice, no_slice,&
                      l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy,&
                      nv, ii, i0, in, j0v, jnv, k0, 'm',0 ,0 ) 

      call adv_get_indices(ii, pxm, pym, pzm, num, nm, i0, in, j0, jn, k0, l_nk, 'm') 
      call adv_cubic('RHSC_S', rhsc ,orhsc , pxm, pym, pzm,& 
                      no_slice, no_slice, no_slice, no_slice, no_slice, no_slice,&
                      l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy,&
                      nm, ii, i0, in, j0, jn, k0, 'm', 0, 0 )    


! RHS Interpolation: lnk-1 thermo levels + surface
! ------------------------------------------------
      call adv_get_indices(ii, pxt, pyt, pzt, num, nt, i0, in, j0, jn, k0, l_nk, 'x')   
      call adv_cubic('RHSX_S', rhsx, orhsx, pxt, pyt, pzt, & 
                        no_slice, no_slice, no_slice, no_slice, no_slice, no_slice, &
                        l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy, &
                        nt, ii, i0, in, j0, jn, k0t, 'x', 0, 0 )      

      if(.not.Schm_hydro_L) then
         call adv_cubic('RHSQ_S', rhsq ,orhsq , pxt, pyt, pzt, & 
                        no_slice, no_slice, no_slice, no_slice, no_slice, no_slice, &
                        l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy, &
                        nt, ii, i0, in, j0, jn, k0t, 'x', 0, 0 )
      endif

! RHS Interpolation: l_nk thermo levels       
! -------------------------------------
      pxt(:,:,l_nk)=pxtn(:,:)
      pyt(:,:,l_nk)=pytn(:,:)
      pzt(:,:,l_nk)=pztn(:,:) 
     
      call adv_get_indices(ii,  pxt, pyt, pzt, num, nt, i0, in, j0, jn, k0, l_nk, 't')
   
      call adv_cubic('RHST_S', rhst, orhst, pxt, pyt, pzt, & 
                      no_slice, no_slice, no_slice, no_slice, no_slice, no_slice, &
                      l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy, &
                      nt, ii, i0, in, j0, jn, k0t, 't', 0, 0 )

      if(.not.Schm_hydro_L) then
         call adv_cubic('RHSF_S', rhsf, orhsf, pxt, pyt, pzt, & 
                        no_slice, no_slice, no_slice, no_slice, no_slice, no_slice, &
                        l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy, &
                        nt, ii, i0, in, j0, jn, k0t, 't', 0, 0 )  
  
         call adv_cubic('RHSW_S', rhsw ,orhsw , pxt, pyt, pzt, &
                        no_slice, no_slice, no_slice, no_slice, no_slice, no_slice, &
                        l_ni, l_nj, l_nk, l_minx, l_maxx, l_miny, l_maxy, &
                        nt, ii, i0, in, j0, jn, k0t, 't', 0, 0 )
      endif

      call timing_stop (31)  

! Compute Courant numbers (CFL) for stats 
! --------------------------------------- 
      if ( doAdwStat_L ) then 
         call  adv_cfl_lam3 (pxm, pym, pzm, i0,in,j0,jn, l_ni,l_nj,k0,l_nk,'m')
         call  adv_cfl_lam3 (pxt, pyt, pzt, i0,in,j0,jn, l_ni,l_nj,k0,l_nk,'t')                                        
      endif

      deallocate (ii)
      deallocate (pxm,pym,pzm,ua,va,wa,wat,ud,vd,wd)

1000  format(3X,'COMPUTE ADVECTION: (S/R ADV_MAIN)')
!
!     ---------------------------------------------------------------
!     
      return
      end subroutine adv_main
