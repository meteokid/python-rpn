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

!**   s/r ens_marfield_cg - define a markov chain field
!     

!     
      subroutine ens_marfield_cg(fgem,E_nk)
!
      use phy_itf
      use harmonsphere, only : harmon, harmon_init, harmon_destroy, harmons
      implicit none
#include <arch_specific.hf>

      integer                               :: E_nk
      real,    dimension(l_ni,l_nj,E_nk)    :: fgem
!     
!author   
!     Lubos Spacek - February 2005
!     
!revision
! v4_11 - Spacek L.         - staggered + gmm version
! v_4.1.3 - N. Gagnon.      - Change name of some parameters from NAMELIST and calculation of mean with (min+max)/2
!
!object
!	
!arguments     none
!

#include <rmnlib_basics.hf>
#include "gmm.hf"
#include "ens_gmm_dim.cdk"
#include "ens_gmm_var.cdk"
#include "dcst.cdk"
#include "lun.cdk"
#include "mem.cdk"
#include "lctl.cdk"
#include "cstv.cdk"
#include "glb_ld.cdk"
#include "ptopo.cdk"
#include "hgc.cdk"
#include "ens_param.cdk"
#include "step.cdk"
#include "grd.cdk"
!


      integer p_ni,p_nj,p_offi,p_offj
       integer, external :: read_db_file,write_db_file
       real,    external :: plgndr, ran1, gasdev
!       integer, external :: fnom,fstouv,fstfrm,fclos
!       integer, external :: ezgdef_fmem,ezqkdef,ezsint,ezdefset,ezsetopt

!
! nlat, nlon                 dimension of the Gaussian grid
! mzt                        Nombre d'onde vertical
! idum                       Semence du générateur de nombres aléatoires
!
      integer :: nlat, nlon, ntrunc, nmdim, mzt, ndim_tot, dgid_myp
      integer :: l ,m, mz, mzp, i, j, k, naloc, indx, ier, ii, jj
      integer :: soit,lght,indx_n, gmmstat, istat
      integer :: gdgem,gdgauss, keys(5), nmstrt, nm, n, mtrunc, ltrunc
      real    :: xmn, xmx, xme, std, aa, tau, ar, Ens_mc2d_mean
      real    :: xf,offi,offj
      real xfi(l_ni),yfi(l_nj)
      real*8  :: rad2deg_8
      logical, save :: init_done=.false.
!
! placer les chaines de Markov dans perbus
      character*3 bus
      integer, save :: mrkv,mrk2
!
! pabusper pointer vers le bus permanent busper (busper2)
! paidum   pointer vers l'etat du generateur sauvegarde idum
      type(gmm_metadata) :: meta
      integer, pointer :: paiv,paiy,paiset,pagset,paidum
!
!      variables et champs auxiliaires reels
! dt   Pas de temps du modèle (secondes)
! tau  Temps de décorrélation du champ aléatoire f(i,j) (secondes)
! eps  EXP(-dt/tau)
      real    :: pi, dt, eps, sig2, x, z
      type (harmon) :: harmon_dsc 
      real,    dimension(:),    allocatable :: veco1,xsiv,xcov
      complex, dimension(:,:,:),allocatable :: pnm
      real,    dimension(:,:,:),allocatable :: fgau, fgau_str
      complex, dimension(:,:),  allocatable :: pnm2
      real,    dimension(:,:),  allocatable :: fgau2, fgau2_str
      real,    dimension(:,:,:),pointer     :: fgem2_str, fgem_str, ptr3d
!
! variables et champs auxiliaires complexes
      complex,dimension(:,:),allocatable :: hoco,veco
!
! Initialise les variables de Ens_nml.cdk
!

      xmn = Ens_mc3d_min ; xmx = Ens_mc3d_max
!NG  Mean of the 3D Markov chains now always middle between min and max
      xme = (Ens_mc3d_min+Ens_mc3d_max)/2.

      std = Ens_mc3d_std ; tau = Ens_mc3d_tau   ; aa     = Ens_mc3d_str
      nlon= Ens_mc3d_nlon; nlat= Ens_mc3d_nlat  ; ntrunc = Ens_mc3d_trnh
      nmdim=Ens_mc3d_dim ; mzt = Ens_mc3d_mzt   ; ar     = 6.37122e6
                                              mtrunc = Ens_mc3d_trnl

      dt=real(Cstv_dt_8)
      pi=real(Dcst_pi_8)
      rad2deg_8=180.0d0/Dcst_pi_8
!
!     Look for the spectral coefficients
!
      gmmstat = gmm_get(gmmk_anm_s,anm,meta2d_anm)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'anm'
      gmmstat = gmm_get(gmmk_znm_s,znm,meta2d_znm)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'znm'
      gmmstat = gmm_get(gmmk_dumdum_s,dumdum,meta2d_dum)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'dumdum'
!     
!     Valeurs initiales des composantes principales
!     
!     Assure that busper is to 1 at the begining in inichamp1
!
      INITIALIZE: if (.not.init_done) then
         if (Lun_out.gt.0) then
            write( Lun_out,1000)
            write( Lun_out,1009)xmn,xmx,xme,std,tau,ntrunc,nlon,nlat,2*mzt+1
         endif
         init_done=.true.
      endif INITIALIZE

      paiv  => dumdum( 1,1)
      paiy  => dumdum(33,1)
      paiset=> dumdum(34,1)
      pagset=> dumdum(35,1)
      paidum=> dumdum(36,1)
!     Rstri_rstn_L
      if ( Step_kount .eq. 1 ) then

         paiv  => dumdum( 1,1)
         paiy  => dumdum(33,1)
         paiset=> dumdum(34,1)
         pagset=> dumdum(35,1)
         paidum=> dumdum(36,1)
         dumdum(:,1)=0

         eps=exp(-dt/Ens_mc3d_tau)
         std=Ens_mc3d_std
         paidum=-Ens_mc_seed

         xf=pi/REAL((ntrunc-mtrunc+1)*(ntrunc+mtrunc+1)/2)/REAL(2*mzt+1)
         do mz=-mzt,mzt
           do l=1,nmdim
             anm(l,mz)=CMPLX(std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt(xf),&
               std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt(xf))
           enddo
         enddo
!
         do mz=1,Ens_mc2d_ncha
         mzp=mz+1
         dumdum(:,mzp)=0
         paiv  => dumdum( 1,mzp)
         paiy  => dumdum(33,mzp)
         paiset=> dumdum(34,mzp)
         pagset=> dumdum(35,mzp)
         paidum=> dumdum(36,mzp)
         paidum=-Ens_mc_seed
         xf=pi/REAL((Ens_mc2d_trnh(mz)-Ens_mc2d_trnl(mz)+1)* &
                                (Ens_mc2d_trnh(mz)+Ens_mc2d_trnl(mz)+1)/2)
         eps=exp(-dt/Ens_mc2d_tau(mz))
         std=Ens_mc2d_std(mz)
         do l=1,Ens_dim2(mz)
             znm(l,mz)=CMPLX(std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt(xf),&
                std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt(xf))
           enddo
         enddo
      endif
!     
!     Precalcul exponentials in vertical
!     
      naloc=(2*mzt+1)*E_nk
      allocate(veco1(naloc),xsiv(naloc),xcov(naloc))
      allocate(veco(E_nk,-mzt:mzt))
      allocate(fgau(nlon,nlat,E_nk))
      allocate(pnm(ntrunc+1,ntrunc+1,E_nk))
!     
      indx=0

      do mz=-mzt,mzt
         do k=1,E_nk
            z=real(k-1)*2.*pi/real(E_nk)
            indx=indx+1
            veco1(indx)=mz*z
         enddo
      enddo
!     
      naloc=(2*mzt+1)*E_nk

      call vssincos(xsiv,xcov,veco1,naloc)
!     
      indx=0
      do mz=-mzt,mzt
         do k=1,E_nk
            indx = indx+1
            veco(k,mz)=cmplx(xcov(indx),xsiv(indx))
         enddo
      enddo
!     
      deallocate(veco1,xsiv,xcov)
!     
!     Begin Markov chains
!
      paiv  => dumdum( 1,1)
      paiy  => dumdum(33,1)
      paiset=> dumdum(34,1)
      pagset=> dumdum(35,1)
      paidum=> dumdum(36,1)

      xf=pi/REAL((ntrunc-mtrunc+1)*(ntrunc+mtrunc+1)/2)/REAL(2*mzt+1)
      eps=exp(-dt/Ens_mc3d_tau)
      std=Ens_mc3d_std
      do mz=-mzt,mzt
         do l=1,nmdim
            anm(l,mz)=CMPLX(eps*real (anm(l,mz))+ &
            std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt((1.-eps**2)*xf), &
            eps*aimag(anm(l,mz))+ &
            std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt((1.-eps**2)*xf))
          enddo
      enddo
!
      do mz=1,Ens_mc2d_ncha
         mzp=mz+1
         paiv  => dumdum( 1,mzp)
         paiy  => dumdum(33,mzp)
         paiset=> dumdum(34,mzp)
         pagset=> dumdum(35,mzp)
         paidum=> dumdum(36,mzp)
         xf=pi/REAL((Ens_mc2d_trnh(mz)-Ens_mc2d_trnl(mz)+1)* &
                                (Ens_mc2d_trnh(mz)+Ens_mc2d_trnl(mz)+1)/2)
         eps=exp(-dt/Ens_mc2d_tau(mz))
         std=Ens_mc2d_std(mz)
         do l=1,Ens_dim2(mz)
            znm(l,mz)=CMPLX(eps*real (znm(l,mz))+ &
            std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt((1.-eps**2)*xf), &
            eps*aimag(znm(l,mz))+ &
            std*gasdev(paiv,paiy,paiset,pagset,paidum)*sqrt((1.-eps**2)*xf))
           enddo
      enddo

      pnm(:,:,:)=(0.,0.)
      do mz=-mzt,mzt
         do k=1,E_nk
            l=0
            do j=2,ntrunc+1
            ltrunc=max(mtrunc+1,j)
            do i=ltrunc,ntrunc+1
               l=l+1
               pnm(i,j,k)=pnm(i,j,k)+anm(l,mz)*veco(k,mz)
            enddo
           enddo
         enddo
      enddo

      call harmon_init(harmon_dsc,nlon,nlat,ntrunc,ntrunc,ntrunc)

      do k=1,E_nk
         call harmons(harmon_dsc,fgau(1,1,k),pnm(1,1,k),nlon,nlat,ntrunc,ntrunc,ntrunc)
      enddo
       call harmon_destroy(harmon_dsc)
!
!     
!
!
!*    Interpolation to the processors grids and fill in perbus
!
     p_offi = Ptopo_gindx(1,Ptopo_myproc+1)-1
     p_offj = Ptopo_gindx(3,Ptopo_myproc+1)-1
     do i=1,l_ni
        indx = p_offi + i
        xfi(i) = G_xg_8(indx)*rad2deg_8
     end do
     do i=1,l_nj
        indx = p_offj + i
        yfi(i) = G_yg_8(indx)*rad2deg_8
     end do

     dgid_myp = ezgdef_fmem (l_ni , l_nj , 'Z', 'E', Hgc_ig1ro, &
                Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, xfi , yfi )

      gdgauss = ezqkdef(nlon,nlat,'G', 0,0,0,0,0)
      ier = ezdefset(dgid_myp, gdgauss)
      ier = ezsetopt('INTERP_DEGREE', 'LINEAR')
!      ier = ezsetopt('VERBOSE', 'YES')
      do k=1,E_nk
         ier = ezsint(fgem(1,1,k),fgau(1,1,k))
      enddo
      if(Ens_stat)then
          call stat(fgau,nlon,nlat,E_nk,'MCSP','GAUSS')
          call glbstat2 (fgem,'MCSP','NOSTR', &
           1,l_ni,1,l_nj,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      if(.not.Ens_ptp_conf)then
         deallocate(fgau,pnm,veco)
         return
      endif

!    Check the limits, stretch, and add mean if stretching asked
!    for the physics perturbation
!

      !TODO: should not work on perbus directly, replace by bus_unfold + opr + bus_fold?

      if(aa/=0.and.aa/=1)then
         where (abs(fgau) > 1.)
            fgau=sign(1.,fgau)
         end where

         allocate(fgau_str(nlon,nlat,E_nk),fgem_str(l_ni,l_nj,E_nk+1))
         sig2=1./LOG((aa/(aa-1.))**2)
         fgau_str=fgau*(aa*exp(-fgau**2/2./sig2)+2.-aa)*(xmx-xme)
         if(xme/=0.0) fgau_str=fgau_str+xme
!
         do k=1,E_nk
            ier = ezsint(fgem_str(1,1,k),fgau_str(1,1,k))
         enddo
         fgem_str(:,:,E_nk+1)=0.0
!
         if(Ens_stat)then
            call glbstat2 (fgem_str,'MCSP','STR', &
             1,l_ni,1,l_nj,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         endif

      else
         fgem_str=1.0
      endif
      ptr3d => fgem_str(Grd_lphy_i0:Grd_lphy_in, &
                        Grd_lphy_j0:Grd_lphy_jn, 1:E_nk+1)
      istat = phy_put(ptr3d,'mrkv',F_npath='V',F_bpath='P')
      deallocate(fgau,pnm,veco,fgau_str,fgem_str)

!====================================
      allocate(fgem2_str(l_ni,l_nj,Ens_mc2d_ncha))

      do mz=1,Ens_mc2d_ncha
         allocate(fgau2(Ens_mc2d_nlon(mz),Ens_mc2d_nlat(mz)))
         allocate(fgau2_str(Ens_mc2d_nlon(mz),Ens_mc2d_nlat(mz)))
         allocate(pnm2(Ens_mc2d_trnh(mz)+1,Ens_mc2d_trnh(mz)+1))
         mtrunc=Ens_mc2d_trnl(mz)
         ntrunc=Ens_mc2d_trnh(mz)
            pnm2(:,:)=(0.,0.)
            l=0
            do j=2,ntrunc+1
               ltrunc=max(mtrunc+1,j)
               do i=ltrunc,ntrunc+1
                  l=l+1
                  pnm2(i,j)=znm(l,mz)
               enddo
            enddo

      call harmon_init(harmon_dsc,Ens_mc2d_nlon(mz),Ens_mc2d_nlat(mz),               &
                        Ens_mc2d_trnh(mz),Ens_mc2d_trnh(mz),Ens_mc2d_trnh(mz))

      call harmons(harmon_dsc,fgau2,pnm2,Ens_mc2d_nlon(mz),Ens_mc2d_nlat(mz),        &
                        Ens_mc2d_trnh(mz),Ens_mc2d_trnh(mz),Ens_mc2d_trnh(mz))
      call harmon_destroy(harmon_dsc)

!
!
!*    Interpolation to the processors grids and fill in perbus
!
        gdgauss = ezqkdef(Ens_mc2d_nlon(mz),Ens_mc2d_nlat(mz),'G', 0,0,0,0,0)

        ier = ezdefset(dgid_myp, gdgauss)
        ier = ezsetopt('INTERP_DEGREE', 'LINEAR')
!       ier = ezsetopt('VERBOSE', 'YES')

!         ier = ezsint(fgem,fgau)


!    Check the limits, stretch, and add mean if stretching asked
!    for the physics perturbation
!
           where (abs(fgau2) > 1.)
              fgau2=sign(1.,fgau2)
           end where
      if(Ens_mc2d_str(mz)/=0.0)then
!
!NG  Mean of the 2D Markov chains now always middle between min and max
         Ens_mc2d_mean=(Ens_mc2d_min(mz)+Ens_mc2d_max(mz))/2.
         sig2=1./LOG((Ens_mc2d_str(mz)/(Ens_mc2d_str(mz)-1.))**2)
         fgau2_str=fgau2*(Ens_mc2d_str(mz)*exp(-fgau2**2/2./sig2)+2.-Ens_mc2d_str(mz))* &
        (Ens_mc2d_max(mz)-Ens_mc2d_mean)
         if(Ens_mc2d_mean/=0.0) fgau2_str=fgau2_str+Ens_mc2d_mean
!
            ier = ezsint(fgem2_str(:,:,mz),fgau2_str)
         else
            fgem2_str(:,:,mz)=1.0
         endif
!

        if(Ens_stat)then
          call glbstat2 (fgem2_str(:,:,mz),'MC2D','STR',&
           1,l_ni,1,l_nj,1,1,1,G_ni,1,G_nj,1,1)
!          call glbstat2 (fgau2(Ens_mc2d_nlon(mz),Ens_mc2d_nlat(mz)),'MCGA','NOSTR',&
!           1,Ens_mc2d_nlon(mz),1,Ens_mc2d_nlat(mz),1,1,1,G_ni,1,G_nj,1,1)
!          call glbstat2 (fgau2_str,'MCGA','STR',&
!           1,Ens_mc2d_nlon(mz),1,Ens_mc2d_nlat(mz),1,1,1,&
!           Ens_mc2d_nlon(mz),1,Ens_mc2d_nlat(mz),1,1)
        endif

        deallocate(fgau2,fgau2_str,pnm2)

     enddo
!      allocate ( wk1(iend2(1),iend2(2),iend2(3)) )
!      wk1(:,:,:)= fgem2_str(istart2(1)+1:l_ni-istart2(1),istart2(2)+1:l_nj-istart2(1),:)
!      istat = phy_put(wk1,'mrk2',F_npath='V',F_bpath='P')
!      deallocate ( wk1 )

     ptr3d => fgem2_str(Grd_lphy_i0:Grd_lphy_in, &
                        Grd_lphy_j0:Grd_lphy_jn, 1:Ens_mc2d_ncha)
     istat = phy_put(ptr3d ,'mrk2',F_npath='V',F_bpath='P')
     deallocate(fgem2_str)
!     
!
 1000 format( &
           /,'INITIALIZE SCHEMES CONTROL PARAMETERS (S/R ENS_MARFIELD_CG)', &
           /,'======================================================')
 1008 format(' Physics must be incore!')
 1009 format( &
           /,'MARKOV CHAIN FIELD      MIN    MAX    MEA    STD   TAU  TRUNC ', &
             'NLON/NLAT/NNIV', &
           /,20X,4F7.1,1X,F7.0,I4,1X,3(I4,' '),/)
 6000 format('ens_marfield_cg at gmm_get(',A,')')
      return

contains

      subroutine stat(field,nx,ny,nz,msg1,msg2)
        implicit none
        integer :: nx,ny,nz
        real :: mean, std
        character(len=*) :: msg1,msg2
        real, dimension(nx,ny,nz) :: field
        mean=sum(field)/(nx*ny*nz)
        std=sqrt(sum((field-mean)**2)/(nx*ny*nz-1))
        if (Lun_out.gt.0)write(Lun_out,99)Lctl_step,msg1,mean,std,         &
        minloc(field),minval(field),maxloc(field),maxval(field),msg2
 99   format (i4,a4,' Mean:',e14.7,' Std:',e14.7, &
              ' Min:[(',i3,',',i3,',',i3,')', &
              e14.7,']',' Max:[(',i3,',',i3,',',i3,')', &
              e14.7,']',a6)
      end subroutine stat
      END subroutine ens_marfield_cg

