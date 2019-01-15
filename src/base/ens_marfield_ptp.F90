!---------------------------------- LICENCE BEGIN -------------------------------
! GEM - Library of kernel routines for the GEM numerical atmospheric model
! Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!                          Environnement Canada
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

!**   s/r ens_marfield_ptp - define a markov chain field for PTP
!     

      subroutine ens_marfield_ptp
!
      use phy_itf, only : phy_put
 
      implicit none
#include <arch_specific.hf>

      integer                               :: E_nk
!     
!author: Rabah Aider R.P.N.-A  (from ens_marfield_cg.ftn90)
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
       real,    external ::  gasdev
       real*8 :: polg

!
! nlat, nlon                 dimension of the Gaussian grid
! idum                       Semence du générateur de nombres aléatoires
!
      integer :: nlat, nlon, dgid_myp
      integer :: l ,m, nc, i, j, k, indx, ier
      integer lmin,lmax 
      integer ::  gmmstat, istat
      integer :: gdgem,gdgauss, keys(5), n
      real    ::  fstd, fstr, tau,  Ens_ptp_mean
      real    :: sumsp , fact,fact2
      real    :: xf,offi,offj
      real xfi(l_ni),yfi(l_nj)
      real*8  :: rad2deg_8,  pri_8
      logical, save :: init_done=.false.
!
! placer les chaines de Markov dans perbus
      integer, save :: mrk2
!
! paidum   pointer vers l'etat du generateur sauvegarde idum
      type(gmm_metadata) :: meta
      integer, pointer :: paiv,paiy,paiset,pagset,paidum
!
!      variables et champs auxiliaires reels
! dt   Pas de temps du modèle (secondes)
! tau  Temps de décorrélation du champ aléatoire f(i,j) (secondes)
! eps  EXP(-dt/tau/2.146)
      real*8    :: pi, dt, eps  
      real*8    :: fmax, fmin , fmean 
      real*8,    dimension(:), allocatable :: pspectrum , fact1, fact1n
      real*8,    dimension(:), allocatable  :: wrk1
      real*8,    dimension(:,:,:),allocatable :: wrk2, p,cc
      real  ,    dimension(:,:),allocatable :: f, f_str
      real,    dimension(:,:,:),pointer   ::  ptr3d, fgem_str
      integer, dimension(:,:) , allocatable :: sig
       
!
! Initialise les variables de Ens_nml.cdk
!

      dt=real(Cstv_dt_8)
      pi=real(Dcst_pi_8)
      rad2deg_8=180.0d0/Dcst_pi_8
!
!     Look for the spectral coefficients
!
      gmmstat = gmm_get(gmmk_ar_p,ar_p,meta3d_ar_p)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'ar_p'
      gmmstat = gmm_get(gmmk_ai_p,ai_p,meta3d_ai_p)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'ai_p'

      gmmstat = gmm_get(gmmk_br_p,br_p,meta3d_br_p)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'br_p'
      gmmstat = gmm_get(gmmk_bi_p,bi_p,meta3d_bi_p)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'bi_p'

      gmmstat = gmm_get(gmmk_dumdum_s,dumdum,meta2d_dum)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'dumdum'

!     Valeurs initiales des composantes principales
!     
!     Assure that busper is to 1 at the begining in inichamp1

       if (.not.init_done) then
         if (Lun_out.gt.0) then
            write( Lun_out,1000)
         endif
         init_done=.true.
      endif 
            
!     Rstri_rstn_L
     if ( Step_kount .eq. 1 ) then            
    
       do nc=1,Ens_ptp_ncha
       
         lmin = Ens_ptp_trnl(nc)
         lmax = Ens_ptp_trnh(nc)
         nlon = Ens_ptp_nlon(nc)
         nlat = Ens_ptp_nlat(nc)         
         fstd = Ens_ptp_std(nc)
         tau  = Ens_ptp_tau(nc)/2.146
 
         eps  = exp(-dt/tau)

        allocate ( pspectrum(lmin:lmax) , fact1(lmin:lmax) )                         
!Bruit blanc en nombre d'ondes   
         do l=lmin,lmax    
           pspectrum(l)=1.D0
         enddo
!Normalisation du spectre pour que la variance du champ aléatoire soit std**2
         sumsp=0.D0
         do l=lmin,lmax
          sumsp=sumsp+pspectrum(l)
         enddo
         pspectrum=pspectrum/sumsp

         do l=lmin,lmax
          fact1(l)=fstd*SQRT(4.*pi/real((2*l+1))*pspectrum(l))
         enddo
         
         dumdum(:,nc)=0
         paiv  => dumdum( 1,nc)
         paiy  => dumdum(33,nc)
         paiset=> dumdum(34,nc)
         pagset=> dumdum(35,nc)
         paidum=> dumdum(36,nc)
         paidum=-Ens_mc_seed


! Valeurs initiales des coefficients spectraux
              ar_p=0.d0
              br_p=0.d0
              ai_p=0.d0
              bi_p=0.d0

        do l=lmin,lmax
            br_p(lmax-l+1,1,nc)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)
            ar_p(lmax-l+1,1,nc)=br_p(lmax-l+1,1,nc)
          do m=2,l+1
             br_p(lmax-l+1,m,nc)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)/sqrt(2.)
             ar_p(lmax-l+1,m,nc)=br_p(lmax-l+1,m,nc)
             bi_p(lmax-l+1,m,nc)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)/sqrt(2.)
             ai_p(lmax-l+1,m,nc)=bi_p(lmax-l+1,m,nc)
          enddo
        enddo  

         deallocate (pspectrum, fact1)     
       enddo

    endif

!  Begin Markov chains
   
 allocate(fgem_str(l_ni, l_nj,Ens_ptp_ncha))  

 do nc=1,Ens_ptp_ncha

        lmin = Ens_ptp_trnl(nc) ; nlon = Ens_ptp_nlon(nc)  
        lmax = Ens_ptp_trnh(nc) ; nlat = Ens_ptp_nlat(nc)      
        fstd = Ens_ptp_std(nc)  ; fmin = Ens_ptp_min(nc)
        fstr = Ens_ptp_str(nc)  ; fmax = Ens_ptp_max(nc) 
        tau  =  Ens_ptp_tau(nc)/2.146
     
        eps  = exp(-dt/tau)

! Spectrum choice   

         allocate ( pspectrum(lmin:lmax) )
         allocate ( fact1(lmin:lmax),fact1n(lmin:lmax) )  
        
         do l=lmin,lmax    
           pspectrum(l)=1.D0
         enddo
         sumsp=0.D0
         do l=lmin,lmax
          sumsp=sumsp+pspectrum(l)
         enddo
         pspectrum=pspectrum/sumsp
         fact2 =(1.-eps*eps)/SQRT(1.+eps*eps)

! Random generator function             		 
         paiv  => dumdum( 1,nc) ; paiy  => dumdum(33,nc)
         paiset=> dumdum(34,nc) ; pagset=> dumdum(35,nc)
         paidum=> dumdum(36,nc)
 
       do l=lmin,lmax      
          fact1n(l)=fstd*SQRT(4.*pi/real((2*l+1)) &
                   *pspectrum(l))*SQRT((1.-eps*eps)) 
                
          br_p(lmax-l+1,1,nc) = eps*br_p(lmax-l+1,1,nc)  &
                               + gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)
          ar_p(lmax-l+1,1,nc) = eps*ar_p(lmax-l+1,1,nc)  + br_p(lmax-l+1,1,nc)*fact2
            do m=2,l+1
              br_p(lmax-l+1,m,nc) = eps*br_p(lmax-l+1,m,nc) &
                              + gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)/SQRT(2.)                                         
              ar_p(lmax-l+1,m,nc) = eps*ar_p(lmax-l+1,m,nc)+br_p(lmax-l+1,m,nc)*fact2   
              bi_p(lmax-l+1,m,nc) = eps*bi_p(lmax-l+1,m,nc) &
                              + gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)/SQRT(2.)                                         
              ai_p(lmax-l+1,m,nc) = eps*ai_p(lmax-l+1,m,nc)+bi_p(lmax-l+1,m,nc)*fact2                                                      
           enddo
       enddo
deallocate (pspectrum, fact1, fact1n)  

allocate(p( nlat, lmax-lmin+1, 0:lmax))
allocate(cc(2 , nlat, lmax+1))
allocate(wrk1( nlat * (nlon+2)))
allocate(f(nlon, nlat) )
allocate(f_str(nlon, nlat))
allocate(sig(lmax-lmin+1, 0:lmax))

! Associated Legendre polynomials
       p=0.D0
       do l=lmin,lmax
          fact=DSQRT((2.D0*DBLE(l)+1.D0)/(4.D0*pi))
          do m=0,l
            sig(lmax-l+1,m)=(-1.D0)**(l+m)
            do j=1,nlat/2
             call pleg (l, m, j, nlat, polg)
              p(j,lmax-l+1,m)=polg*fact
              p(nlat-j+1,lmax-l+1,m)=polg*fact*sig(lmax-l+1,m)
            enddo
          enddo
       enddo
     cc=0.D0 

!$omp parallel
!$omp do
    do m=1,lmax+1
        do j=1,nlat/2
              cc(1,j,m)=0.d0
              cc(1,nlat-j+1,m)=0.d0
              cc(2,j,m)=0.d0
              cc(2,nlat-j+1,m)=0.d0 

              cc(1,j,m)=cc(1,j,m) + Dot_product(p(j,1:lmax-lmin+1,m-1),ar_p(1:lmax-lmin+1,m,nc))
              cc(1,nlat-j+1,m)=cc(1,nlat-j+1,m) &
                                  + Dot_product(p(j,1:lmax-lmin+1,m-1),ar_p(1:lmax-lmin+1,m,nc)*sig(1:lmax-lmin+1,m-1))
              cc(2,j,m)=cc(2,j,m) + Dot_product(p(j,1:lmax-lmin+1,m-1),ai_p(1:lmax-lmin+1,m,nc))
              cc(2,nlat-j+1,m)=cc(2,nlat-j+1,m) &
                                  + Dot_product(p(j,1:lmax-lmin+1,m-1),ai_p(1:lmax-lmin+1,m,nc)*sig(1:lmax-lmin+1,m-1))
        enddo
    enddo 

!$omp enddo
!$omp end parallel

! Fourier Transform (inverse)
      
	    wrk1=0.0
            n=-1
            do i=1,nlat
               do j=1,lmax+1      
                 n = n + 2                         
                 wrk1(n)   = cc(1,i,j) 
                 wrk1(n+1) = cc(2,i,j)
               enddo
              n=n+nlon-2*lmax
            enddo

        call itf_fft_set(nlon,'PERIODIC',pri_8)
        call ffft8(wrk1,1,nlon+2,nlat,1)

            n=0
            do j=1,nlat
               do i=1,nlon+2  
                  n = n + 1              
                  if (i .le. nlon) then
                     f(i,j) = wrk1(n)
                  end if
               enddo
            enddo 
  deallocate(p,cc,wrk1,sig)
            
!*    Interpolation to the processors grids and fill in perbus
             
       offi = Ptopo_gindx(1,Ptopo_myproc+1)-1
       offj = Ptopo_gindx(3,Ptopo_myproc+1)-1
  
       do i=1,l_ni
         indx = offi + i
         xfi(i) = G_xg_8(indx)*rad2deg_8
       end do
       do i=1,l_nj
         indx = offj + i
         yfi(i) = G_yg_8(indx)*rad2deg_8
       end do

        gdgauss = ezqkdef(nlon,nlat,'A', 0,0,0,0,0)

        ier = ezdefset(Grd_local_gid, gdgauss)
        ier = ezsetopt('INTERP_DEGREE', 'LINEAR')

!    Check the limits, stretch, and add mean if stretching asked
!    for the physics perturbation
!

        if(Ens_ptp_str(nc)/=0.0)then
          fmean=(fmin+fmax)/2.
          f_str=ERF(f/(fstr*fstd)/SQRT(2.)) *(fmax-fmin)/2. + fmean  
          ier = ezsint(fgem_str(:,:,nc),f_str)
        else
            fgem_str(:,:,nc)=1.0
        endif
!
        if(Ens_stat)then
          call glbstat2 (fgem_str(:,:,nc),'MCPTP','STR',&
           1,l_ni,1,l_nj,1,1,1,G_ni,1,G_nj,1,1)
        endif
     
    deallocate(f,f_str)
 
 enddo

      ptr3d => fgem_str(Grd_lphy_i0:Grd_lphy_in, &
                        Grd_lphy_j0:Grd_lphy_jn, 1:Ens_ptp_ncha)
      istat = phy_put(ptr3d,'mrk2',F_npath='V',F_bpath='P')
   
      deallocate(fgem_str)
     

 1000 format( &
           /,'INITIALIZE SCHEMES CONTROL PARAMETERS (S/R ENS_MARFIELD_PTP)', &
           /,'======================================================')
 6000 format('ens_marfield_ptp at gmm_get(',A,')')


      return

contains

 subroutine pleg(l, m, jlat, nlat , plg )
 implicit none
      
      integer l,m ,i,j , jlat, nlat     
      real*8  factor , x , plg , lat, theta     
      real*8 , dimension(0:l+1) :: pl
      real*8, parameter :: ZERO=0.0D0  , ONE_8=1.0d0 , TWO_8=2.0d0
!
      if ( m < 0 .OR. m > l ) then            
        print*, ' error :  m must non-negative and m <l '    
        stop 
      end if

      pi=real(Dcst_pi_8)
      lat=(-90.D0+90.D0/DBLE(nlat)+DBLE(jlat-1)*180.D0/DBLE(nlat))*pi/180.D0
      theta=pi/2.D0-lat
      x=DCOS(theta)

      pl=ZERO
       if ( m <= l ) then        
           pl(m) = ONE_8     
           factor = ONE_8

          do i = 1, m         
            pl(m) = -pl(m)*factor*sqrt(ONE_8 - x**2)/ &
                   dsqrt(dble((l+i)*(l-m+i)))
            factor = factor + TWO_8
          end do
         plg=pl(m)
       end if

       if ( m + 1 <= l ) then
        plg = x * dble ( 2 * m + 1 ) * pl(m)    
        pl(m+1)=plg
       endif 

       do j = m + 2, l       
          pl(j) = ( x * dble (2*j-1) * pl(j-1) & 
                     + dble (-j-m+1) * pl(j-2) ) &
                     / dble (j-m)
       end do

      plg=pl(l)
     
  end subroutine pleg
      
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
 
END subroutine ens_marfield_ptp

