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

!**   s/r ens_marfield_skeb - define a markov chain field for SKEB	
!     
!     
      subroutine ens_marfield_skeb(fgem)
!
      use phy_itf, only : phy_put
 
      implicit none
#include <arch_specific.hf>

      integer                               :: E_nk
      real ,    dimension(l_ni,l_nj)        :: fgem
!     
!author: Rabah Aider R.P.N-A (from ens_marfield_cg.ftn90)
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
       real,    external :: gasdev
       real*8  pl
!
! nlat, nlon                 dimension of the Gaussian grid
! idum                       Semence du générateur de nombres aléatoires
!
      integer :: nlat, nlon , dgid_myp
      integer :: l ,m, i, j, k, indx, ier
      integer lmin,lmax 
      integer ::  gmmstat, istat
      integer :: gdgem,gdgauss, keys(5), n
      real    ::  fstd, fstr, tau,  Ens_skeb_mean
      real    :: sumsp , fact,fact2
      real    :: xf,offi,offj
      real xfi(l_ni),yfi(l_nj)
      real*8  ::  lat, theta ,x, rad2deg_8, pri_8
      logical, save :: init_done=.false.
!
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
      real*8,    dimension(:,:),allocatable :: wrk2
      real*8,    dimension(:,:,:),allocatable :: cc
      real  ,    dimension(:,:),allocatable :: f
      integer :: sig
!
! Initialise les variables de Ens_nml.cdk
!

      dt=real(Cstv_dt_8)
      pi=real(Dcst_pi_8)
      rad2deg_8=180.0d0/Dcst_pi_8
!
!     Look for the spectral coefficients
!
      gmmstat = gmm_get(gmmk_ar_s,ar_s,meta2d_ar_s)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'ar_s'

      gmmstat = gmm_get(gmmk_ai_s,ai_s,meta2d_ai_s)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'ai_s'

      gmmstat = gmm_get(gmmk_br_s,br_s,meta2d_br_s)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'br_s'
      gmmstat = gmm_get(gmmk_bi_s,bi_s,meta2d_bi_s)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'bi_s'
      gmmstat = gmm_get(gmmk_dumdum_s,dumdum,meta2d_dum)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'dumdum'

      gmmstat = gmm_get(gmmk_plg_s,plg,meta3d_plg)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'plg'

!     Valeurs initiales des composantes principales
!     
       if (.not.init_done) then
         if (Lun_out.gt.0) then
            write( Lun_out,1000)
         endif
         init_done=.true.
      endif 
           
      paiv  => dumdum( 1,1)
      paiy  => dumdum(33,1)
      paiset=> dumdum(34,1)
      pagset=> dumdum(35,1)
      paidum=> dumdum(36,1)
 
!     Rstri_rstn_L
     if ( Step_kount .eq. 1 ) then            
    
         lmin = Ens_skeb_trnl
         lmax = Ens_skeb_trnh
         nlon = Ens_skeb_nlon
         nlat =Ens_skeb_nlat         
         fstd=Ens_skeb_std

         tau = Ens_skeb_tau/2.146
         eps=exp(-dt/tau)

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

         paiv  => dumdum( 1,1)
         paiy  => dumdum(33,1)
         paiset=> dumdum(34,1)
         pagset=> dumdum(35,1)
         paidum=> dumdum(36,1)
         dumdum(:,1)=0
         paidum=-Ens_mc_seed

! Valeurs initiales des coefficients spectraux
              ar_s=0.d0
              br_s=0.d0
              ai_s=0.d0
              bi_s=0.d0

        do l=lmin,lmax
            br_s(lmax-l+1,1)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)
            ar_s(lmax-l+1,1)=br_s(lmax-l+1,1)
          do m=2,l+1
             br_s(lmax-l+1,m)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)/sqrt(2.)
             ar_s(lmax-l+1,m)=br_s(lmax-l+1,m)
             bi_s(lmax-l+1,m)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)/sqrt(2.)
             ai_s(lmax-l+1,m)=bi_s(lmax-l+1,m)
          enddo
        enddo  


! Associated Legendre polynomials
      plg=0.D0
     do l=lmin,lmax
       fact=DSQRT((2.D0*DBLE(l)+1.D0)/(4.D0*pi))
        do m=0,l
          sig=(-1.D0)**(l+m)
!$omp parallel private(j,pl)  &
!$omp shared (l,m,nlat,lmax,sig,fact,pi)
!$omp  do    
          do j=1,nlat/2
             call pleg (l, m, j, nlat, pl)
              plg(j,lmax-l+1,m+1)=pl*fact
              plg(nlat-j+1,lmax-l+1,m+1)=pl*fact*sig
          enddo
!$omp enddo 
!$omp end parallel
       enddo
     enddo
 deallocate (pspectrum, fact1)     

    endif
!     
!     Begin Markov chains
!   

        lmin = Ens_skeb_trnl ; nlon = Ens_skeb_nlon  
        lmax = Ens_skeb_trnh ; nlat = Ens_skeb_nlat      
        fstd = Ens_skeb_std  ; fmin = Ens_skeb_min
        fmax = Ens_skeb_max  ; 
        tau  = Ens_skeb_tau/2.146      
      
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
         paiv  => dumdum( 1,1) 
         paiy  => dumdum(33,1)
         pagset=> dumdum(35,1)
         paiset=> dumdum(34,1)    
         paidum=> dumdum(36,1)

       do l=lmin,lmax  
          fact1n(l)=fstd*SQRT(4.*pi/real((2*l+1)) &
                   *pspectrum(l))*SQRT((1.-eps*eps)) 

          br_s(lmax-l+1,1) = eps*br_s(lmax-l+1,1)  &
                               + gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)
          ar_s(lmax-l+1,1) = eps*ar_s(lmax-l+1,1)  + br_s(lmax-l+1,1)*fact2
            do m=2,l+1
              br_s(lmax-l+1,m) = eps*br_s(lmax-l+1,m) &
                              + gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)/SQRT(2.)                                         
              ar_s(lmax-l+1,m) = eps*ar_s(lmax-l+1,m)+br_s(lmax-l+1,m)*fact2   
              bi_s(lmax-l+1,m) = eps*bi_s(lmax-l+1,m) &
                              + gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)/SQRT(2.)                                         
              ai_s(lmax-l+1,m) = eps*ai_s(lmax-l+1,m)+bi_s(lmax-l+1,m)*fact2                                                      
           enddo
       enddo
deallocate (pspectrum, fact1, fact1n)  

allocate(cc(2 , nlat, lmax+1))
allocate(wrk1( nlat * (nlon+2)))
allocate(f(nlon, nlat) )

       cc=0.D0

!$omp parallel
!$omp do
 	do m=1,lmax+1
          do j=1,nlat
               cc(1,j,m)=0.d0
               cc(2,j,m)=0.d0

              cc(1,j,m)=cc(1,j,m) + Dot_product(plg(j,1:lmax-lmin+1,m),ar_s(1:lmax-lmin+1,m))
              cc(2,j,m)=cc(2,j,m) + Dot_product(plg(j,1:lmax-lmin+1,m),ai_s(1:lmax-lmin+1,m))
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
 
  deallocate(cc,wrk1)

!*    Interpolation to the processors grids 
             
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
 
        ier = ezsint(fgem,f)
                
       if(Ens_stat)then
          call glbstat2 (fgem,'MCSK','',&
           1,l_ni,1,l_nj,1,1,1,G_ni,1,G_nj,1,1)
        endif

    deallocate(f)
 
 

 1000 format( &
           /,'INITIALIZE SCHEMES CONTROL PARAMETERS (S/R ENS_MARFIELD_SKEB)', &
           /,'======================================================')
 6000 format('ens_marfield_skeb at gmm_get(',A,')')

      return

contains

 subroutine pleg(l, m, jlat, nlat, plg )
 implicit none
      
      integer l,m ,i,j ,jlat ,nlat
      real*8   plg
      real*8  factor , x  ,lat, theta     
      real*8 , dimension(0:l+1) :: pl
      real*8, parameter :: ZERO=0.0D0  , ONE_8=1.0d0 , TWO_8=2.0d0
      

!
      if ( m < 0 .OR. m > l ) then            
        print*, ' error :  m must non-negative and m <=l '    
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
            pl(m) = -pl(m)*factor*sqrt(1.d0 - x**2)/ &
                   dsqrt(dble((l+i)*(l-m+i)))
            factor = factor + 2.d0			
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
 
END subroutine ens_marfield_skeb

