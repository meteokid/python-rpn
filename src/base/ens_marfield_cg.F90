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

!**   s/r ens_marfield - define a markov chain field
!     

!     
      subroutine ens_marfield (fgem,E_nk)
!
      use phy_itf, only : phy_put
 
      implicit none
#include <arch_specific.hf>

      integer                               :: E_nk
      real,    dimension(l_ni,l_nj,E_nk)    :: fgem
!     
!author: *Model Infrastructure Group*  R.P.N-A
!     
!object
!	ens_marfield_cg.ftn90
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
       real,    external :: ran1, gasdev
       real*8 :: plg
!       integer, external :: fnom,fstouv,fstfrm,fclos
!       integer, external :: ezgdef_fmem,ezqkdef,ezsint,ezdefset,ezsetopt

!
! nlat, nlon                 dimension of the Gaussian grid
! idum                       Semence du générateur de nombres aléatoires
!
      integer :: nlat, nlon,  ndim_tot, dgid_myp
      integer :: l ,m, nc,nc2, i, j, k, indx, ier, sig
      integer lmin,lmax 
      integer :: soit,lght,indx_n, gmmstat, istat
      integer :: gdgem,gdgauss, keys(5), nmstrt, nm, n
      real    :: xmn, xmx, xme, fstd, fstr, aa, tau, ar, Ens_mc2d_mean
      real    :: sumsp , fact,fact2
      real    :: xf,offi,offj
      real xfi(l_ni),yfi(l_nj)
      real*8  :: rad2deg_8, lat, theta ,x, pri_8
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
      real rand
!
!      variables et champs auxiliaires reels
! dt   Pas de temps du modèle (secondes)
! tau  Temps de décorrélation du champ aléatoire f(i,j) (secondes)
! eps  EXP(-dt/tau)
      real    :: pi, dt, eps, sig2   
      real    :: fmax, fmin , fmean 
      real,    dimension(:), allocatable :: pspectrum , fact1, fact1n
      real,    dimension(:), allocatable  :: wrk1
      real,    dimension(:,:,:),allocatable :: wrk2, p,cc 
      real,    dimension(:,:),allocatable :: fgau, fgau_str
      real,    dimension(:,:,:),pointer   :: fgem2_str, ptr2d
      
!
!
!
! Initialise les variables de Ens_nml.cdk
!
      
      dt=real(Cstv_dt_8)
      pi=real(Dcst_pi_8)
      rad2deg_8=180.0d0/Dcst_pi_8
!
!     Look for the spectral coefficients
!
      gmmstat = gmm_get(gmmk_anm_s,anm,meta3d_anm)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'anm'
      gmmstat = gmm_get(gmmk_bnm_s,bnm,meta3d_bnm)  
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'bnm'
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
    
       do nc=1,Ens_mc2d_ncha
       
         lmin = Ens_mc2d_trnh(nc)
         lmax = Ens_mc2d_trnl(nc)
         nlon = Ens_mc2d_nlon(nc)
         nlat =Ens_mc2d_nlat(nc)         
         fstd=Ens_mc2d_std(nc)
         eps=exp(-dt/Ens_mc2d_tau(nc))

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
       
         rand=gasdev(paiv,paiy,paiset,pagset,paidum)

! Valeurs initiales des coefficients spectraux
         do l=lmin,lmax
              bnm(lmax-l+1,0,nc)=CMPLX(fact1(l)*rand, 0)
              anm(lmax-l+1,0,nc)=bnm(lmax-l+1,0,nc)
           do m=1,l
              bnm(lmax-l+1,m,nc)=CMPLX(fact1(l)*rand/sqrt(2.), fact1(l)*rand/sqrt(2.))
              anm(lmax-l+1,m,nc)=bnm(lmax-l+1,m,nc)
           enddo
         enddo
    
         deallocate (pspectrum, fact1)     
       enddo

    endif
                    
!     
!     Begin Markov chains
!   

    
allocate(p( maxval(Ens_mc2d_nlat), Ens_dim2_lmax, 0:maxval(Ens_mc2d_trnh)))
allocate(cc(2, maxval(Ens_mc2d_trnh)+1 , maxval(Ens_mc2d_nlat)))
allocate(wrk1( maxval(Ens_mc2d_nlat) * (maxval(Ens_mc2d_nlon)+2)))
allocate(fgau(maxval(Ens_mc2d_nlat), maxval(Ens_mc2d_nlon)) )
allocate(fgau_str(maxval(Ens_mc2d_nlat), maxval(Ens_mc2d_nlon)))
allocate(fgem2_str(l_ni, l_nj, Ens_mc2d_ncha-1))  
 
 do nc=1,Ens_mc2d_ncha
          
        lmin = Ens_mc2d_trnl(nc) ; nlon = Ens_mc2d_nlon(nc)  
        lmax = Ens_mc2d_trnh(nc) ; nlat = Ens_mc2d_nlat(nc)      
        fstd = Ens_mc2d_std(nc)  ; fmin = Ens_mc2d_min(nc)
        fstr = Ens_mc2d_str(nc)  ; fmax = Ens_mc2d_max(nc) 
          
        eps  = exp(-dt/Ens_mc2d_tau(nc))

! Choix du spectre   

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

         do l=lmin,lmax
          fact1(l)=fstd*SQRT(4.*pi/real((2*l+1))*pspectrum(l))
          fact1n=fact1(l)*SQRT((1.-eps*eps))   ! assure l'ergodicité des chaînes
         enddo        
         fact2 =(1.-eps*eps)/SQRT(1.+eps*eps)
! Random generator function             		 
         paiv  => dumdum( 1,nc) ; paiy  => dumdum(33,nc)
         paiset=> dumdum(34,nc) ; pagset=> dumdum(35,nc)
         paidum=> dumdum(36,nc)
        
         rand=gasdev(paiv,paiy,paiset,pagset,paidum)
!                 
       do l=lmin,lmax                      
          bnm(lmax-l+1,0,nc) = CMPLX( eps*REAL( bnm(lmax-l+1,0,nc) ) + rand*fact1n(l), 0)
          anm(lmax-l+1,0,nc) = CMPLX( eps*REAL( anm(lmax-l+1,0,nc) ) + REAL(bnm(lmax-l+1,0,nc)*fact2), 0)
            do m=1,l
              bnm(lmax-l+1,m,nc) = CMPLX( eps*REAL(bnm(lmax-l+1,m,nc)) + rand*fact1n(l)/SQRT(2.),  &
                                         eps*AIMAG(bnm(lmax-l+1,m,nc)) + rand*fact1n(l)/SQRT(2.) )     
              anm(lmax-l+1,m,nc) = CMPLX(eps*REAL(anm(lmax-l+1,m,nc))+REAL(bnm(lmax-l+1,m,nc)*fact2), &
                                         eps*AIMAG(anm(lmax-l+1,m,nc))+AIMAG(bnm(lmax-l+1,m,nc)*fact2) )
           enddo
       enddo
      
      deallocate (pspectrum, fact1, fact1n)  
! Associated Legendre polynomials	    
        p=0.D0
        do l=lmin,lmax
          fact=DSQRT((2.D0*DBLE(l)+1.D0)/(4.D0*pi))
          do m=0,l
            sig=(-1.D0)**(l+m)
            do j=1,nlat/2
             lat=(-90.D0+90.D0/DBLE(nlat)+DBLE(j-1)*180.D0/DBLE(nlat))*pi/180.D0
             theta=pi/2.D0-lat
             x=DCOS(theta)
             call pleg (l, m, x, plg)
              p(j,lmax-l+1,m)=plg*fact
              p(nlat-j+1,lmax-l+1,m)=p(j,lmax-l+1,m)*sig
            enddo
          enddo
       enddo
            
        cc=0.D0
       do m=0,lmax
         call DGEMV('N',nlat , lmax-lmin+1, 1.D0, p(1,1,m), nlat, &
                        REAL(anm(1,m,nc)),1,0.D0,cc(1,m+1,1),2)
         call DGEMV('N',nlat , lmax-lmin+1, 1.D0, p(1,1,m), nlat, &
                       AIMAG(anm(1,m,nc)),1,0.D0,cc(2,m+1,1),2)
       enddo 

! Fourier Transform (inverse)
      
	         wrk1=0.0
            n=-1
            do j=1,nlat
               do i=1,lmax+1          
                 n = n + 2                         
                 wrk1(n)   = cc(1,i,j) 
                 wrk1(n+1) = cc(2,i,j)
               enddo
              n=n+nlon-2*lmax
            enddo
 
          call itf_fft_set(nlon,'PERIODIC',pri_8)
          call ffft8(wrk1,nlon+2,nlat,1)		
          
            fgau=0
            n=0
            do j=1,nlat
               do i=1,nlon+2  
                  n = n + 1
                  if (i .le. nlon) then
                     fgau(i,j) = wrk1(n)
                  end if
               enddo
            enddo 
             

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

       dgid_myp = ezgdef_fmem (l_ni , l_nj , 'Z', 'E', Hgc_ig1ro, &
                Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, xfi , yfi )

        gdgauss = ezqkdef(nlon,nlat,'G', 0,0,0,0,0)

        ier = ezdefset(dgid_myp, gdgauss)
        ier = ezsetopt('INTERP_DEGREE', 'LINEAR')
!       ier = ezsetopt('VERBOSE', 'YES')

! Markov Chain for SKEB
         
    if (nc .eq. 1) then
          allocate(wrk2(l_ni,l_nj,E_nk))            
            do k=1,E_nk           
            wrk2(:,:,k)=fgau(:,:)
            ier = ezsint(fgem(1,1,k),wrk2(1,1,k))
            enddo
          deallocate(wrk2)
    else
                
!Markov Chain for ptp: nc=2,3, ...

!    Check the limits, stretch, and add mean if stretching asked
!    for the physics perturbation
!
        where (abs(fgau) > 1.)
          fgau=sign(1.,fgau)
        end where

        nc2=nc-1

        if(Ens_mc2d_str(nc)/=0.0)then
          fmean=fmin+fmax/2.
          fgau_str=ERF(fgau/(fstr*fstd)/SQRT(2.)) *(fmax-fmin)/2. + fmean  
          ier = ezsint(fgem2_str(:,:,nc2),fgau_str)
        else
            fgem2_str(:,:,nc2)=1.0
        endif
!
        if(Ens_stat)then
          call glbstat2 (fgem2_str(:,:,nc2),'MC2D','STR',&
           1,l_ni,1,l_nj,1,1,1,G_ni,1,G_nj,1,1)
        endif
            
        deallocate(fgau,fgau_str)
        deallocate(p,wrk1,cc)
    endif
  
 enddo
      ptr2d => fgem2_str(Grd_lphy_i0:Grd_lphy_in, &
                        Grd_lphy_j0:Grd_lphy_jn, 1:Ens_mc2d_ncha-1)
      istat = phy_put(ptr2d,'mrk2',F_npath='V',F_bpath='P')
    
      deallocate(fgem2_str)
!     
!
 1000 format( &
           /,'INITIALIZE SCHEMES CONTROL PARAMETERS (S/R ENS_MARFIELD_CG)', &
           /,'======================================================')
 6000 format('ens_marfield_cg at gmm_get(',A,')')
      return

contains

 subroutine pleg(l, m, x, plg )
 implicit none
      
      integer l,m ,i,j      
      real*8  factor , x , plg      
      real*8 , dimension(0:l+1) :: pl
      real*8, parameter :: ZERO=0.0D0  , ONE_8=1.0d0 , TWO_8=2.0d0
!
      if ( m < 0 .OR. m > l ) then            
        print*, ' error :  m must non-negative and m <l '    
        stop 
      end if
     
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
 
 
END subroutine ens_marfield

