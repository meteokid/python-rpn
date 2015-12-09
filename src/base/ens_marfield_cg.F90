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

!**   s/r ens_marfield - define a markov chain field
!     

!     
 !     subroutine ens_marfield (E_nk)
 		 subroutine ens_marfield(fgem)
!
      use phy_itf, only : phy_put
 
      implicit none
#include <arch_specific.hf>

      integer                               :: E_nk
      real ,    dimension(l_ni,l_nj)        :: fgem
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
       real*8,    external :: ran1, gasdev
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
      real    ::  fstd, fstr, tau,  Ens_mc2d_mean
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
      real*8    :: pi, dt, eps, sig2  
      real*8    :: fmax, fmin , fmean 
      real*8,    dimension(:), allocatable :: pspectrum , fact1, fact1n,ai
      real*8,    dimension(:), allocatable  :: wrk1
      real*8,    dimension(:,:,:),allocatable :: wrk2, p,cc
      real,    dimension(:,:),allocatable :: fgau, fgau_str,ar
      real,    dimension(:,:,:),pointer   :: fgem2_str, ptr3d, fgem_str
!       
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
       
         lmin = Ens_mc2d_trnl(nc)
         lmax = Ens_mc2d_trnh(nc)
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
  

! Valeurs initiales des coefficients spectraux
  		   anm=0.d0
		   bnm=0.d0
         do l=lmin,lmax
              bnm(lmax-l+1,1,nc)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum)
              anm(lmax-l+1,1,nc)=bnm(lmax-l+1,1,nc)
           do m=2,l+1
              bnm(lmax-l+1,m,nc)=fact1(l)*gasdev(paiv,paiy,paiset,pagset,paidum) &
                                 /sqrt(2.)
              anm(lmax-l+1,m,nc)=bnm(lmax-l+1,m,nc)
           enddo
         enddo  

         deallocate (pspectrum, fact1)     
       enddo

    endif
              
!     
!     Begin Markov chains
!   


 allocate(fgem2_str(l_ni, l_nj,Ens_mc2d_ncha-1))  

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
                 
       do l=lmin,lmax                      
          bnm(lmax-l+1,1,nc) = eps*bnm(lmax-l+1,1,nc)  + &
                               gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)
          anm(lmax-l+1,1,nc) = eps*anm(lmax-l+1,1,nc)  + bnm(lmax-l+1,1,nc)*fact2
            do m=2,l+1
              bnm(lmax-l+1,m,nc) = eps*bnm(lmax-l+1,m,nc) +  &
                              gasdev(paiv,paiy,paiset,pagset,paidum)*fact1n(l)/SQRT(2.)                                         
              anm(lmax-l+1,m,nc) = eps*anm(lmax-l+1,m,nc)+bnm(lmax-l+1,m,nc)*fact2                                        
           enddo
       enddo
      
deallocate (pspectrum, fact1, fact1n)  


allocate(p( nlat, lmax-lmin+1, 0:lmax))
allocate(cc(2 , nlat, lmax+1))
allocate(wrk1( nlat * (nlon+2)))
allocate(ai(lmax-lmin+1))
allocate(fgau(nlon, nlat) )
allocate(fgau_str(nlon, nlat))

   
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

        ai(1:lmax-lmin+1)=0.d0
        cc=0.D0
       do m=1,lmax+1

           call DGEMV('N',nlat , lmax-lmin+1, 1.d0, p(1,1,m-1), nlat, &
                       anm(:,m,nc) ,1,0.d0,cc(1,1,m),2)
          if (m==1)then          
           call DGEMV('N',nlat , lmax-lmin+1, 1.D0, p(1,1,m-1), nlat, &
                        ai,1,0.D0,cc(2,m,1),2)
           deallocate(ai)
          else 
           call DGEMV('N',nlat , lmax-lmin+1, 1.d0, p(1,1,m-1), nlat, &
                        anm(:,m,nc),1,0.D0,cc(2,1,m),2)
          endif
       
       enddo 

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
                     fgau(i,j) = wrk1(n)
                  end if
               enddo
            enddo 
 
  deallocate(p,cc,wrk1)
             
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
 
         ier = ezsint(fgem,fgau)
                
         if(.not.Ens_ptp_conf) then
         deallocate(fgem2_str)           
          return
         endif
    !    fgem_str=ERF(fgem/(fstr*fstd)/SQRT(2.)) *(fmax-fmin)/2. + fmean 
    !    fgem_str(:,:,E_nk+1)=0.0          
    ! ptr3d => fgem_str(Grd_lphy_i0:Grd_lphy_in, &
    !                   Grd_lphy_j0:Grd_lphy_jn, 1:E_nk+1)
    !  istat = phy_put(fgem_str,'mrkv',F_npath='V',F_bpath='P')      
    !   deallocate(fgem_str)
   
   else

  if(.not.Ens_ptp_conf) return

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
            
    endif
    deallocate(fgau,fgau_str)
 
 enddo


      ptr3d => fgem2_str(Grd_lphy_i0:Grd_lphy_in, &
                        Grd_lphy_j0:Grd_lphy_jn, 1:Ens_mc2d_ncha-1)
      istat = phy_put(ptr3d,'mrk2',F_npath='V',F_bpath='P')
   
      deallocate(fgem2_str)
     

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
 
  FUNCTION PLGNDR(L,M,X)
      IMPLICIT NONE
      real*8 PLGNDR,X,PMM,SOMX2,FACT,PMMP1,PLL
      integer L,M,I,LL
      IF(M.LT.0.OR.M.GT.L.OR.DABS(X).GT.1.D0) THEN
        PRINT *, 'Error - Stop in PLGNDR (1)'
        PRINT *, 'bad arguments'
        STOP
      ENDIF
      PMM=1.D0
      IF(M.GT.0) THEN
        SOMX2=DSQRT((1.D0-X)*(1.D0+X))
        FACT=1.D0
        DO 11 I=1,M
          PMM=-PMM*FACT*SOMX2/DSQRT(DBLE((L+I)*(L-M+I)))
          FACT=FACT+2.D0
11      CONTINUE
      ENDIF
      IF(L.EQ.M) THEN
        PLGNDR=PMM
      ELSE
        PMMP1=X*DBLE(2*M+1)*PMM
        IF(L.EQ.M+1) THEN
          PLGNDR=PMMP1
        ELSE
          DO 12 LL=M+2,L
            PLL=(X*DBLE(2*LL-1)*PMMP1-DBLE(LL+M-1)*PMM)/DBLE(LL-M)
            PMM=PMMP1
            PMMP1=PLL
12        CONTINUE
          PLGNDR=PLL
        ENDIF
      ENDIF
      RETURN
      END function PLGNDR
!-------

END subroutine ens_marfield

