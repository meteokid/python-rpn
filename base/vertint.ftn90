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

subroutine vertint (F_dch,F_dstlev,nkd, F_sch,F_srclev,nks, &
                    Minx,Maxx,Miny,Maxy,F_i0,F_in,F_j0,F_jn,F_type, F_sc_L)
   implicit none

#include <arch_specific.hf>

   character(len=*) :: F_type
   logical F_sc_L
   integer,intent(in) :: Minx,Maxx,Miny,Maxy,F_i0,F_in,F_j0,F_jn,nkd, nks
   real :: F_dch(Minx:Maxx,Miny:Maxy,nkd), F_sch(Minx:Maxx,Miny:Maxy,nks), &
           F_srclev(Minx:Maxx,Miny:Maxy,nks), F_dstlev(Minx:Maxx,Miny:Maxy,nkd)

   !@ojective vertical interpolation logp to logp

   !@arguments
   ! F_dch                   Interpolated field           
   ! F_sch                   source       field           
   ! F_srclev                source levels (logp)
   ! F_dstlev                destination levels (logp)
   ! F_type                  order of interpolation ('linear' or 'cubic')
   ! F_sc_L                  Schumann extrapolation

   !@description
   !     General case, we'll care about VT later
   !
   !     First, we find the level we are by squeezing the destination 
   !     between increasing bot() and decreasing top(). We need log_2_(nks)
   !     iteration to squeeze it completely (each integer between 1 and 
   !     nks can be expressed as a sum of power of 2 such as sum c_i 2^i 
   !     where c_i = 0 or 1 and i lower or equal than log_2_(nks)
   !
   !     WARNING
   !  niter calculation is ok for nks.lt. 2097152: should be ok for now...
   !  (Maybe the grid will be that precise in 2010!) (I don't even bother
   !  to add an if statement, for performance purpose...)

#include <thermoconsts.inc>

   logical :: ascending_L
   integer :: i,j,k,iter,niter,lev,lev_lin,k_surf,k_ciel,nlinbot
   integer :: top   (Minx:Maxx,Miny:Maxy),bot   (Minx:Maxx,Miny:Maxy), &
              topcub(Minx:Maxx,Miny:Maxy),botcub(Minx:Maxx,Miny:Maxy), &
              ref   (Minx:Maxx,Miny:Maxy)
   real*8  :: deltalev,prxd,prda,prdb,prsaf,prsbf,prsad,prsbd
   real*8  :: rgasd_8,stlo_8
!
!-------------------------------------------------------------------
!  
   nlinbot=0
   if (any(trim(F_type) == (/'linear','LINEAR'/))) nlinbot=nkd

   ascending_L = (F_srclev(1,1,1) < F_srclev(1,1,nks))
   k_surf = 1
   k_ciel = nks
   if (.not.ascending_L) then
      k_surf = nks
      k_ciel = 1
   endif
   if (real(int(log(real(nks))/log(2.0))).eq. &
        log(real(nks))/log(2.0)) then
      niter=int(log(real(nks))/log(2.0))
   else
      niter=int(log(real(nks))/log(2.0))+1
   endif

   rgasd_8 = dble(rgasd)
   stlo_8  = dble(stlo)

!$omp parallel private(i,j,k,iter,lev,lev_lin,top,bot,topcub,botcub,ref, &
!$omp                  deltalev,prxd,prda,prdb,prsaf,prsbf,prsad,prsbd)  &
!$omp    shared(ascending_L,nlinbot,k_surf,k_ciel,niter,rgasd_8,stlo_8)

!$omp do
   do k=1,nkd
      top=nks
      bot=1
      if (ascending_L) then
         do iter=1,niter
            do j= F_j0, F_jn
            do i= F_i0, F_in
               !     divide by two (the old fashioned way...)
               ref(i,j)=ishft(top(i,j)+bot(i,j),-1)
               !     adjust top or bot
               if(F_dstlev(i,j,k).lt.F_srclev(i,j,ref(i,j))) then
                  top(i,j)=ref(i,j)
               else
                  bot(i,j)=ref(i,j)
               endif
            enddo
            enddo
         enddo
      else
         do iter=1,niter
            do j= F_j0, F_jn
            do i= F_i0, F_in
               !     divide by two (the old fashioned way...)
               ref(i,j)=ishft(top(i,j)+bot(i,j),-1)
               !     adjust top or bot
               if(F_dstlev(i,j,k).gt.F_srclev(i,j,ref(i,j))) then
                  top(i,j)=ref(i,j)
               else
                  bot(i,j)=ref(i,j)
               endif
            enddo
            enddo
         enddo
      endif
      !- adjusting top and bot to ensure we can perform cubic interpolation
      do j= F_j0, F_jn
      do i= F_i0, F_in
         botcub(i,j)=max(    2,bot(i,j))
         topcub(i,j)=min(nks-1,top(i,j))
      enddo
      enddo
      !- cubic or linear interpolation
      do j= F_j0, F_jn
      do i= F_i0, F_in
         lev=botcub(i,j)
         lev_lin=bot(i,j)
         deltalev=(F_srclev(i,j,lev_lin+1)-F_srclev(i,j,lev_lin))

         !- Interpolation: either if not enough points to perform cubic interp
         !                 or if linear interpolation is requested use linear
         !                 interpolation and constant extrapolation
         if((lev.ne.lev_lin).or.(topcub(i,j).ne.top(i,j)).or.k<=nlinbot) then
            !- persistancy of this interval
            if(F_dstlev(i,j,k).le.F_srclev(i,j,k_surf)) then
               F_dch(i,j,k) = F_sch(i,j,k_surf)
            else if(F_dstlev(i,j,k).ge.F_srclev(i,j,k_ciel)) then
               F_dch(i,j,k) = F_sch(i,j,k_ciel)
            else
               !- linear interpolation
               prxd=(F_dstlev(i,j,k)-F_srclev(i,j,lev_lin))/deltalev
               F_dch(i,j,k) = (1.0-prxd)*F_sch(i,j,lev_lin  ) &
                                  +prxd *F_sch(i,j,lev_lin+1)
            endif
         else
            !- cubic interpolation
            prxd = (F_dstlev(i,j,k)-F_srclev(i,j,lev_lin)) / deltalev
            prda = ((F_sch   (i,j,lev_lin+1)-F_sch   (i,j,lev_lin-1))/ &
                    (F_srclev(i,j,lev_lin+1)-F_srclev(i,j,lev_lin-1))* &
                     deltalev)
            prdb = ((F_sch    (i,j,lev_lin+2)-F_sch   (i,j,lev_lin))/ &
                     (F_srclev(i,j,lev_lin+2)-F_srclev(i,j,lev_lin))* &
                     deltalev)
            prsaf= (1.0+2.0*prxd)*(1.0-prxd)*(1.0-prxd)
            prsbf= (3.0-2.0*prxd)*prxd*prxd
            prsad= prxd*(1.0-prxd)*(1.0-prxd)
            prsbd= (1.0-prxd)*prxd*prxd
            F_dch(i,j,k) =  F_sch(i,j,lev_lin  )*prsaf            &
                          + F_sch(i,j,lev_lin+1)*prsbf+prda*prsad &
                          - prdb*prsbd
         endif
      enddo
      enddo
   enddo
!$omp enddo

   if (F_sc_L) then
!$omp do
      do k=1,nkd
      do j= F_j0, F_jn
      do i= F_i0, F_in
         if(F_srclev(i,j,k_ciel).lt.F_dstlev(i,j,k)) then
            F_dch(i,j,k) = F_sch(i,j,k_ciel) * exp (  &
                 rgasd_8*stlo_8*(F_dstlev(i,j,k)-F_srclev(i,j,k_ciel)) )
         endif
      enddo
      enddo
      enddo
!$omp enddo
   endif

!$omp end parallel

!
!-------------------------------------------------------------------
!
   return
   end
subroutine vertint3
print*, 'STOP in vertint3'
stop
end
