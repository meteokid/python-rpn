!**s/p ILMC_GU - Ensures monotonicity of interpolated field while preserving mass (Sorenson et al.,2013) - Global Uniform (LEGACY)

      subroutine adx_ILMC_GU (F_name_S,F_out,F_high,F_min,F_max,Minx,Maxx,Miny,Maxy,F_nk,k0)

      use array_ILMC

      implicit none

      !Arguments
      !---------
      character (len=*), intent(in) :: F_name_S                           !I, Name of field to be ajusted
      integer,           intent(in) :: Minx,Maxx,Miny,Maxy                !I, Dimension H
      integer,           intent(in) :: k0                                 !I, Scope of operator
      integer,           intent(in) :: F_nk                               !I, Number of vertical levels
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(out)    :: F_out  !I: Corrected ILMC solution 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)     :: F_high !I: High-order SL solution 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)     :: F_min  !I: MIN over cell
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)     :: F_max  !I: MAX over cell
 
      !Author Monique Tanguay 
      !
      !Revision
      ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation
      !
      !Object
      !     Based on Sorenson et al.,2013: A mass conserving and multi-tracer
      !     efficient transport scheme in the online integrated Enviro-HIRLAM model.
      !     Geosci. Model Dev., 6, 1029-1042, 2013   
      !
!*@/
#include "glb_ld.cdk"
#include "lun.cdk"
#include "schm.cdk"
#include "adx_nml.cdk"
#include "adx_poles.cdk"
#include "adx_dims.cdk"

      !----------------------------------------------------------
      integer SIDE_max_0,i,j,k,err,sweep,i_rd,j_rd,k_rd,ii,ix,jx,kx,kx_m,kx_p, &
              istat,shift,nrow,j1,j2,reset(k0:F_nk,3),l_reset(3),g_reset(3),time_m,w1,w2,size,n,e1,e2,en

      real F_new(Minx:Maxx,Miny:Maxy,F_nk),l_density(Minx:Maxx,Miny:Maxy,F_nk),l_mass(Minx:Maxx,Miny:Maxy,F_nk), &
           dif_p(400),dif_m(400),ok_p,s_dif_p,ok_m,s_dif_m,o_shoot,u_shoot,ratio_p,ratio_m

      real a_new (adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk), &
           a_min (adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk), &
           a_max (adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk), &
           a_mass(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk), &
           a_copy(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk)

      real*8 mass_new_8,mass_out_8,ratio_8,mass_deficit_8

      logical limit_i_L, verbose_L
      logical,save :: done_L=.false.

      !----------------------------------------------------------

      verbose_L = Adw_verbose/=0 

      time_m = 0

      reset  = 0

      if (verbose_L) then
      if (Lun_out>0) then
         write(Lun_out,*) 'TRACERS: --------------------------------------------------------------------------------'
         write(Lun_out,*) 'TRACERS: Reset Monotonicity without changing Mass of Cubic: Sorensen et al,ILMC, 2013,GMD'
         write(Lun_out,*) 'TRACERS: --------------------------------------------------------------------------------'
      endif
      endif

      F_new = F_high

      !Default values if no Mass correction
      !------------------------------------
      F_out = F_high 

      call mass_tr (mass_new_8,time_m,F_name_S(4:7),F_new,.TRUE.,Minx,Maxx,Miny,Maxy,F_nk-k0+1,k0,"",.TRUE.)

      if (verbose_L) then
      if (Lun_out>0) then
         write(Lun_out,*)    'TRACERS: Do MONO (CLIPPING)      =',Adw_ILMC_min_max_L
         write(Lun_out,1000) 'TRACERS: Mass BEFORE ILMC        =',mass_new_8,F_name_S(4:6)
      endif
      endif

      call get_density (l_density,l_mass,time_m,Minx,Maxx,Miny,Maxy,F_nk,k0)

      nrow = 999
      if (adx_lam_L) nrow = 0

      !-------------------------------
      !Fill East-West/North-South Halo
      !-------------------------------
      call rpn_comm_xch_halox (F_min,Minx,Maxx,Miny,Maxy,                  &
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_min,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      call rpn_comm_xch_halox (F_max,Minx,Maxx,Miny,Maxy,                  &
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_max,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      call rpn_comm_xch_halox (l_mass,Minx,Maxx,Miny,Maxy,                 &
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_mass,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      call rpn_comm_xch_halox (F_new,Minx,Maxx,Miny,Maxy,                  & !NEEDED to Fill East-West band 
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_new,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      if (.NOT.l_west) a_new = 0. 

      !********************************************
      if (l_west) then !ONLY L_WEST DO CALCULATIONS
      !********************************************

      !-----------------------------------------------------------------------------------------------------------------
      !Set East-West/North-South halo of a_new to ZERO since Adjoint (Needed for East-West band) is done after CORE loop
      !-----------------------------------------------------------------------------------------------------------------
      a_new(adx_lminx:adx_lmaxx,adx_lminy :0        ,k0:F_nk) = 0. 
      a_new(adx_lminx:adx_lmaxx,adx_mlnj+1:adx_lmaxy,k0:F_nk) = 0. 
      a_new(adx_lminx:0        ,adx_lminy :adx_lmaxy,k0:F_nk) = 0. 
      a_new(adx_lni+1:adx_lmaxx,adx_lminy :adx_lmaxy,k0:F_nk) = 0. 

      !------------------------------------------------------
      !Allocation and Define surrounding cells for each sweep
      !------------------------------------------------------
      if (.NOT.done_L) then

      allocate (sweep_rd(Adw_ILMC_sweep_max,adx_lni,adx_mlnj,F_nk))

      do k=1,F_nk
         do j=1,adx_mlnj
         do i=1,adx_lni
         do n=1,Adw_ILMC_sweep_max

            w1   = 2*n + 1
            w2   = w1-2
            size = 2*(w1**2 + w1*w2 + w2*w2)

            allocate (sweep_rd(n,i,j,k)%i_rd(size))
            allocate (sweep_rd(n,i,j,k)%j_rd(size))
            allocate (sweep_rd(n,i,j,k)%k_rd(size))

         enddo
         enddo
         enddo
      enddo

!$omp parallel do                                        &
!$omp private(sweep,i,j,ii,ix,jx,kx,kx_m,kx_p,limit_i_L) &
!$omp shared (j1,j2,e1,e2,en,sweep_rd)

      do k=k0,F_nk

         kx_m = k
         kx_p = k

         do j=1,adx_mlnj
         do i=1,adx_lni

            do sweep = 1,Adw_ILMC_sweep_max

               sweep_rd(sweep,i,j,k)%cell = 0

               if (.NOT.Schm_autobar_L) then
                  kx_m = max(k-sweep,k0)
                  kx_p = min(k+sweep,F_nk)
               endif

               j1 = max(j-sweep,adx_lminy)
               j2 = min(j+sweep,adx_lmaxy)

               if ((j-sweep<=0.and.l_south).or.(j+sweep>=adx_mlnj+1.and.l_north)) then

                   if (j-sweep<=0.and.l_south) then
                      e1 = 1
                      e2 = 1-(j-sweep)
                      en = 1
                      j1 = 1
                   else
                      e1 = adx_mlnj 
                      e2 = 2*adx_mlnj+1-(j+sweep)
                      en =-1
                      j2 = adx_mlnj 
                   endif

                   do jx = e1,e2,en

                      do kx = kx_m,kx_p

                         limit_i_L = (jx/=e2).and.((.NOT.Schm_autobar_L.and.(kx>k-sweep.and.k<k+sweep)).or.Schm_autobar_L)

                         do ii = adx_iln(i)-sweep,adx_iln(i)+sweep

                            if (limit_i_L.and.ii/=adx_iln(i)-sweep.and.ii/=adx_iln(i)+sweep) cycle

                            ix = ii

                            if (ix<1      ) ix = ix + adx_lni
                            if (ix>adx_lni) ix = ix - adx_lni

                            sweep_rd(sweep,i,j,k)%i_rd(sweep_rd(sweep,i,j,k)%cell + 1) = ix
                            sweep_rd(sweep,i,j,k)%j_rd(sweep_rd(sweep,i,j,k)%cell + 1) = jx
                            sweep_rd(sweep,i,j,k)%k_rd(sweep_rd(sweep,i,j,k)%cell + 1) = kx

                            sweep_rd(sweep,i,j,k)%cell = sweep_rd(sweep,i,j,k)%cell + 1

                         enddo

                      enddo

                   enddo

               endif

               do jx = j1,j2 

                  do kx = kx_m,kx_p

                     limit_i_L = (jx/=j-sweep.and.jx/=j+sweep).and.((.NOT.Schm_autobar_L.and.(kx>k-sweep.and.k<k+sweep)).or.Schm_autobar_L)

                     do ii = i-sweep,i+sweep

                        if (limit_i_L.and.ii/=i-sweep.and.ii/=i+sweep) cycle

                        ix = ii

                        if (ix<1      ) ix = ix + adx_lni
                        if (ix>adx_lni) ix = ix - adx_lni

                        sweep_rd(sweep,i,j,k)%i_rd(sweep_rd(sweep,i,j,k)%cell + 1) = ix
                        sweep_rd(sweep,i,j,k)%j_rd(sweep_rd(sweep,i,j,k)%cell + 1) = jx
                        sweep_rd(sweep,i,j,k)%k_rd(sweep_rd(sweep,i,j,k)%cell + 1) = kx

                        sweep_rd(sweep,i,j,k)%cell = sweep_rd(sweep,i,j,k)%cell + 1

                     enddo

                  enddo

               enddo

            enddo ! Do sweep

         enddo
         enddo

      enddo
!$omp end parallel do

      done_L = .TRUE.

      endif 

      !---------------------------------------------------------------------------
      !Compute ILMC solution F_out while preserving mass: USE ELEMENTS INSIDE CORE
      !---------------------------------------------------------------------------

      !Evaluate admissible distance between threads
      !--------------------------------------------
      SIDE_max_0 = (2*Adw_ILMC_sweep_max + 1)*2+1

#define CORE
#include "ilmc_gu_loop.cdk"

      !********************************************
      endif !ONLY L_WEST DO CALCULATIONS
      !********************************************

      F_new = 0.

      !----------------------------------------------------------------------
      !Adjoint of Fill East-West/North-South Halo (Needed for East-West band) 
      !----------------------------------------------------------------------
      call rpn_comm_adj_halox (F_new,Minx,Maxx,Miny,Maxy,                  &
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_new,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      !-------------------------------
      !Fill East-West/North-South Halo
      !-------------------------------
      call rpn_comm_xch_halox (F_new,Minx,Maxx,Miny,Maxy,                  & 
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_new,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      if (.NOT.l_west) a_new = 0. 

      !********************************************
      if (l_west) then !ONLY L_WEST DO CALCULATIONS
      !********************************************

      !---------------------------------------------------------------------------------
      !Set East-West halo of a_new to ZERO since Adjoint is done after OUTSIDE CORE loop
      !---------------------------------------------------------------------------------
      a_new(adx_lminx:0        ,adx_lminy :adx_lmaxy,k0:F_nk) = 0.
      a_new(adx_lni+1:adx_lmaxx,adx_lminy :adx_lmaxy,k0:F_nk) = 0.

      !----------------------------------------------------------------------------
      !Compute ILMC solution F_out while preserving mass: USE ELEMENTS OUTSIDE CORE
      !----------------------------------------------------------------------------

      a_copy = a_new

#undef CORE
#include "ilmc_gu_loop.cdk"

      !-------------------------------------------------
      !Recover perturbation in North-South halo of a_new
      !-------------------------------------------------
!$omp parallel do private(i,j) shared(a_new,a_copy)
      do k=k0,F_nk
         do j=adx_lminy,adx_lmaxy
            if (j>=1.and.j<=adx_mlnj) cycle
            do i=adx_lminx,adx_lmaxx
               a_new(i,j,k) = a_new(i,j,k) - a_copy(i,j,k) 
            enddo
         enddo
      enddo
!$omp end parallel do

      !********************************************
      endif !ONLY L_WEST DO CALCULATIONS
      !********************************************

      F_out = 0.

      !------------------------------------------
      !Adjoint of Fill East-West/North-South Halo
      !------------------------------------------
      call rpn_comm_adj_halox (F_out,Minx,Maxx,Miny,Maxy,                  &
                               adx_mlni,adx_mlnj,F_nk,adx_halox,adx_haloy, &
                               adx_is_period_x,adx_is_period_y,            &
                               a_new,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,adx_lni,nrow)

      !---------------------------------------
      !Reset Min-Max Monotonicity if requested
      !---------------------------------------
      if (Adw_ILMC_min_max_L) then

!$omp parallel do private(i,j) shared(F_out,F_min,F_max,reset)
      do k=k0,F_nk
         do j=1,adx_mlnj
         do i=1,adx_mlni

            if (F_out(i,j,k) < F_min(i,j,k)) then

                reset(k,1)   = reset(k,1) + 1
                F_out(i,j,k) = F_min(i,j,k)

            endif

            if (F_out(i,j,k) > F_max(i,j,k)) then

                reset(k,2)   = reset(k,2) + 1
                F_out(i,j,k) = F_max(i,j,k)

            endif

         enddo
         enddo
      enddo
!$omp end parallel do

      endif

      call mass_tr (mass_out_8,time_m,F_name_S(4:7),F_out,.TRUE.,Minx,Maxx,Miny,Maxy,F_nk-k0+1,k0,"",.FALSE.)

      mass_deficit_8 = mass_out_8 - mass_new_8

      ratio_8 = 0.0d0
      if (mass_new_8/=0.d0) ratio_8 = mass_deficit_8/mass_new_8*100.

      !--------------------------------------
      !Print diagnostics Min-Max Monotonicity
      !--------------------------------------
      l_reset = 0
      g_reset = 0

      do k=k0,F_nk
         l_reset(1) = reset(k,1) + l_reset(1)
         l_reset(2) = reset(k,2) + l_reset(2)
         l_reset(3) = reset(k,3) + l_reset(3)
      enddo

      call RPN_COMM_allreduce (l_reset,g_reset,3,"MPI_INTEGER","MPI_SUM","NS",err)

      if (verbose_L) then
      if (Lun_out>0) then
         write(Lun_out,1000) 'TRACERS: Mass  AFTER ILMC        =',mass_out_8,F_name_S(4:6)
         write(Lun_out,*)    'TRACERS: # pts OVER/UNDER SHOOT  =',g_reset(3),'over',G_ni*G_nj*F_nk
         write(Lun_out,*)    'TRACERS: # pts CLIPPED           =',g_reset(1) + g_reset(2),'over',G_ni*G_nj*F_nk
         write(Lun_out,*)    'TRACERS: RESET_MIN_ILMC          =',g_reset(1),'over',G_ni*G_nj*F_nk
         write(Lun_out,*)    'TRACERS: RESET_MAX_ILMC          =',g_reset(2),'over',G_ni*G_nj*F_nk
         write(Lun_out,1001) 'TRACERS: Rev. Diff. of ',ratio_8
      endif

      if (Lun_out>0) write(Lun_out,*) 'TRACERS: --------------------------------------------------------------------------------'
      endif

 1000 format(1X,A34,E20.12,1X,A3)
 1001 format(1X,A23,E11.4,'%')

      return
      end
