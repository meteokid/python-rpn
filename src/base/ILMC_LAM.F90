!**s/p ILMC_LAM - Ensures monotonicity of interpolated field while preserving mass (Sorenson et al.,2013) - Yin-Yang/LAM 

      subroutine ILMC_LAM (F_name_S,F_out,F_high,F_min,F_max,Minx,Maxx,Miny,Maxy,F_nk,k0,F_ILMC_min_max_L,F_ILMC_sweep_max)

      use array_ILMC

      implicit none

      !Arguments
      !---------
      character (len=*), intent(in) :: F_name_S                           !I, Name of field to be ajusted
      integer,           intent(in) :: Minx,Maxx,Miny,Maxy                !I, Dimension H
      integer,           intent(in) :: k0                                 !I, Scope of operator
      integer,           intent(in) :: F_nk                               !I, Number of vertical levels
      logical,           intent(in) :: F_ILMC_min_max_L                   !I, T IF MONO(CLIPPING) after ILMC 
      integer,           intent(in) :: F_ILMC_sweep_max                   !I, T Number of neighborhood zones in ILMC 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(out)    :: F_out  !I: Corrected ILMC solution 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)     :: F_high !I: High-order SL solution 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)     :: F_min  !I: MIN over cell
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)     :: F_max  !I: MAX over cell
 
      !Author Qaddouri/Tanguay
      !
      !Revision
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation
      !
      !Object
      !     Based on Sorenson et al.,2013: A mass conserving and multi-tracer
      !     efficient transport scheme in the online integrated Enviro-HIRLAM model.
      !     Geosci. Model Dev., 6, 1029-1042, 2013   
      !
!**/
#include "glb_ld.cdk"
#include "lun.cdk"
#include "schm.cdk"
#include "grd.cdk"
#include "tracers.cdk"

      !----------------------------------------------------------
      integer SIDE_max_0,i,j,k,err,sweep,i_rd,j_rd,k_rd,ii,ix,jx,kx,kx_m,kx_p,iprod, &
              istat,shift,j1,j2,reset(k0:F_nk,3),l_reset(3),g_reset(3),time_m,w1,w2,size,n,il,ir,jl,jr,il_c,ir_c,jl_c,jr_c

      real F_new(Minx:Maxx,Miny:Maxy,F_nk),l_density(Minx:Maxx,Miny:Maxy,F_nk),l_mass(Minx:Maxx,Miny:Maxy,F_nk), &
           dif_p(400),dif_m(400),ok_p,s_dif_p,ok_m,s_dif_m,o_shoot,u_shoot,ratio_p,ratio_m

      real F_copy(Minx:Maxx,Miny:Maxy,F_nk)

      real*8 mass_new_8,mass_out_8,ratio_8,mass_deficit_8

      logical limit_i_L, verbose_L
      logical,save :: done_L=.false.

      !----------------------------------------------------------

      verbose_L = Tr_verbose/=0 

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

      !----------------------------
      !Define CORE and OUTSIDE CORE
      !----------------------------
      il_c = pil_w+1
      ir_c = l_ni-pil_e
      jl_c = pil_s+1
      jr_c = l_nj-pil_n

      il = Minx
      ir = Maxx
      jl = Miny
      jr = Maxy
      if (l_west)  il = il_c
      if (l_east)  ir = ir_c
      if (l_south) jl = jl_c
      if (l_north) jr = jr_c

      !Default values if no Mass correction
      !------------------------------------
      F_out = F_high 

      call mass_tr (mass_new_8,time_m,F_name_S(4:7),F_new,.TRUE.,Minx,Maxx,Miny,Maxy,F_nk-k0+1,k0,"",.TRUE.)

      if (verbose_L) then
      if (Lun_out>0) then
         write(Lun_out,*)    'TRACERS: Do MONO (CLIPPING)      =',F_ILMC_min_max_L
         write(Lun_out,1000) 'TRACERS: Mass BEFORE ILMC        =',mass_new_8,F_name_S(4:6)
      endif
      endif

      call get_density (l_density,l_mass,time_m,Minx,Maxx,Miny,Maxy,F_nk,k0)

      !-------------------------------
      !Fill East-West/North-South Halo
      !-------------------------------
      call rpn_comm_xch_halo (F_min, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,F_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      call rpn_comm_xch_halo (F_max, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,F_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      call rpn_comm_xch_halo (l_mass,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,F_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      !------------------------------------------------------
      !Allocation and Define surrounding cells for each sweep
      !------------------------------------------------------
      if (.NOT.done_L) then

      allocate (sweep_rd(F_ILMC_sweep_max,l_ni,l_nj,F_nk))

      do k=1,F_nk
         do j=1,l_nj
         do i=1,l_ni
         do n=1,F_ILMC_sweep_max

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
!$omp shared (j1,j2,sweep_rd,il,ir,jl,jr,il_c,ir_c,jl_c,jr_c)

      do k=k0,F_nk

         kx_m = k
         kx_p = k

         do j=jl_c,jr_c
         do i=il_c,ir_c

            do sweep = 1,F_ILMC_sweep_max

               sweep_rd(sweep,i,j,k)%cell = 0

               if (.NOT.Schm_autobar_L) then
                  kx_m = max(k-sweep,k0)
                  kx_p = min(k+sweep,F_nk)
               endif

               j1 = max(j-sweep,jl)
               j2 = min(j+sweep,jr)

               do jx = j1,j2

                  do kx = kx_m,kx_p

                     limit_i_L = (jx/=j-sweep.and.jx/=j+sweep).and.((.NOT.Schm_autobar_L.and.(kx>k-sweep.and.k<k+sweep)).or.Schm_autobar_L)

                     do ix = max(i-sweep,il),min(i+sweep,ir)

                        if (limit_i_L.and.ix/=i-sweep.and.ix/=i+sweep) cycle

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
      SIDE_max_0 = (2*F_ILMC_sweep_max + 1)*2+1

#define CORE
#include "ilmc_lam_loop.cdk"

      !-------------------------------
      !Fill East-West/North-South Halo
      !-------------------------------
      call rpn_comm_xch_halo (F_new, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,F_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      !------------------------------------------------------------------------------------
      !Set LAM NEST boundary of F_new to ZERO since Adjoint is done after OUTSIDE CORE loop
      !------------------------------------------------------------------------------------
      if (l_west)  F_new(Minx:il-1,Miny:Maxy,k0:F_nk) = 0. 
      if (l_east)  F_new(ir+1:Maxx,Miny:Maxy,k0:F_nk) = 0. 
      if (l_south) F_new(Minx:Maxx,Miny:jl-1,k0:F_nk) = 0. 
      if (l_north) F_new(Minx:Maxx,jr+1:Maxy,k0:F_nk) = 0. 

      !----------------------------------------------------------------------------
      !Compute ILMC solution F_out while preserving mass: USE ELEMENTS OUTSIDE CORE
      !----------------------------------------------------------------------------

      F_copy = F_new

#undef CORE
#include "ilmc_lam_loop.cdk"

      !-----------------------------------------------------------
      !Recover perturbation in East-West/North-South halo of F_new
      !-----------------------------------------------------------
!$omp parallel do private(i,j) shared(F_new,F_copy,il,ir,jl,jr,il_c,ir_c,jl_c,jr_c)
      do k=k0,F_nk
         do j=jl,jr
            do i=il,ir
               if ((i>=il_c.and.i<=ir_c).and.(j>=jl_c.and.j<=jr_c)) cycle
               F_new(i,j,k) = F_new(i,j,k) - F_copy(i,j,k)
            enddo
         enddo
      enddo
!$omp end parallel do

      F_copy = F_new

      !------------------------------------------
      !Adjoint of Fill East-West/North-South Halo
      !------------------------------------------
      call rpn_comm_adj_halo (F_copy,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,F_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      !To maintain nesting values
      !--------------------------
      F_out (1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k0:F_nk) = &
      F_copy(1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k0:F_nk)

      !---------------------------------------
      !Reset Min-Max Monotonicity if requested
      !---------------------------------------
      if (F_ILMC_min_max_L) then

!$omp parallel do private(i,j) shared(F_out,F_min,F_max,reset,il_c,ir_c,jl_c,jr_c)
      do k=k0,F_nk
         do j=jl_c,jr_c
         do i=il_c,ir_c

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

      if (Grd_yinyang_L) then
         call RPN_COMM_allreduce (l_reset,g_reset,3,"MPI_INTEGER","MPI_SUM","MULTIGRID",err)
         iprod = 2
      else
         call RPN_COMM_allreduce (l_reset,g_reset,3,"MPI_INTEGER","MPI_SUM","GRID",err)
         iprod = 1 
      endif

      if (verbose_L) then
      if (Lun_out>0) then
         write(Lun_out,1000) 'TRACERS: Mass  AFTER ILMC        =',mass_out_8,F_name_S(4:6)
         write(Lun_out,*)    'TRACERS: # pts OVER/UNDER SHOOT  =',g_reset(3),'over',G_ni*G_nj*F_nk*iprod
         write(Lun_out,*)    'TRACERS: # pts CLIPPED           =',g_reset(1) + g_reset(2)
         write(Lun_out,*)    'TRACERS: RESET_MIN_ILMC          =',g_reset(1)
         write(Lun_out,*)    'TRACERS: RESET_MAX_ILMC          =',g_reset(2)
         write(Lun_out,1001) 'TRACERS: Rev. Diff. of ',ratio_8
      endif

      if (Lun_out>0) write(Lun_out,*) 'TRACERS: --------------------------------------------------------------------------------'
      endif

 1000 format(1X,A34,E20.12,1X,A3)
 1001 format(1X,A23,E11.4,'%')

      return
      end
