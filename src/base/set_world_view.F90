!---------------------------------- LICENCE BEGIN -------------------------------

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

!**s/r set_world_view

      subroutine set_world_view
use iso_c_binding
      implicit none
#include <arch_specific.hf>

!author
!     Michel Desgagne

#include <WhiteBoard.hf>
#include "lun.cdk"
#include "ptopo.cdk"
#include "grd.cdk"
#include "grid.cdk"
#include "schm.cdk"
#include "glb_ld.cdk"
#include "step.cdk"
#include "path.cdk"
#include "wil_williamson.cdk"
#include "wil_nml.cdk"
      include "rpn_comm.inc"

      integer, external :: gem_nml,gemdm_config,grid_nml2, &
                           adv_nml,adv_config,adx_nml,adx_config, &
                           step_nml,from_ntr
      character*50 LADATE,dumc1_S
      integer ierr,err(8),f1,f2,f3,f4,n1,n2,n3,n4,n5,n6,n7
!
!-------------------------------------------------------------------
!
      err = RPN_COMM_bloc ( Ptopo_nblocx, Ptopo_nblocy )
      call gem_error(err,'pe_all_topo','rpn_comm_bloc')

      call RPN_COMM_carac ( Ptopo_npex,Ptopo_npey,Ptopo_myproc, &
                            n1,n2,n3,n4,n5,n6,n7 ,Ptopo_mybloc, &
              Ptopo_myblocx,Ptopo_myblocy,Ptopo_blocme,dumc1_S )

      call RPN_COMM_size ( RPN_COMM_GRIDPEERS, Grid_nnodes, ierr )

      if (Grd_yinyang_L) then         
         Path_ind_S=trim(Path_input_S)//'/MODEL_INPUT/'//trim(Grd_yinyang_S)
      else
         Path_ind_S=trim(Path_input_S)//'/MODEL_INPUT'
      endif
      Path_phy_S=trim(Path_input_S)//'/'

      err(:) = 0

      if ( Schm_theoc_L ) then
         call theo_cfg
      else
!
! Read namelists from file Path_nml_S
!
         err(1) = grid_nml2   (Path_nml_S,G_lam)
         err(2) = step_nml    (Path_nml_S)
         err(3) = gem_nml     (Path_nml_S)
         if (Advection_lam_legacy) then
            err(4) = adx_nml  (Path_nml_S)
         else
            err(4) = adv_nml  (Path_nml_S)
         endif
         err(5) = from_ntr ()

      endif

      call gem_error(minval(err(:)),'set_world_view','Error reading nml')
!
! Read physics namelist
!
      call itf_phy_nml
!
! Establish final configuration
!
      err(1) = gemdm_config ()
      if (.not.G_lam) err(2) = adx_config ()

      call gem_error(min(err(1),err(2)),'set_world_view','config')

      ierr = grid_nml2 ('print',G_lam)
      ierr = step_nml  ('print')
      ierr = gem_nml   ('print')

      if ( Schm_cub_traj_L .and. (.not.G_lam) ) &
           call gem_error(-1,'set_world_view','Schm_cub_traj_L=.true. cannot be used with non LAM grid')
 
      if (Advection_lam_legacy) then
         call adx_nml_print ()
      else
         call adv_nml_print ()
      endif

      if (Williamson_case.ne.0.and.Lun_out.gt.0) write (Lun_out, nml=williamson) 
!
! Establish domain decomposition (mapping subdomains and processors)
!
      call domain_decomp2 (Ptopo_npex, Ptopo_npey, .false.)

      if (lun_out.gt.0) then
         f1 = G_ni/Ptopo_npex + min(1,mod(G_ni,Ptopo_npex))
         f2 = G_ni-f1*(Ptopo_npex-1)
         f3 = G_nj/Ptopo_npey + min(1,mod(G_nj,Ptopo_npey))
         f4 = G_nj-f3*(Ptopo_npey-1)
         write (lun_out,1001) Grd_typ_S,G_ni,G_nj,G_nk,f1,f3,f2,f4
         LADATE='RUNSTART='//Step_runstrt_S(1:8)//Step_runstrt_S(10:11)
         call write_status_file3 (trim(LADATE))
         call write_status_file3 ('communications_established=YES' )
         if (Grd_yinyang_L)    &
         call write_status_file3 ('GEM_YINYANG=1')
      endif
!
! Master output PE for all none distributed components
!
      f1= 0
      ierr= wb_put('model/outout/pe_master', f1)
!
! Establish a grid id for RPN_COMM package and obtain Grid_comm_iome
!
      Grid_comm_id    = RPN_COMM_create_2dgrid (  G_ni, G_nj, &
                              l_minx, l_maxx, l_miny, l_maxy  )
      Grid_comm_setno = RPN_COMM_create_io_set (              &
                              Ptopo_nblocx*Ptopo_nblocy , 0   )
      Grid_iome       = RPN_COMM_is_io_pe ( Grid_comm_setno )
!
! Initializes GMM
!
      call set_gmm
			
 1001 format (' GRID CONFIG: GRTYP=',a,5x,'GLB=(',i5,',',i5,',',i5,')    maxLCL(',i4,',',i4,')    minLCL(',i4,',',i4,')')
!
!-------------------------------------------------------------------
!
      return
      end
