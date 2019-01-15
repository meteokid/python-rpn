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
      use out_collector, only: block_collect_set, Bloc_me
      implicit none
#include <arch_specific.hf>

#include <WhiteBoard.hf>
#include "lun.cdk"
#include "ptopo.cdk"
#include "inp.cdk"
#include "grd.cdk"
#include "grid.cdk"
#include "schm.cdk"
#include "glb_ld.cdk"
#include "step.cdk"
#include "path.cdk"
#include "out3.cdk"
#include "tr3d.cdk"
#include "wil_williamson.cdk"
#include "wil_nml.cdk"
      include 'out_meta.cdk'
      include "rpn_comm.inc"

      integer, external :: gem_nml,gemdm_config,grid_nml2       ,&
                           adv_nml,adv_config,adx_nml,adx_config,&
                           step_nml, set_io_pes, domain_decomp3
      character*50 LADATE,dumc1_S
      integer :: istat,options,wload,hzd,monot,massc
      integer err(8),f1,f2,f3,f4
      real vmin
!
!-------------------------------------------------------------------
!
      err(:) = 0
      err(1) = wb_put( 'model/Hgrid/is_yinyang',Grd_yinyang_L,&
                       WB_REWRITE_NONE+WB_IS_LOCAL )
      if (Grd_yinyang_L) then         
         Path_ind_S=trim(Path_input_S)//'/MODEL_INPUT/'&
                                      //trim(Grd_yinyang_S)
         err(2) = wb_put( 'model/Hgrid/yysubgrid',Grd_yinyang_S,&
                          WB_REWRITE_NONE+WB_IS_LOCAL )
      else
         Path_ind_S=trim(Path_input_S)//'/MODEL_INPUT'
      endif
      Path_phy_S=trim(Path_input_S)//'/'

      if ( Schm_theoc_L ) then
         call theo_cfg
      else

! Read namelists from file Path_nml_S

         err(3) = grid_nml2   (Path_nml_S,G_lam)
         err(4) = step_nml    (Path_nml_S)
         err(5) = gem_nml     (Path_nml_S)
         if (G_lam .and. .not. Schm_adxlegacy_L ) then
            err(6) = adv_nml  (Path_nml_S)
         else
            err(6) = adx_nml  (Path_nml_S)
         endif

      endif

      call gem_error ( minval(err(:)),'set_world_view',&
                       'Error reading nml or with wb_put' )

! Read physics namelist

      call itf_phy_nml

! Establish final configuration

      err(1) = gemdm_config ()
      if (.not.G_lam) err(2) = adx_config ()

      call gem_error(min(err(1),err(2)),'set_world_view','config')

      err(1) = grid_nml2 ('print',G_lam)
      err(1) = step_nml  ('print')
      err(1) = gem_nml   ('print')

      if (G_lam .and. .not. Schm_adxlegacy_L ) then
         call adv_nml_print ()
      else
         call adx_nml_print ()
      endif

      if (Williamson_case.ne.0.and.Lun_out.gt.0) &
                   write (Lun_out, nml=williamson) 

! Establish domain decomposition (mapping subdomains and processors)

      err(1)= domain_decomp3 (Ptopo_npex, Ptopo_npey, .false.)
      call gem_error ( err(1),'DOMAIN_DECOMP', &
                      'ILLEGAL DOMAIN PARTITIONING' )

      if (lun_out.gt.0) then 
         f1 = G_ni/Ptopo_npex + min(1,mod(G_ni,Ptopo_npex))
         f2 = G_ni-f1*(Ptopo_npex-1)
         f3 = G_nj/Ptopo_npey + min(1,mod(G_nj,Ptopo_npey))
         f4 = G_nj-f3*(Ptopo_npey-1)
         write (lun_out,1001) Grd_typ_S,G_ni,G_nj,G_nk,f1,f3,f2,f4
         LADATE='RUNSTART='//Step_runstrt_S(1:8)//Step_runstrt_S(10:11)
         call write_status_file3 (trim(LADATE))
         call write_status_file3 ( 'communications_established=YES' )
         if (Grd_yinyang_L)    &
         call write_status_file3 ('GEM_YINYANG=1')
      endif

! Master output PE for all none distributed components

      options = WB_REWRITE_NONE+WB_IS_LOCAL
      f1= 0
      istat = wb_put('model/outout/pe_master', f1,options)
      istat = min(wb_put('model/l_minx',l_minx,options),istat)
      istat = min(wb_put('model/l_maxx',l_maxx,options),istat)
      istat = min(wb_put('model/l_miny',l_miny,options),istat)
      istat = min(wb_put('model/l_maxy',l_maxy,options),istat)
      call gem_error ( istat,'set_world_view', &
                       'Problem with min-max wb_put')

! Establish a grid id for RPN_COMM package and obtain Out3_iome,Inp_iome

      Out3_npes= max(1,min(Out3_npes,min(Ptopo_npex,Ptopo_npey)**2))
      Inp_npes = max(1,min(Inp_npes ,min(Ptopo_npex,Ptopo_npey)**2))

      err= 0
      if (lun_out.gt.0) write (lun_out,1002) 'Output',Out3_npes
      err(1)= set_io_pes (Out3_comm_id,Out3_comm_setno,Out3_iome,&
                          Out3_comm_io,Out3_iobcast,Out3_npes)
      if (lun_out.gt.0) write (lun_out,1002) 'Input',Inp_npes
      err(2)= set_io_pes (Inp_comm_id ,Inp_comm_setno ,Inp_iome ,&
                          Inp_comm_io ,Inp_iobcast ,Inp_npes )
      call gem_error ( min(err(1),err(2)),'set_world_view', &
                       'IO pes config is invalid' )

      Out3_ezcoll_L= .true.
      if ( (Out3_npex > 0) .and. (Out3_npey > 0) ) then
         Out3_npex= min(Out3_npex,Ptopo_npex)
         Out3_npey= min(Out3_npey,Ptopo_npey)
         call block_collect_set ( Out3_npex, Out3_npey )
         Out3_npes= Out3_npex * Out3_npey
         Out3_iome= -1
         if (Bloc_me == 0) Out3_iome= 0
         Out3_ezcoll_L= .false.
      endif
      out_stk_size= Out3_npes*2

      call tracers_attributes2 ( 'DEFAULT,'//trim(Tr3d_default_s), &
                                 wload, hzd, monot, massc, vmin )

      if (Lun_out.gt.0) then
         if (trim(Tr3d_default_s)=='') then
            write (Lun_out,'(/a)') &
            ' SYSTEM DEFAULTS FOR TRACERS ATTRIBUTES:'
         else
            write (Lun_out,'(/a)') &
            ' USER DEFAULTS FOR TRACERS ATTRIBUTES:'
         endif
         write (Lun_out,2001)
         write (Lun_out,2002) wload,hzd,monot,massc,vmin
      endif

! Initializes GMM

      call heap_paint

      call set_gmm
			
 1001 format (' GRID CONFIG: GRTYP=',a,5x,'GLB=(',i5,',',i5,',',i5,')    maxLCL(',i4,',',i4,')    minLCL(',i4,',',i4,')')
 1002 format (/ ' Creating IO pe set for ',a,' with ',i4,' Pes')
 2001 format ( ' DEFAULT tracers attributes:'/3x,'Wload  Hzd   Mono  Mass    Min')
 2002 format (4i6,3x,e9.3)
!
!-------------------------------------------------------------------
!
      return
      end
