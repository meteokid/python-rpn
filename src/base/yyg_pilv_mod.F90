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

module yyg_pilv
   implicit none
   public
   save

!
!______________________________________________________________________
!                                                                      |
!  VARIABLES ASSOCIATED WITH BCS communication for Yin-Yang            |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! Pil_recvw_len(M)   | Number of values to receive from each PE for    |
!                    | its West Pilot area  (M=Ptopo_numproc)          |
! Pil_recvw_i(*,M)   | local gridpoint I to receive value from PE(*)   |
! Pil_recvw_j(*,M)   | local gridpoint J to receive value from PE(*)   |
! Pil_sendw_len(*)   | Number of values to send to  PE (*) for West    |
! Pil_sendw_imx(*,M) | closest I gridpoint on the other panel to find  |
!                    | the value for Pil_sendw_xxr,Pil_sendw_yyr       |
! Pil_sendw_imy(*,M) | closest J gridpoint on the other panel to find  |
!                    | the value for  Pil_sendw_xxr,Pil_sendw_yyr      |
! Pil_sendw_xxr(*,M) | longitude in the other panel to find the value  |
!                    | for receiving panel                             |
! Pil_sendw_yyr(*,M) | latitude in the other panel to find the value   |
!                    | for receiving panel                             |
! Pil_sendw_s1(*,M)  | element (s(1,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pil_sendw_s2(*,M)  | element (s(1,2)) in matrix for polar vectors    |
!                    | transformation                                  |
!______________________________________________________________________|
!
!Declarations for V variables (on V grid)
   integer  Pil_vsend_all, Pil_vrecv_all,Pil_vsendmaxproc,Pil_vrecvmaxproc
   integer, dimension (: ), pointer :: &
               Pil_vsendproc, Pil_vrecvproc, &
               Pil_vrecv_len, Pil_vsend_len, &
               Pil_vrecv_adr, Pil_vsend_adr, &
               Pil_vrecv_i   ,Pil_vrecv_j   ,Pil_vsend_imx1, Pil_vsend_imy1, &
               Pil_vsend_imx2,Pil_vsend_imy2
!
   real*8,  dimension (: ), pointer ::  &
               Pil_vsend_xxr,Pil_vsend_yyr, &
               Pil_vsend_s1, Pil_vsend_s2

end module yyg_pilv
