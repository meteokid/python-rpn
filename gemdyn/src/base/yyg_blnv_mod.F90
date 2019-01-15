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

module yyg_blnv
   implicit none
   public
   save

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
!Declarations for V variables (on V grid)
   integer  Bln_vsend_all, Bln_vrecv_all,Bln_vsendmaxproc,Bln_vrecvmaxproc
   integer, dimension (: ), pointer :: &
               Bln_vsendproc, Bln_vrecvproc, &
               Bln_vrecv_len, Bln_vsend_len, &
               Bln_vrecv_adr, Bln_vsend_adr, &
               Bln_vrecv_i   ,Bln_vrecv_j   ,Bln_vsend_imx1, Bln_vsend_imy1, &
               Bln_vsend_imx2,Bln_vsend_imy2

   real*8,  dimension (: ), pointer ::  &
               Bln_vsend_xxr,Bln_vsend_yyr, &
               Bln_vsend_s1, Bln_vsend_s2
end module yyg_blnv
