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

module yyg_bln
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
! Bln_recv_len(M)    | Number of values to receive from each PE for    |
!                    | its Overlap    area  (M=Ptopo_numproc)          |
! Bln_recv_i(*,M)    | local gridpoint I to receive value from PE(*)   |
! Bln_recv_j(*,M)    | local gridpoint J to receive value from PE(*)   |
! Bln_send_len(*)    | Number of values to send to  PE (*) for West    |
! Bln_send_imx(*,M)  | closest I gridpoint on the other panel to find  |
!                    | the value for Bln_sendw_xxr,Bln_sendw_yyr       |
! Bln_send_imy(*,M)  | closest J gridpoint on the other panel to find  |
!                    | the value for  Bln_sendw_xxr,Bln_sendw_yyr      |
! Bln_send_xxr(*,M)  | longitude in the other panel to find the value  |
!                    | for receiving panel                             |
! Bln_send_yyr(*,M)  | latitude in the other panel to find the value   |
!                    | for receiving panel                             |
! Bln_send_s1(*,M)   | element (s(1,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Bln_send_s2(*,M)   | element (s(1,2)) in matrix for polar vectors    |
!                    | transformation                                  |
! Bln_send_s3(*,M)   | element (s(2,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Bln_send_s4(*,M)   | element (s(2,2)) in matrix for polar vectors    |
!                    | transformation                                  |
!______________________________________________________________________|
!
!Declarations for Scalar variables (on Phi grid)
   integer  Bln_send_all, Bln_recv_all,Bln_sendmaxproc,Bln_recvmaxproc
   integer, dimension (:  ), pointer :: &
            Bln_sendproc, Bln_recvproc, &
            Bln_recv_len, Bln_send_len, &
            Bln_recv_adr, Bln_send_adr, &
            Bln_recv_i   , Bln_recv_j   , Bln_send_imx , Bln_send_imy
!
   real*8,  dimension (: ), pointer :: &
            Bln_send_xxr, Bln_send_yyr, &
            Bln_send_s1,  Bln_send_s2, Bln_send_s3, Bln_send_s4
end module yyg_bln
