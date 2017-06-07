      subroutine exp_cal_vor ( F_QR,F_QQ, F_uu,F_vv          , &
                               F_filtqq, F_coefqq, F_absvor_L, &
                               Minx,Maxx,Miny,Maxy,Nk )
      implicit none
      logical  F_absvor_L
      integer  F_filtqq, Minx,Maxx,Miny,Maxy,Nk
      real     F_QR (Minx:Maxx,Miny:Maxy,Nk), &
               F_QQ (Minx:Maxx,Miny:Maxy,Nk), &
               F_uu (Minx:Maxx,Miny:Maxy,Nk), &
               F_vv (Minx:Maxx,Miny:Maxy,Nk), F_coefqq

      return
      end

      subroutine exp_dynstep
      implicit none
      return
      end

      subroutine exp_init_bar ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, gz_t, F_topo,&
                            Mminx,Mmaxx,Mminy,Mmaxy, Nk               ,&
                            F_trprefix_S, F_trsuffix_S, F_datev )

      implicit none

      character* (*) F_trprefix_S, F_trsuffix_S, F_datev
      integer Mminx,Mmaxx,Mminy,Mmaxy,Nk
      real F_u (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_v (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_w (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_t (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_zd(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_s (Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_q (Mminx:Mmaxx,Mminy:Mmaxy,Nk+1), &
           gz_t(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_topo(Mminx:Mmaxx,Mminy:Mmaxy)

      return
      end

      subroutine exp_t02t1
      implicit none
      return
      end

      subroutine exp_set_vt()
      return
      end subroutine exp_set_vt
      integer function exp_nml (F_namelistf_S)
      implicit none
      character* (*) F_namelistf_S
      exp_nml = 0
      return
      end
