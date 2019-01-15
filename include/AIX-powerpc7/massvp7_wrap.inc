!  Wrapper interfaces for Power7 vector MASS functions (v6.1)

interface vsacos
   subroutine mw_vsacos(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsacos
   subroutine mw_vsacos_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsacos_2d
   subroutine mw_vsacos_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsacos_s
end interface

interface vsacosh
   subroutine mw_vsacosh(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsacosh
   subroutine mw_vsacosh_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsacosh_2d
   subroutine mw_vsacosh_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsacosh_s
end interface

interface vsasin
   subroutine mw_vsasin(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsasin
   subroutine mw_vsasin_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsasin_2d
   subroutine mw_vsasin_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsasin_s
end interface

interface vsasinh
   subroutine mw_vsasinh(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsasinh
   subroutine mw_vsasinh_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsasinh_2d
   subroutine mw_vsasinh_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsasinh_s
end interface

interface vsatan
   subroutine mw_vsatan(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsatan
   subroutine mw_vsatan_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsatan_2d
   subroutine mw_vsatan_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsatan_s
end interface

interface vsatan2
   subroutine mw_vsatan2(z,x,y,n)
      real*4 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vsatan2
   subroutine mw_vsatan2_2d(z,x,y,n)
      real*4 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vsatan2_2d
   subroutine mw_vsatan2_s(z,x,y,n)
      real*4 z,x,y
      integer*4 n
   end subroutine mw_vsatan2_s
end interface

interface vsatanh
   subroutine mw_vsatanh(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsatanh
   subroutine mw_vsatanh_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsatanh_2d
   subroutine mw_vsatanh_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsatanh_s
end interface

interface vscbrt
   subroutine mw_vscbrt(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vscbrt
   subroutine mw_vscbrt_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vscbrt_2d
   subroutine mw_vscbrt_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vscbrt_s
end interface

interface vscos
   subroutine mw_vscos(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vscos
   subroutine mw_vscos_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vscos_2d
   subroutine mw_vscos_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vscos_s
end interface

interface vscosh
   subroutine mw_vscosh(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vscosh
   subroutine mw_vscosh_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vscosh_2d
   subroutine mw_vscosh_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vscosh_s
end interface

interface vscosisin
   subroutine mw_vscosisin(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vscosisin
   subroutine mw_vscosisin_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vscosisin_2d
   subroutine mw_vscosisin_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vscosisin_s
end interface

interface vsdiv
   subroutine mw_vsdiv(z,x,y,n)
      real*4 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vsdiv
   subroutine mw_vsdiv_2d(z,x,y,n)
      real*4 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vsdiv_2d
   subroutine mw_vsdiv_s(z,x,y,n)
      real*4 z,x,y
      integer*4 n
   end subroutine mw_vsdiv_s
end interface

interface vserf
   subroutine mw_vserf(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vserf
   subroutine mw_vserf_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vserf_2d
   subroutine mw_vserf_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vserf_s
end interface

interface vserfc
   subroutine mw_vserfc(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vserfc
   subroutine mw_vserfc_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vserfc_2d
   subroutine mw_vserfc_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vserfc_s
end interface

interface vsexp
   subroutine mw_vsexp(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsexp
   subroutine mw_vsexp_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsexp_2d
   subroutine mw_vsexp_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsexp_s
end interface

interface vsexp2
   subroutine mw_vsexp2(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsexp2
   subroutine mw_vsexp2_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsexp2_2d
   subroutine mw_vsexp2_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsexp2_s
end interface

interface vsexp2m1
   subroutine mw_vsexp2m1(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsexp2m1
   subroutine mw_vsexp2m1_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsexp2m1_2d
   subroutine mw_vsexp2m1_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsexp2m1_s
end interface

interface vsexpm1
   subroutine mw_vsexpm1(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsexpm1
   subroutine mw_vsexpm1_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsexpm1_2d
   subroutine mw_vsexpm1_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsexpm1_s
end interface

interface vshypot
   subroutine mw_vshypot(z,x,y,n)
      real*4 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vshypot
   subroutine mw_vshypot_2d(z,x,y,n)
      real*4 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vshypot_2d
   subroutine mw_vshypot_s(z,x,y,n)
      real*4 z,x,y
      integer*4 n
   end subroutine mw_vshypot_s
end interface

interface vslgamma
   subroutine mw_vslgamma(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vslgamma
   subroutine mw_vslgamma_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vslgamma_2d
   subroutine mw_vslgamma_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vslgamma_s
end interface

interface vslog
   subroutine mw_vslog(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vslog
   subroutine mw_vslog_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vslog_2d
   subroutine mw_vslog_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vslog_s
end interface

interface vslog10
   subroutine mw_vslog10(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vslog10
   subroutine mw_vslog10_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vslog10_2d
   subroutine mw_vslog10_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vslog10_s
end interface

interface vslog1p
   subroutine mw_vslog1p(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vslog1p
   subroutine mw_vslog1p_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vslog1p_2d
   subroutine mw_vslog1p_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vslog1p_s
end interface

interface vslog2
   subroutine mw_vslog2(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vslog2
   subroutine mw_vslog2_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vslog2_2d
   subroutine mw_vslog2_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vslog2_s
end interface

interface vslog21p
   subroutine mw_vslog21p(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vslog21p
   subroutine mw_vslog21p_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vslog21p_2d
   subroutine mw_vslog21p_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vslog21p_s
end interface

interface vspow
   subroutine mw_vspow(z,x,y,n)
      real*4 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vspow
   subroutine mw_vspow_2d(z,x,y,n)
      real*4 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vspow_2d
   subroutine mw_vspow_s(z,x,y,n)
      real*4 z,x,y
      integer*4 n
   end subroutine mw_vspow_s
end interface

interface vsqdrt
   subroutine mw_vsqdrt(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsqdrt
   subroutine mw_vsqdrt_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsqdrt_2d
   subroutine mw_vsqdrt_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsqdrt_s
end interface

interface vsrcbrt
   subroutine mw_vsrcbrt(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsrcbrt
   subroutine mw_vsrcbrt_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsrcbrt_2d
   subroutine mw_vsrcbrt_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsrcbrt_s
end interface

interface vsrec
   subroutine mw_vsrec(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsrec
   subroutine mw_vsrec_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsrec_2d
   subroutine mw_vsrec_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsrec_s
end interface

interface vsrqdrt
   subroutine mw_vsrqdrt(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsrqdrt
   subroutine mw_vsrqdrt_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsrqdrt_2d
   subroutine mw_vsrqdrt_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsrqdrt_s
end interface

interface vsrsqrt
   subroutine mw_vsrsqrt(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vsrsqrt
   subroutine mw_vsrsqrt_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsrsqrt_2d
   subroutine mw_vsrsqrt_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vsrsqrt_s
end interface

interface vssin
   subroutine mw_vssin(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vssin
   subroutine mw_vssin_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vssin_2d
   subroutine mw_vssin_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vssin_s
end interface

interface vssincos
   subroutine mw_vssincos(y,z,x,n)
      real*4 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vssincos
   subroutine mw_vssincos_2d(y,z,x,n)
      real*4 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vssincos_2d
   subroutine mw_vssincos_s(y,z,x,n)
      real*4 z,x,y
      integer*4 n
   end subroutine mw_vssincos_s
end interface

interface vssinh
   subroutine mw_vssinh(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vssinh
   subroutine mw_vssinh_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vssinh_2d
   subroutine mw_vssinh_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vssinh_s
end interface

interface vssqrt
   subroutine mw_vssqrt(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vssqrt
   subroutine mw_vssqrt_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vssqrt_2d
   subroutine mw_vssqrt_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vssqrt_s
end interface

interface vstan
   subroutine mw_vstan(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vstan
   subroutine mw_vstan_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vstan_2d
   subroutine mw_vstan_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vstan_s
end interface

interface vstanh
   subroutine mw_vstanh(z,x,n)
      real*4 z(*),x(*)
      integer*4 n
   end subroutine mw_vstanh
   subroutine mw_vstanh_2d(z,x,n)
      real*4 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vstanh_2d
   subroutine mw_vstanh_s(z,x,n)
      real*4 z,x
      integer*4 n
   end subroutine mw_vstanh_s
end interface

interface vacos
   subroutine mw_vacos(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vacos
   subroutine mw_vacos_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vacos_2d
   subroutine mw_vacos_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vacos_s
end interface

interface vacosh
   subroutine mw_vacosh(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vacosh
   subroutine mw_vacosh_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vacosh_2d
   subroutine mw_vacosh_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vacosh_s
end interface

interface vasin
   subroutine mw_vasin(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vasin
   subroutine mw_vasin_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vasin_2d
   subroutine mw_vasin_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vasin_s
end interface

interface vasinh
   subroutine mw_vasinh(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vasinh
   subroutine mw_vasinh_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vasinh_2d
   subroutine mw_vasinh_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vasinh_s
end interface

interface vatan
   subroutine mw_vatan(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vatan
   subroutine mw_vatan_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vatan_2d
   subroutine mw_vatan_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vatan_s
end interface

interface vatan2
   subroutine mw_vatan2(z,x,y,n)
      real*8 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vatan2
   subroutine mw_vatan2_2d(z,x,y,n)
      real*8 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vatan2_2d
   subroutine mw_vatan2_s(z,x,y,n)
      real*8 z,x,y
      integer*4 n
   end subroutine mw_vatan2_s
end interface

interface vatanh
   subroutine mw_vatanh(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vatanh
   subroutine mw_vatanh_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vatanh_2d
   subroutine mw_vatanh_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vatanh_s
end interface

interface vcbrt
   subroutine mw_vcbrt(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vcbrt
   subroutine mw_vcbrt_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vcbrt_2d
   subroutine mw_vcbrt_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vcbrt_s
end interface

interface vcos
   subroutine mw_vcos(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vcos
   subroutine mw_vcos_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vcos_2d
   subroutine mw_vcos_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vcos_s
end interface

interface vcosh
   subroutine mw_vcosh(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vcosh
   subroutine mw_vcosh_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vcosh_2d
   subroutine mw_vcosh_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vcosh_s
end interface

interface vcosisin
   subroutine mw_vcosisin(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vcosisin
   subroutine mw_vcosisin_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vcosisin_2d
   subroutine mw_vcosisin_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vcosisin_s
end interface

interface vdiv
   subroutine mw_vdiv(z,x,y,n)
      real*8 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vdiv
   subroutine mw_vdiv_2d(z,x,y,n)
      real*8 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vdiv_2d
   subroutine mw_vdiv_s(z,x,y,n)
      real*8 z,x,y
      integer*4 n
   end subroutine mw_vdiv_s
end interface

interface verf
   subroutine mw_verf(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_verf
   subroutine mw_verf_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_verf_2d
   subroutine mw_verf_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_verf_s
end interface

interface verfc
   subroutine mw_verfc(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_verfc
   subroutine mw_verfc_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_verfc_2d
   subroutine mw_verfc_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_verfc_s
end interface

interface vexp
   subroutine mw_vexp(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vexp
   subroutine mw_vexp_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vexp_2d
   subroutine mw_vexp_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vexp_s
end interface

interface vexp2
   subroutine mw_vexp2(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vexp2
   subroutine mw_vexp2_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vexp2_2d
   subroutine mw_vexp2_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vexp2_s
end interface

interface vexp2m1
   subroutine mw_vexp2m1(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vexp2m1
   subroutine mw_vexp2m1_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vexp2m1_2d
   subroutine mw_vexp2m1_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vexp2m1_s
end interface

interface vexpm1
   subroutine mw_vexpm1(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vexpm1
   subroutine mw_vexpm1_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vexpm1_2d
   subroutine mw_vexpm1_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vexpm1_s
end interface

interface vhypot
   subroutine mw_vhypot(z,x,y,n)
      real*8 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vhypot
   subroutine mw_vhypot_2d(z,x,y,n)
      real*8 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vhypot_2d
   subroutine mw_vhypot_s(z,x,y,n)
      real*8 z,x,y
      integer*4 n
   end subroutine mw_vhypot_s
end interface

interface vlgamma
   subroutine mw_vlgamma(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vlgamma
   subroutine mw_vlgamma_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vlgamma_2d
   subroutine mw_vlgamma_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vlgamma_s
end interface

interface vlog
   subroutine mw_vlog(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vlog
   subroutine mw_vlog_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vlog_2d
   subroutine mw_vlog_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vlog_s
end interface

interface vlog10
   subroutine mw_vlog10(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vlog10
   subroutine mw_vlog10_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vlog10_2d
   subroutine mw_vlog10_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vlog10_s
end interface

interface vlog1p
   subroutine mw_vlog1p(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vlog1p
   subroutine mw_vlog1p_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vlog1p_2d
   subroutine mw_vlog1p_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vlog1p_s
end interface

interface vlog2
   subroutine mw_vlog2(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vlog2
   subroutine mw_vlog2_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vlog2_2d
   subroutine mw_vlog2_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vlog2_s
end interface

interface vlog21p
   subroutine mw_vlog21p(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vlog21p
   subroutine mw_vlog21p_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vlog21p_2d
   subroutine mw_vlog21p_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vlog21p_s
end interface

interface vpow
   subroutine mw_vpow(z,x,y,n)
      real*8 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vpow
   subroutine mw_vpow_2d(z,x,y,n)
      real*8 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vpow_2d
   subroutine mw_vpow_s(z,x,y,n)
      real*8 z,x,y
      integer*4 n
   end subroutine mw_vpow_s
end interface

interface vqdrt
   subroutine mw_vqdrt(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vqdrt
   subroutine mw_vqdrt_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vqdrt_2d
   subroutine mw_vqdrt_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vqdrt_s
end interface

interface vrcbrt
   subroutine mw_vrcbrt(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vrcbrt
   subroutine mw_vrcbrt_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vrcbrt_2d
   subroutine mw_vrcbrt_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vrcbrt_s
end interface

interface vrec
   subroutine mw_vrec(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vrec
   subroutine mw_vrec_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vrec_2d
   subroutine mw_vrec_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vrec_s
end interface

interface vrqdrt
   subroutine mw_vrqdrt(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vrqdrt
   subroutine mw_vrqdrt_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vrqdrt_2d
   subroutine mw_vrqdrt_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vrqdrt_s
end interface

interface vrsqrt
   subroutine mw_vrsqrt(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vrsqrt
   subroutine mw_vrsqrt_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vrsqrt_2d
   subroutine mw_vrsqrt_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vrsqrt_s
end interface

interface vsin
   subroutine mw_vsin(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vsin
   subroutine mw_vsin_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsin_2d
   subroutine mw_vsin_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vsin_s
end interface

interface vsincos
   subroutine mw_vsincos(y,z,x,n)
      real*8 z(*),x(*),y(*)
      integer*4 n
   end subroutine mw_vsincos
   subroutine mw_vsincos_2d(y,z,x,n)
      real*8 z(:,:),x(:,:),y(:,:)
      integer*4 n
   end subroutine mw_vsincos_2d
   subroutine mw_vsincos_s(y,z,x,n)
      real*8 z,x,y
      integer*4 n
   end subroutine mw_vsincos_s
end interface

interface vsinh
   subroutine mw_vsinh(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vsinh
   subroutine mw_vsinh_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsinh_2d
   subroutine mw_vsinh_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vsinh_s
end interface

interface vsqrt
   subroutine mw_vsqrt(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vsqrt
   subroutine mw_vsqrt_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vsqrt_2d
   subroutine mw_vsqrt_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vsqrt_s
end interface

interface vtan
   subroutine mw_vtan(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vtan
   subroutine mw_vtan_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vtan_2d
   subroutine mw_vtan_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vtan_s
end interface

interface vtanh
   subroutine mw_vtanh(z,x,n)
      real*8 z(*),x(*)
      integer*4 n
   end subroutine mw_vtanh
   subroutine mw_vtanh_2d(z,x,n)
      real*8 z(:,:),x(:,:)
      integer*4 n
   end subroutine mw_vtanh_2d
   subroutine mw_vtanh_s(z,x,n)
      real*8 z,x
      integer*4 n
   end subroutine mw_vtanh_s
end interface

interface vpopcnt4
   integer*4 function mw_vpopcnt4(x,n)
      integer*4 x(*),n
   end function mw_vpopcnt4
   integer*4 function mw_vpopcnt4_2d(x,n)
      integer*4 x(:,:),n
   end function mw_vpopcnt4_2d
   integer*4 function mw_vpopcnt4_s(x,n)
      integer*4 x,n
      !ADD integer*4, external :: vpopcnt4
   end function mw_vpopcnt4_s
end interface

interface vpopcnt8
   integer*4 function mw_vpopcnt8(x,n)
      integer*8 x(*)
      integer*4 n
   end function mw_vpopcnt8
   integer*4 function mw_vpopcnt8_2d(x,n)
      integer*8 x(:,:)
      integer*4 n
   end function mw_vpopcnt8_2d
   integer*4 function mw_vpopcnt8_s(x,n)
      integer*8 x
      integer*4 n
      !ADD integer*4, external :: vpopcnt8
   end function mw_vpopcnt8_s
end interface


