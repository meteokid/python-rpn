#TODO: KEEP OR NOT? not used in model
stg_2_reg <- function(file_level_stg='missing',rcoef_stg=c(-1,-1),ptop_stg=-1,pref_reg=-1,rcoef_reg=-1,zoom=0,overplot=0,digits=6,p_match=100000.) {

   #===========================================================================
   #
   # Autor Andre Plante december 2008
   #
   # Language : R
   #
   # Usage : 
   #         in R type :
   #            source('stg_2_reg.r')
   #            stg_2_reg(file_level_stg='stg_levels.txt',...)
   #
   #===========================================================================

   version="1.0.0"
   print(paste('Version',version))

   #===========================================================================
   # Scale parameters in Pa if needed
   if ( ptop_stg <= 0 ) {
      print("ptop must be defined.")
      q(status=1)
   }
   if ( pref_reg <= 0 ) {
      print("pref for GEM V3 must be defined.")
      q(status=1)
   }

   pref_reg=pref_reg*100.
   pref_stg=1000.*100

   #===========================================================================
   # Read stg level list
   hyb_stg <- scan(file_level_stg,sep=",")

   nk_stg=length(hyb_stg)

   print(paste("Found",nk_stg," stg levels"))

   zeta_top=log(ptop_stg)
   zeta_srf=log(pref_stg)
   zeta=zeta_srf+log(hyb_stg)
   A_stg=zeta
   B_stg=(zeta-zeta_top)/(zeta_srf-zeta_top)
   rcoef=rcoef_stg[2]-(rcoef_stg[2]-rcoef_stg[1])*B_stg
   for (k in 1:nk_stg){
      B_stg[k]=B_stg[k]^rcoef[k]
   }

   #===========================================================================
   # Get hyb_reg to fit levels with stg one for the pressure below the hill

   # Note, the reg top is at the first momentum level for the highest surface pressure 
   # in the domain set to 100000. here. therefore output from the stg model at ptop_reg
   # will never be an extrapolation.
   hybtop_reg=(exp(A_stg[1]+B_stg[1]*log(100000./pref_stg)))/pref_reg

   nk_reg=nk_stg+1
   hyb_reg=seq(1,1,nk_reg)

   hyb_reg[k+1]=hybtop_reg
   
   for (k in 2:nk_reg-1){
      guess=max(hybtop_reg,hyb_stg[k-1])
      hyb_reg[k]=New_Raph_Hyb(p_match,pref_reg,guess,A_stg[k],B_stg[k],rcoef_reg,hybtop_reg)
   }
   hyb_reg[nk_reg]=1.

   #===========================================================================
   # Compute hyb A and B regular grid

   B_reg=((hyb_reg-hybtop_reg)/(1.-hybtop_reg))^rcoef_reg
   A_reg=(hyb_reg-B_reg)*pref_reg
   ptop_reg=ptop_stg

   file_out='out_stg_2_reg.txt'
   write('Use the following parameter in your gem_settings',file_out, append = FALSE)
   write('&gem_cfgs',file_out, append = TRUE)
   write(paste('Pres_pref=',pref_reg/100.),file_out, append = TRUE)
   write(paste('Grd_rcoef=',rcoef_reg),file_out, append = TRUE)
   write('hyb=',file_out, append = TRUE)
   for ( k in 1:nk_reg ){
      write(paste(format(hyb_reg[k],digits=digits),','),file_out, append = TRUE)
   }	
   write('/',file_out, append = TRUE)

}
#=============================================================
New_Raph_Hyb <- function(p_match,pref,guess,A,B,rcoef,hybtop) {

   # Methode de Newton-Raphson

   cst_a=(p_match-pref)/pref
   cst_b=exp(A+B*log(p_match/100000.))/pref

   diff=1

   val=guess

   niter=0

   while ( diff > .000001 & niter < 30 ) {

      ff=val+cst_a*((val-hybtop)/(1.-hybtop))^rcoef-cst_b
      fp=1.+cst_a*rcoef*(val-hybtop)^(rcoef-1)/(1.-hybtop)^rcoef
      val2=val-ff/fp        
	
      diff=abs(val-val2)/val2

      val=val2
      niter=niter+1

   }

   return=val
}
