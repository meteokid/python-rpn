reg_2_stg <- function(file_level_reg='missing',rcoef_reg=-1,ptop_reg=-1,pref_reg=-1,rcoef_stg=c(1,1),zoom=0,overplot=0,digits=6,p_match=100000.) {

   #===========================================================================
   #
   # Autor Andre Plante december 2008
   #
   # Language : R
   #
   # Usage : 
   #         in R type :
   #            source('reg_2_stg.r')
   #            reg_2_stg(file_level_reg='reg_levels.txt',...)
   #
   #===========================================================================

   version="1.0.0"
   print(paste('Version',version))

   #===========================================================================
   # Scale parameters in Pa
   if ( pref_reg <= 0 ) {
      print("pref must be defined.")
      q(status=1)
   }
   if ( ptop_reg <= 0 ) {
      hyb_reg <- scan(file_level_reg,sep=",")
      if(hyb_reg[1] == 0 ){
         print("ptop must be defined.")
         q(status=1)
      }
      ptop_reg=hyb_reg[1]*pref_reg
      print(paste('NOTE: ptop was comuted from hyb(1) and pref and equals',ptop_reg,'hPa'))
   }

   ptop_reg=ptop_reg*100.
   pref_reg=pref_reg*100.

   #===========================================================================
   # Read regular level list
   hyb_reg <- scan(file_level_reg,sep=",")

   nk_reg=length(hyb_reg)

   print(paste("Found",nk_reg," regular levels"))
   
   # Checking parameters
   
   if ( hyb_reg[1] == 0 ){
      print("Normalised levels detected")
      hybtop_reg=ptop_reg/pref_reg
      hyb_reg=hybtop_reg+hyb_reg*(1.-hybtop_reg)
   } else {
      print("Un normalised levels detected")
   }

   #===========================================================================
   # Compute hyb A and B regular grid

   hybtop_reg=ptop_reg/pref_reg
   B_reg=((hyb_reg-hybtop_reg)/(1.-hybtop_reg))^rcoef_reg
   A_reg=(hyb_reg-B_reg)*pref_reg

   #===========================================================================
   # Get hyb_stg to fit levels with reg ones for the pressure below the hill   

   if (length(rcoef_stg) != 2 ){
      print(paste('rcoef must have two values, got only',rcoef_stg))
      q(status=1)
   }

   nk_stg=nk_reg-1
   zeta=seq(1,nk_stg,1)*0.-1.

   pref_stg=1000.*100.
   # For the stg levels, we add a half laver on top of the reg grid so that both momentum and thermo 
   # levels are at or above the reg model top
   # lnPt = lnp1 - .5*(lnp2-lnp1) where p1 p2 are the reg levels, we take a surface pressure of 100000.	
   p1=A_reg[1] + B_reg[1]*100000.
   p2=A_reg[2] + B_reg[2]*100000.	

   ptop_stg=exp(log(p1)-.5*(log(p2/p1)))

   zeta_top=log(ptop_stg)
   zeta_srf=log(pref_stg)

   gem_s=log(p_match)-zeta_srf

   if( rcoef_stg[1] == 1 ){
      for (k in 1:nk_stg){
         zeta[k]=(log(A_reg[k]+B_reg[k]*p_match)*(zeta_srf-zeta_top)+gem_s*zeta_top)/(zeta_srf-zeta_top+gem_s)
      }
   }else if (rcoef_stg[1] == rcoef_stg[2]){
      for (k in 1:nk_stg){
         guess=max(zeta_top,zeta[k])
         zeta[k]=New_Raph_Zeta(p_match,zeta_srf,zeta_top,guess,A_reg[k],B_reg[k],rcoef_stg[1])
      }
   } else {
      if ( rcoef_stg[1] != rcoef_stg[2] ){
         print(paste('rcoef_stg[1] and rcoef_stg[2] must be the same for this version: got',rcoef_stg[1],'and',rcoef_stg[2]))
         q(status=1)
      }
   }
	
   hyb_stg=exp(zeta-zeta_srf)

   file_out='out_reg_2_stg.txt'
   write('Use the following parameter in your gem_settings',file_out, append = FALSE)
   write('&gem_cfgs',file_out, append = TRUE)
   write(paste('Cstv_ptop_8=',format(ptop_stg,digits=digits)),file_out, append = TRUE)
   write(paste('Grd_rcoef=',rcoef_stg[1],',',rcoef_stg[2]),file_out, append = TRUE)
   write('hyb=',file_out, append = TRUE)
   for ( k in 1:nk_stg ){
      write(paste(format(hyb_stg[k],digits=digits),','),file_out, append = TRUE)
   }	
   write('/',file_out, append = TRUE)

}
#=============================================================
New_Raph_Zeta <- function(p_match,zeta_srf,zeta_top,guess,A,B,rcoef) {

   # Methode de Newton-Raphson

   gem_s=log(p_match)-zeta_srf
   cst_b=log(A+B*p_match)

   diff=1

   val=guess

   niter=0

   den=(zeta_srf-zeta_top)^rcoef

   while ( diff > .000001 & niter < 30 ) {

      ff=val+gem_s*(val-zeta_top)^rcoef/den-cst_b
      fp=1.+gem_s*rcoef*(val-zeta_top)^(rcoef-1)/den
      val2=val-ff/fp       
	
      diff=abs(val-val2)/val2

      val=val2
      niter=niter+1

   }

   return=val
}
