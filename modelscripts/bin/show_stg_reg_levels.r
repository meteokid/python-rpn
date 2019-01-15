show_stg_reg_levels <- function(
   title='Levels',
   file_levels='missing',
   file_rcoef='missing',
   file_ptop='missing',
   gbpil_t=-1,
   file_levels_3='missing',
   file_rcoef_3='missing',
   file_pref_3='missing', 
   file_ptop_3='missing', 
   p_below_hill=100000.,
   p_top_hill=50000.,
   ptop_graph=-1,
   pbot_graph=-1,
   zoom=0,
   overplot=0,
   plot_thermo=0) {

   #===========================================================================
   #
   # Autor Andre Plante dec 2009
   #
   # Language : R
   #
   #===========================================================================

   version="1.0.0"
   print(paste('Version',version))

   #===========================================================================
   postscript("out.ps", bg='white', horizontal=FALSE, width=7.5, height=10., pointsize=10, paper = "letter")

   #==============================================================================
   # Load libraries
   library(grid)

   #===========================================================================
   # Define montain, height is in Pa

   if(pbot_graph == -1){pbot_graph=max(p_below_hill,p_top_hill)}

   ni=51
   xx=seq(0,ni-1,1)/(ni-1)*2*pi

   ps=p_below_hill-(.5*(1-cos(xx))*(p_below_hill-p_top_hill))

   #===========================================================================
   # Read current levels

   hyb <- scan(file_levels,sep=",")

   nk=length(hyb)

   print(paste("Found",nk," levels"))
   
   # Checking parameters
   
   if ( hyb[nk] == 1. ){
      print('Invalide level at nk, it is equal to 1.0')
      q(status=1)
   }

   #===========================================================================

   if ( file_ptop == 'missing' ){
      print("key file_ptop nust be defined.")
      q(status=1)
   }
   ptop <- scan(file_ptop)
   
   pref=1000.*100.
   zeta_srf=log(pref)
   if(ptop == -2){
      # Make first level flat
      zeta_top=zeta_srf+log(hyb[1])
      ptop=exp(.5*( 3.*(log(hyb[1]*100000.)) - log(hyb[2]*100000.)))
      print(paste('Ptop computed ->',ptop))
   } else {
      # Only top is flat
      zeta_top=log(ptop)
      print(paste('Using ptop=',ptop,'Pa'))
   }

   if ( ptop_graph == -1 ){ptop_graph=ptop}

   #===========================================================================

   if ( file_rcoef == 'missing' ){
      print("key file_rcoef nust be defined.")
      q(status=1)
   }
   rcoef <- scan(file_rcoef,sep=",")
   print(paste('Using rcoef',rcoef[1],', ',rcoef[2]))

   #===========================================================================

   zeta=zeta_srf+log(hyb)
   A=zeta
   B=(zeta-zeta_top)/(zeta_srf-zeta_top)

   rcoef_=rcoef[2]-(rcoef[2]-rcoef[1])*B
   for (k in 1:nk){
      B[k]=B[k]^rcoef_[k]
   }

   # Compute stg level pressure
   lnpi=matrix(data=0,nrow=ni,ncol=nk)

   for (i in 1:ni){
      Ss=log(ps[i]/pref)
      for (k in 1:nk){
         lnpi[i,k]=A[k]+B[k]*Ss
      }
   }

   #===========================================================================
   # V3 levels
   hyb_3 <- scan(file_levels_3,sep=",")

   nk_3=length(hyb_3)

   print(paste("Found",nk_3," levels for version 3"))
   
   # Checking parameters

   if ( file_rcoef_3 == 'missing' ){
      print("key file_rcoef_3 nust be defined.")
      q(status=1)
   }
   rcoef_3 <- scan(file_rcoef_3,sep=",")
   print(paste('Using rcoef=',rcoef_3,' for version 3'))

   #===========================================================================

   if ( file_pref_3 == 'missing' ){
      print("key file_pref_3 nust be defined.")
      q(status=1)
   }
   pref_3 <- scan(file_pref_3,sep=",")
   pref_3=pref_3*100.
   print(paste('Using pref=',pref_3,' for version 3'))

   #===========================================================================

   if ( file_ptop_3 == 'missing' ){
      print("key file_ptop_3 nust be defined.")
      q(status=1)
   }
   ptop_3 <- scan(file_ptop_3,sep=",")

   #===========================================================================

   if ( hyb_3[1] == 0. ){
      print("Normalised levels detected")
      if(ptop_3 == -1 ){
         print(paste("ptop must be define but is ",ptop_3))
         q(status=1)
      }
      ptop_3=ptop_3*100.
      hybtop_3=ptop_3/pref_3
      hyb_3=hybtop_3+hyb_3*(1.-hybtop_3)
   } else {
      print("Un normalised levels detected")
   }

   ptop_3=hyb_3[1]*pref_3
   print(paste('Using ptop=',ptop_3,' for version 3'))

   if ( hyb_3[1] == 0. ){
      print('Invalide level at k=1, it is equal to .0, levels should be non-normalized, abort')
      q(status=1)
   }
   if ( hyb_3[nk_3] != 1. ){
      print('Invalide level at nk, it is not equal to 1., abort')
      q(status=1)
   }
  	
   #===========================================================================


   B_3=((hyb_3-hyb_3[1])/(1.-hyb_3[1]))^rcoef_3
   A_3=(hyb_3-B_3)*pref_3

   # Compute reg level pressure
   lnpi_3=matrix(data=0,nrow=ni,ncol=nk_3)
   pi_3=matrix(data=0,nrow=ni,ncol=nk_3)

   for (i in 1:ni){
      for (k in 1:nk_3){
	 pi_3[i,k]=A_3[k]+B_3[k]*ps[i]
      }
   }
   lnpi_3=log(pi_3)

   #===========================================================================
   # Plot two graph side by side

   #--------------
   # Espace totale
   #--------------
   vp0 <- viewport(x = 0.5, y = 0.5,
         width = 1, height = 1,
         just = "centre", 
         gp = gpar(), clip = "off",
         xscale =c(0,1), yscale =c(0,1),
         angle = 0,
         layout = NULL,
         layout.pos.row = NULL, layout.pos.col = NULL,
         name = NULL,
	)
   #@@@@@@@@@@@@@@@@
   pushViewport(vp0)
   #@@@@@@@@@@@@@@@@

   grid.text(title,x=.5,y=.98,default.units = "npc",gp = gpar(fontsize=18),just=c("centre","top"))
   
   grid.text(paste("ptop = ",sprintf("%10.2f",ptop)," pa",sep=""),x=.3,y=.94,default.units = "npc",gp = gpar(fontsize=14),just=c("centre","top"))
   my_strg=paste("rcoefs = ",rcoef[1],',',rcoef[2],sep="")
   if( gbpil_t != -1 ){
      my_strg=paste(my_strg,' gbpil_t=',gbpil_t,sep="")
   }
   grid.text(my_strg,x=.3,y=.91,default.units = "npc",gp = gpar(fontsize=14),just=c("centre","top"))
   yyy=.03
   grid.lines(x=c(.4,.5),y=c(yyy,yyy),gp=gpar(col='green'),default.units="npc")
   grid.text(' Top/Bot',x=.5,y=yyy,default.units = "npc",gp = gpar(fontsize=12,col='darkgreen'),just=c("left"))

   #===========================================================================
   # Plot pressure levels for the current model

   nih=floor(ni/2)+1

   xrange=c(xx[1],xx[ni])   
   if ( zoom == 0 ) {
      # Pas de zoom
      yrange=c(log(pbot_graph),log(ptop_graph))
   } else if ( zoom == 1 ){
      # Zoom en bas de la montagne
      yrange=c(log(p_below_hill),log(700*100.))
   } else if ( zoom == 2 ){
      # Zoom en haut de la montagne
      yrange=c(log(505*100.),log(300*100.))
   } else  if ( zoom == 3 ){
      # Zoom demi bas
      yrange=c(log(505*100.),log(20*100.))
   } else {
      print('Zoom non defini')
      q(status=1)
   }

   #--------------
   # Levels graphe
   #--------------

   vp1 <- viewport(x = .5, y = 0.45,
         width = .4, height = .8,
         just = "right", 
         gp = gpar(), clip = "off",
         xscale=xrange,yscale=yrange,
         angle = 0,
         layout = NULL,
         layout.pos.row = NULL, layout.pos.col = NULL,
         name = NULL,
	)

   #@@@@@@@@@@@@@@@@
   pushViewport(vp1)
   #@@@@@@@@@@@@@@@@
   #grid.rect()

   titre="ln(pi)=A+B*ln(pis/100000.)"
   grid.text(titre,x=.5,y=1.01,default.units = "npc",gp = gpar(fontsize=14),just=c("centre","bottom"))
   grid.text('Pressure [hPa]',x=-.18,y=.5,default.units = "npc",gp = gpar(fontsize=10),rot=90)

   decade=c(100,70,50,35,25,15)

   tempo=c(1000,950,900,850,800,700,600,500,420,350,300,250,200,160,130,decade,decade*.1,decade*.01,decade*.001)
   for (i in 1:length(tempo)){
      if(tempo[i]*100.<ptop_graph){break}
   }
   yticks=tempo[1:(i-1)]   

   grid.yaxis(at=log(yticks*100.),label=yticks,gp=gpar(fontsize=9))

   for (k in 1:nk){
      tmp=max(exp(lnpi[1:ni,k]))
      if ( tmp >= ptop_graph & tmp <= pbot_graph ){
         grid.lines(x=xx,y=lnpi[1:ni,k],default.units="native")
      }
   }

   grid.lines(x=xx,y=log(ps),gp=gpar(col='green'),default.units="native")
   if ( gbpil_t != -1 ){
      yyy=yyy-.04
      grid.lines(x=c(.25,.4),y=c(yyy,yyy),gp=gpar(col='orange'),default.units="npc")
      grid.text(' Last piloted level',x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=12,col='orange'),just=c("left"))
      grid.lines(x=xx,y=lnpi[1:ni,gbpil_t],gp=gpar(col='orange'),default.units="native")
   }

   lnpi_t=matrix(data=0,nrow=ni,ncol=nk+1)	


   # Staggered thermo levels

   if ( plot_thermo == 1 ){

      yyy=.01
      grid.lines(x=c(.25,.4),y=c(yyy,yyy),gp=gpar(col='blue',lty=2),default.units="npc")
      grid.text(' Thermo',x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=12,col='blue'),just=c("left"))

      for (k in 1:(nk-1)){
         lnpi_t[1:ni,k+1]=.5*( lnpi[1:ni,k+1] + lnpi[1:ni,k] )
      }
      lnpi_t[1:ni,1]   =.5*( lnpi[1:ni,1] + (seq(1,ni,1)*0.+log(ptop)) )
      lnpi_t[1:ni,nk+1]=.5*( log(ps)      + lnpi[1:ni,nk]              )
   
      for (k in 1:(nk+1)){
         tmp=max(exp(lnpi_t[1:ni,k]))
         if ( tmp >= ptop_graph & tmp <= pbot_graph ){
            grid.lines(x=xx,y=lnpi_t[1:ni,k],gp=gpar(col='blue',lty=2),default.units="native")
         }
      }
   }

   grid.lines(x=xx,y=log(ptop),gp=gpar(col='green'),default.units="native")

   #@@@@@@@@@@@
   upViewport()
   #@@@@@@@@@@@

   grid.text(paste("ptop = ",ptop_3," pa",sep=""),x=.7,y=.94,default.units = "npc",gp = gpar(fontsize=14),just=c("centre","top"))
   grid.text(paste("rcoefs = ",rcoef_3,sep=""),x=.7,y=.91,default.units = "npc",gp = gpar(fontsize=14),just=c("centre","top"))

   #-----------------
   # Levels graphe v3
   #-----------------

   vp2 <- viewport(x = .5, y = 0.45,
         width = .4, height = .8,
         just = "left", 
         gp = gpar(), clip = "off",
         xscale=xrange,yscale=yrange,
         angle = 0,
         layout = NULL,
         layout.pos.row = NULL, layout.pos.col = NULL,
         name = NULL,
	)

   #@@@@@@@@@@@@@@@@
   pushViewport(vp2)
   #@@@@@@@@@@@@@@@@
   #grid.rect()
   grid.lines(c(0,0),c(0,1))
   grid.lines(c(1,1),c(0,1))

   titre="pi=A+B*pis"
   grid.text(titre,x=.5,y=1.01,default.units = "npc",gp = gpar(fontsize=14),just=c("centre","bottom"))

   for (i in 1:length(tempo)){
      if(tempo[i]*100.<ptop_graph){break}
   }
   yticks=tempo[1:(i-1)]   

   for (k in 1:nk_3){
      if ( max(pi_3[1:ni,k])  >= ptop_graph & max(pi_3[1:ni,k])  <= pbot_graph ){
         grid.lines(x=xx,y=lnpi_3[1:ni,k],default.units="native")
      }
   }

   grid.lines(x=xx,y=log(ps),gp=gpar(col='green'),default.units="native")

   grid.lines(x=xx,y=log(ptop_3),gp=gpar(col='green'),default.units="native")

   if ( overplot == 1 ){
      for (k in 1:nk){
         grid.lines(x=xx,y=lnpi[1:ni,k],default.units="native",gp=gpar(col='red'))
      }    
   }

}
