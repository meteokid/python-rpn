#TODO: KEEP OR NOT? not used in model
show_stg_levels <- function(
   title='Levels',
   file1_levels='missing',
   file1_rcoef='missing',
   file1_ptop='missing',
   sets=0,
   gbpil_t1=-1,
   file2_levels='missing',
   file2_rcoef='missing',
   file2_ptop='missing',
   gbpil_t2=-1,
   p_below_hill=100000.,
   p_top_hill=50000.,
   ptop_graph=-1,
   pbot_graph=-1,
   xrange_delta=c(0,0),
   thickness=0,
   overplot=0,
   zoom=0,
   plot_thermo=1,
   font_f=1,
   width=7.5,height=10.,
   format='ps'
   ) {

   #===========================================================================
   #
   # Autor Andre Plante dec 2009, version 0.0.0
   #
   # Language : R
   #
   # 1.0.0 A. Plante jul 2010, add possibility to plot two graph from gem4
   #                           side by side
   # 1.1.0 A. Plante Otc 2010, add Boundary condition on temp if lid nesting
   # 1.2.0 A. Plante Jun 2013, add font_f, width, height and format arguments
   #                           Improve representation by increasing ni
   # 1.3.0 A. Plante April 2015, add support to ptop = -2 -> B(1)=0
   #===========================================================================

   version="1.3.0"
   print(paste('Version',version))

   #=========================================================================== 
   if(format == 'eps'){
      postscript("out.eps", bg='white', horizontal=FALSE, width=width, height=height, pointsize=20, , paper = "special")
   } else if (format == 'ps' ){
      postscript("out.ps", bg='white', horizontal=FALSE, width=width, height=height, pointsize=20)
   } else {
      print("ERROR with format, must be 'ps' or 'eps'")
      q()
   }	
   #png("out.png", bg='white', width=800, height=1000, pointsize=10)

   #==============================================================================
   # Load libraries
   library(grid)

   #===========================================================================
   # Define montain, height is in Pa

   if(pbot_graph == -1){pbot_graph=max(p_below_hill,p_top_hill)}

   ni=551
   xx=seq(0,ni-1,1)/(ni-1)*2*pi

   ps=p_below_hill-(.5*(1-cos(xx))*(p_below_hill-p_top_hill))

   ppp=list(p_below_hill=p_below_hill,
            p_top_hill  =p_top_hill,
            ptop_graph  =ptop_graph,
            pbot_graph  =pbot_graph,
            xrange_delta=xrange_delta,
            thickness   =thickness,
            zoom        =zoom,
            plot_thermo =plot_thermo)

   #===========================================================================

   pref=1000.*100.
   if ( sets == 0 ){
      print("key sets must be provided (1,2)")
      q(status=1)
   }

   if ( file1_levels == 'missing' ){
      print("key file1_levels must be defined.")
      q(status=1)
   }
   if ( file1_ptop == 'missing' ){
      print("key file1_ptop must be defined.")
      q(status=1)
   }
   if ( file1_rcoef == 'missing' ){
      print("key file1_rcoef must be defined.")
      q(status=1)
   }
   print('Reading parameters 1')
   prm1=read_param(file1_levels,file1_ptop,file1_rcoef,gbpil_t1)
   print('Computing pressure levels for parameters 1')
   res=get_lnpi(ps,prm1,pref)

   lnpi1=res$lnpi
   prm1$ptop=res$ptop
   
   if ( sets == 2 ){
      if ( file2_levels == 'missing' ){
         print("key file2_levels must be defined.")
         q(status=1)
      }
      if ( file2_ptop == 'missing' ){
         print("key file2_ptop must be defined.")
         q(status=1)
      }
      if ( file2_rcoef == 'missing' ){
         print("key file2_rcoef nust be defined.")
         q(status=1)
      }
      print('Reading parameters 2')
      prm2=read_param(file2_levels,file2_ptop,file2_rcoef,gbpil_t2)

      print('Computing pressure levels for parameters 2')
      res=get_lnpi(ps,prm2,pref)
      lnpi2=res$lnpi
      prm2$ptop=res$ptop

      if ( ppp$ptop_graph == -1 ){ppp$ptop_graph=min(prm1$ptop,prm2$ptop)}

   }

   if ( sets == 1 ){
      if ( ppp$ptop_graph == -1 ){ppp$ptop_graph=min(prm1$ptop)}
   } else {
      if ( ppp$ptop_graph == -1 ){ppp$ptop_graph=min(prm1$ptop,prm2$ptop)}
   }

   #===========================================================================
   # Plot two graph side by side

   if ( zoom == 0 ) {
      # Pas de zoom
      yrange=c(log(pbot_graph),log(ppp$ptop_graph))
   } else if ( zoom == 1 ){
      # Zoom en bas de la montagne
      yrange=c(log(ppp$p_below_hill),log(700*100.))
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

   # Define y ticks
   decade=c(100,70,50,35,25,15)
   tempo=c(1000,950,900,850,800,700,600,500,420,350,300,250,200,160,130,decade,decade*.1,decade*.01,decade*.001)
   for (i in 1:length(tempo)){
      if(tempo[i]*100.<ppp$ptop_graph){break}
   }
   imin=i-1
   for (i in seq(imin,1,-1)){
      if(tempo[i]*100.>pbot_graph){break}
   }   
   yticks=tempo[i:imin]   

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

   #grid.text(title,x=.5,y=.98,default.units = "npc",gp = gpar(fontsize=font_f*18),just=c("centre","top"))
   titre="ln(pi)=A+B*ln(pis/100000.)"
   #grid.text(titre,x=.5,y=.95,default.units = "npc",gp = gpar(fontsize=font_f*14),just=c("centre","top"))

   
   #===========================================================================
   # Plot pressure levels

   if ( sets == 2 ){
      plot_levels('left' ,ps,xx,zoom,prm1,lnpi1,yrange,yticks,ppp,font_f=font_f)
      plot_levels('right',ps,xx,zoom,prm2,lnpi2,yrange,yticks,ppp,font_f=font_f)
      if(overplot == 1 ){
         plot_levels('right',ps,xx,zoom,prm1,lnpi1,yrange,yticks,ppp,overplot=1,font_f=font_f)
      }
   } else {
      plot_levels('left' ,ps,xx,zoom,prm1,lnpi1,yrange,yticks,ppp,space_centre=.1,font_f=font_f)
      plot_delta(ps,xx,zoom,prm1,lnpi1,yrange,yticks,ppp,space_centre=.1,font_f=font_f)
   }

}
#=========================================================================
#=========================================================================
#=========================================================================
get_lnpi <- function(ps,prm,pref){

   zeta_srf=log(pref)
   if(prm$ptop == -2){
      # Make first level flat
      zeta_top=zeta_srf+log(prm$hyb[1])
      prm$ptop=exp(.5*( 3.*(log(prm$hyb[1]*100000.)) - log(prm$hyb[2]*100000.)))
      print(paste('Ptop computed ->',prm$ptop))
   } else {
      # Only top is flat
      zeta_top=log(prm$ptop)
   }
   zeta=zeta_srf+log(prm$hyb)
   A=zeta
   B=(zeta-zeta_top)/(zeta_srf-zeta_top)

   ni=length(ps)
   nk=length(prm$hyb)

   rcoef_=prm$rcoef[2]-(prm$rcoef[2]-prm$rcoef[1])*B
   for (k in 1:nk){
      if ( B[k] < 0. | B[k] > 1. ){
         print('ERROR with hyb or Cstv_ptop_8, look at B we got with these below, B should be in range 0 <= B <= 1')
         print(B)
         q(status=1)
      }
      B[k]=B[k]^rcoef_[k]
   }

   # Compute stg level pressure
   ff=matrix(data=0,nrow=ni,ncol=nk)

   for (i in 1:ni){
      Ss=log(ps[i]/pref)
      for (k in 1:nk){
         ff[i,k]=A[k]+B[k]*Ss
      }
   }
   get_lnpi=list(lnpi=ff,ptop=prm$ptop)
	
}
#=========================================================================
#=========================================================================
#=========================================================================
read_param <- function(file_levels,file_ptop,file_rcoef,gbpil_t){

   # Read regular level list
   hyb <- scan(file_levels,sep=",")

   nk=length(hyb)

   print(paste("Found",nk," levels"))
   
   # Checking parameters
   
   if ( hyb[nk] == 1. ){
      print('Invalide level at nk, it is equal to 1.0')
      q(status=1)
   }

   ptop <- scan(file_ptop)
   if(ptop == -2){
      print('Ptop will be computed, first momentum level will be flat')
   } else {
      print(paste('Using ptop=',ptop,'Pa'))
   }

   rcoef <- scan(file_rcoef,sep=",")
   print(paste('Using rcoef',rcoef[1],', ',rcoef[2]))

   read_param=list(nk=nk,hyb=hyb,ptop=ptop,rcoef=rcoef,gbpil_t=gbpil_t)

}
#=========================================================================
#=========================================================================
#=========================================================================
plot_levels <- function(side,ps,xx,zoom,prm,lnpi,yrange,yticks,ppp,overplot=0,space_centre=0,font_f=font_f){

   ni=length(ps)

   nih=floor(ni/2)+1

   xrange=c(xx[1],xx[ni])   

   if ( overplot == 0 ){
      color='black'
   } else {
      color='red'
   }

   #--------------
   # Levels graphe
   #--------------

   if(side == 'left'){
      vp1 <- viewport(x = .5-space_centre/2, y = 0.48,
            width = .35, height = .75,
            just = "right", 
            gp = gpar(), clip = "off",
            xscale=xrange,yscale=yrange,
            angle = 0,
            layout = NULL,
            layout.pos.row = NULL, layout.pos.col = NULL,
            name = NULL,
	   )
   } else {
      vp1 <- viewport(x=.5+space_centre/2,y=.48,
            width = .35, height = .75,
            just = "left", 
            gp = gpar(), clip = "off",
            xscale=xrange,yscale=yrange,
            angle = 0,
            layout = NULL,
            layout.pos.row = NULL, layout.pos.col = NULL,
            name = NULL,
	   )
   }

   vp1.1 <- viewport(x = 1., y = .5,
         width = 1., height = 1.,
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

   if ( overplot == 0 ){

      grid.text(paste("ptop = ",sprintf("%10.2f",prm$ptop)," pa",sep=""),x=.5,y=1.07,default.units = "npc",gp = gpar(fontsize=font_f*14),just=c("centre","top"))
      my_strg=paste("rcoefs = ",prm$rcoef[1],',',prm$rcoef[2],sep="")
      if( prm$gbpil_t != -1 ){
         my_strg=paste(my_strg,' gbpil_t=',prm$gbpil_t,sep="")
      }
      grid.text(my_strg,x=.5,y=1.04,default.units = "npc",gp = gpar(fontsize=font_f*14),just=c("centre","top"))   

   }

   if(side == 'left'){

      grid.text('Pressure [hPa]',x=-.18*font_f,y=.5,default.units = "npc",gp = gpar(fontsize=font_f*10),rot=90)
      grid.yaxis(at=log(yticks*100.),label=yticks,gp=gpar(fontsize=font_f*9))

   }

   #@@@@@@@@@@@@@@@@
   pushViewport(vp1.1)
   #@@@@@@@@@@@@@@@@

   for (k in 1:prm$nk){
      grid.lines(x=xx,y=lnpi[1:ni,k],gp=gpar(col=color),default.units="native")
   }

   grid.lines(x=xx,y=log(ps),gp=gpar(col='green'),default.units="native")
   grid.lines(x=c(xx[1 ],xx[1 ]),y=yrange,default.units="native")
   grid.lines(x=c(xx[ni],xx[ni]),y=yrange,default.units="native")

   #@@@@@@@@@@@
   upViewport()
   #@@@@@@@@@@@

   lnpi_t=matrix(data=0,nrow=ni,ncol=prm$nk+1)	

   if ( overplot == 0 ){
      yyy=-.035
      grid.lines(x=c(.25,.4),y=c(yyy,yyy),default.units="npc")
      grid.text(paste(' ',prm$nk,' momentum'),x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12),just=c("left"))
      yyy=yyy-.03
      grid.lines(x=c(.25,.4),y=c(yyy,yyy),gp=gpar(col='green'),default.units="npc")
      grid.text(' Top/Bot',x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12,col='darkgreen'),just=c("left"))
      if ( prm$gbpil_t != -1 ){
         yyy=yyy-.03
         if ( ppp$plot_thermo == 1 & overplot == 0 ){
            grid.lines(x=c(0,.25),y=c(yyy,yyy),gp=gpar(col='orange',lty=2),default.units="npc")
            grid.lines(x=c(.25,.4),y=c(yyy,yyy),gp=gpar(col='orange'),default.units="npc")
            grid.text(' Last piloted levels',x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12,col='orange'),just=c("left"))            
         } else {
            grid.lines(x=c(.25,.4),y=c(yyy,yyy),gp=gpar(col='orange'),default.units="npc")
            grid.text(' Last piloted level',x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12,col='orange'),just=c("left"))
         }
      }
   }

   # Staggered thermo levels

   if ( ppp$plot_thermo == 1 & overplot == 0 ){

      yyy=yyy-.03
      grid.lines(x=c(.25,.4),y=c(yyy,yyy),gp=gpar(col='blue',lty=2),default.units="npc")
      grid.text(' Thermo',x=.4,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12,col='blue'),just=c("left"))

      for (k in 1:(prm$nk-1)){
         lnpi_t[1:ni,k+1]=.5*( lnpi[1:ni,k+1] + lnpi[1:ni,k] )
      }
      lnpi_t[1:ni,1]   =.5*( lnpi[1:ni,1] + (seq(1,ni,1)*0.+log(prm$ptop)) )
      lnpi_t[1:ni,prm$nk+1]=.5*( log(ps)      + lnpi[1:ni,prm$nk]              )
   
      #@@@@@@@@@@@@@@@@
      pushViewport(vp1.1)
      #@@@@@@@@@@@@@@@@
      for (k in 1:(prm$nk+1)){
         grid.lines(x=xx,y=lnpi_t[1:ni,k],gp=gpar(col='blue',lty=2),default.units="native")
      }
      if ( prm$gbpil_t != -1 ){
         grid.lines(x=xx,y=lnpi_t[1:ni,prm$gbpil_t+1],gp=gpar(col='orange',lty=2),default.units="native")
         grid.text('B.C. on T-> ',x=xx[1],y=lnpi_t[1,prm$gbpil_t+1],default.units="native",just=c("right","center"),gp = gpar(fontsize=font_f*9))
      }
      #@@@@@@@@@@@
      upViewport()
      #@@@@@@@@@@@
	
   }

   if ( overplot == 0 ){
      grid.lines(x=xx,y=log(prm$ptop),gp=gpar(col='green'),default.units="native")
      if ( prm$gbpil_t != -1 ){
          grid.lines(x=xx,y=lnpi[1:ni,prm$gbpil_t],gp=gpar(col='orange'),default.units="native")
      }
   }


   #@@@@@@@@@@@
   upViewport()
   #@@@@@@@@@@@

}
#=========================================================================
#=========================================================================
#=========================================================================
plot_delta <- function(ps,xx,zoom,prm,lnpi,yrange,yticks,ppp,space_centre=.1,font_f=font_f){

   # Plot delta pressure

   nk=prm$nk
   ni=length(ps)
   nih=floor(ni/2)+1

   if ( ppp$thickness == 0 ){
      delta_sea=lnpi[1  ,2:nk]-lnpi[1  ,1:(nk-1)]
      delta_top=lnpi[nih,2:nk]-lnpi[nih,1:(nk-1)]
   } else {
      lnpi_avg=seq(1,nk+1,1)
      lnpi_avg[2:nk]=.5*(lnpi[1  ,2:nk]+lnpi[1  ,1:(nk-1)])
      lnpi_avg[1]=log(prm$ptop)
      lnpi_avg[nk+1]=log(ps[1])   
      delta_sea=seq(1,nk,1)
      delta_sea=16000/log(10)*(lnpi_avg[2:(nk+1)]-lnpi_avg[1:nk])

      lnpi_avg[2:nk]=.5*(lnpi[nih  ,2:nk]+lnpi[nih  ,1:(nk-1)])
      lnpi_avg[1]=log(prm$ptop)
      lnpi_avg[nk+1]=log(ps[nih])
      delta_top=seq(1,nk,1)
      delta_top=16000/log(10)*(lnpi_avg[2:(nk+1)]-lnpi_avg[1:nk])
   }

   if ( ppp$xrange_delta[1] == 0 & ppp$xrange_delta[2] == 0 ){
      # Find highest level in graph
      for (k in 1:(nk-1)){
         if ( exp(lnpi[1,k]) > ppp$ptop_graph ) {break}
      }
      km=k	
      for (k in (nk-1):1){
         if ( exp(lnpi[nih,k]) < ppp$pbot_graph ) {break}
      }
      kp=k
      pmin=min(delta_top[km:kp])
      pmax=max(delta_sea[km:kp])
      delta=pmax-pmin
      pmin=pmin-.1*delta
      pmax=pmax+.1*delta
      ppp$xrange_delta=c(pmin,pmax)
   }

   #@@@@@@@@@@@
   upViewport()
   #@@@@@@@@@@@

   vp2 <- viewport(x=.5+space_centre/2,y=.48,
         width = .4, height = .75,
         just = "left", 
         gp = gpar(), clip = "off",
         xscale=ppp$xrange_delta,yscale=yrange,
         angle = 0,
         layout = NULL,
         layout.pos.row = NULL, layout.pos.col = NULL,
         name = NULL,
	)

   vp2.1 <- viewport(x = 1., y = .5,
         width = 1., height = 1.,
         just = "right", 
         gp = gpar(), clip = "on",
         xscale=ppp$xrange_delta,yscale=yrange,
         angle = 0,
         layout = NULL,
         layout.pos.row = NULL, layout.pos.col = NULL,
         name = NULL,
	)

   #@@@@@@@@@@@@@@@@
   pushViewport(vp2)
   #@@@@@@@@@@@@@@@@
   grid.rect()
   grid.xaxis(gp=gpar(fontsize=font_f*9))
   grid.yaxis(at=log(yticks*100.),label=yticks,gp=gpar(fontsize=font_f*9))
   grid.text('Pressure [hPa]',x=-.18,y=.5,default.units = "npc",gp = gpar(fontsize=font_f*10),rot=90)

   if( ppp$thickness == 0){
      titre='Delta ln(pi) (Momentum)'
      yy_sea=.5*(lnpi[1  ,2:nk]+lnpi[1  ,1:(nk-1)])
      yy_top=.5*(lnpi[nih,2:nk]+lnpi[nih,1:(nk-1)])
      str1=' Delta below the hill'
      str2=' Delta over top of the hill'
   } else {
   titre='Layer Thickness (Momentum)'
   yy_sea=lnpi[1,1:nk]
   yy_top=lnpi[nih,1:nk]
   str1=' Layer Thickness below the hill [m]'
   str2=' Layer Thickness over top of the hill [m]'
   }

   grid.text(titre,x=.5,y=1.01,default.units = "npc",gp = gpar(fontsize=font_f*14),just=c("centre","bottom"))

   #@@@@@@@@@@@@@@@@
   pushViewport(vp2.1)
   #@@@@@@@@@@@@@@@@	

   grid.lines(x=delta_sea,y=yy_sea,gp=gpar(col='black'),default.units="native")
   grid.lines(x=delta_top,y=yy_top,gp=gpar(col='red'),default.units="native")

   #@@@@@@@@@@@
   upViewport()
   #@@@@@@@@@@@	

   yyy=-.05
   grid.lines(x=c(0.,.15),y=c(yyy,yyy),default.units="npc")
   if(ppp$p_top_hill > ppp$p_below_hill){
      strt=str2
      str2=str1
      str1=strt
   }

   grid.text(str1,x=.15,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12),just=c("left"))
   yyy=yyy-.03
   grid.lines(x=c(0.,.15),y=c(yyy,yyy),gp=gpar(col='red'),default.units="npc")
   grid.text(str2,x=.15,y=yyy,default.units = "npc",gp = gpar(fontsize=font_f*12,col='red'),just=c("left"))
      
}
