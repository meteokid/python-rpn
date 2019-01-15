make_levels <- function(pres_first_level=-1,
                           height_above_ground=-1,
                           pin=c(-1),
                           amp=c(-1)
                        ) {

   #===========================================================================
   #
   # Autors: Natacha Bernier and Andre Plante Oct 2010, version 1.0.0
   #
   # Language : R
   #
   #===========================================================================

   version="1.0.1"
   print(paste('make_levels, version',version))

   postscript("make_levels.ps", bg='white', horizontal=FALSE, width=7.5, height=10., pointsize=10, paper = "letter")

   #==============================================================================
   # Check input
   if ( pres_first_level ==  -1 ){
      print('keyword pres_first_level=(press in hPa) is mandatory')
      q()
   }
   if ( height_above_ground ==  -1 ){
      print('keyword height_above_ground=(height in m) is mandatory')
      q()
   }
   if ( pin[1] ==  -1 ){
      print('keyword pin=(array of pressure in hPa) is mandatory')
      q()
   }
   pin=log(pin*100.)
   if ( amp[1] ==  -1 ){
      print('keyword apm=(array of amplification factors, same size as pin) is mandatory')
      q()
   }

   #==============================================================================
   # Load libraries
   library(grid)

   #==============================================================================
   # Domain definition (en Pascals)
   #------------------
   pres_first_level=pres_first_level*100.
   pref=1000*100.
   zetas=log(pref)
   zetat=log(pres_first_level)

   #==============================================================================
   # Define piecewise linear amplification factors.
   ff=segments_de_droite_diffusees(pin,amp,n=50)
   nfit=length(ff$x)   

   #================================
   # Level production from bottom up
   #--------------------------------
   nmax=1000

   rgasd=0.2870500000000E+03
   tcdk=0.2731500000000E+03
   grav=0.9806160000000E+01
   zeta=seq(1,nmax)

   # Place first two levels
   zeta[1]=zetas-grav*height_above_ground/(rgasd*tcdk)
   zeta[2]=zeta[1]-2*grav*(height_above_ground/(rgasd*tcdk))

   k=3
   iz=nfit
   while ( k <= nmax & zeta[k-1] > pin[length(pin)] ) {
      delta=zeta[k-2]-zeta[k-1]
      for (i in seq(iz,1,-1)){
         if ( ff$x[i] <= zeta[k-1] )
         break
      }
      iz=i-1
      zeta[k]=zeta[k-1]-delta*ff$y[i]
      if ( zeta[k] <= zetat ) {
         print('Top reached')
         break
      }
      k=k+1
   }
   nk=k-1

   # Calculate level height in m
   #heights=-rgasd*tcdk*(zeta[1:nk]-zetas)/grav
   #print(heights)

   print(paste('Number of levels=',nk))

   # Put in GEM order
   tempo=rev(zeta[1:nk])
   zeta=tempo
 
   # Write level related parameters of gem_cfgs namelist in file gem_settings.nml_L??
   digits=5
   file_out=paste('gem_settings.nml_L',nk,sep='')
   write('&gem_cfgs',file_out, append = FALSE)
   write(paste('Cstv_ptop_8=',format(exp(zeta[1]-.5*(zeta[2]-zeta[1])),digits=digits)),file_out, append = TRUE)
   write(paste('Grd_rcoef=',1.0,',',1.0),file_out, append = TRUE)
   write('hyb=',file_out, append = TRUE)
   for ( k in 1:nk ){
      write(paste(format(exp(zeta[k]-zetas),digits=digits),','),file_out, append = TRUE)
   }	
   write('/',file_out, append = TRUE)

   #===========================================================================
   # Plot amplification factor
   #--------------------------

   yrange=c(pin[1],pin[length(pin)])

   # Define y ticks
   decade=c(100,70,50,35,25,15)
   tempo=c(1000,950,900,850,800,700,600,500,420,350,300,250,200,160,130,decade,decade*.1,decade*.01,decade*.001,decade*.0001)
   for (i in 1:length(tempo)){
      if(tempo[i]*100.<exp(pin[length(pin)])){break}
   }
   imin=i-1
   for (i in seq(imin,1,-1)){
      if(tempo[i]*100.>exp(pin[1])){break}
   }   
   yticks=tempo[i:imin]   

   grid.text('Amplification factor',x=.5,y=.98,default.units = "npc",gp = gpar(fontsize=18),just=c("centre","top"))

   xrange=c(min(ff$y,1),max(ff$y))

   space_centre=.1
   vp1 <- viewport(x =.5, y = 0.48,
      width = .8, height = .75,
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
   grid.rect()
   grid.text('Pressure [hPa]',x=-.18,y=.5,default.units = "npc",gp = gpar(fontsize=10),rot=90)
   grid.xaxis(gp=gpar(fontsize=9))
   grid.yaxis(at=log(yticks*100.),label=yticks,gp=gpar(fontsize=9))
   grid.lines(x=ff$y,y=ff$x,gp=gpar(col='red'),default.units="native")
   grid.points(x=amp,y=pin,gp=gpar(col='red'),default.units="native")

}
#===========================================================================
#===========================================================================
#===========================================================================
segments_de_droite_diffusees <- function(x0,y0,n=100) {

   npts=length(x0)

   # Check x0 consistancy
   for (k in seq(2,npts,1)) {
      if(x0[k] >= x0[k-1]){
         print('Point must be in decreasing order:')
	 x0
         q()
      }
   }

   xx=seq(1,n*(npts-1)+1,1)
   yy=xx
   for (k in seq(1,(npts-1),1)) {
      xx[((k-1)*n+1):(k*n+1)]=seq(x0[k],x0[k+1],(x0[k+1]-x0[k])/n)
      yy[((k-1)*n+1):(k*n+1)]=seq(y0[k],y0[k+1],(y0[k+1]-y0[k])/n)
   }

   return(list(x=rev(xx),y=rev(yy)))

}
