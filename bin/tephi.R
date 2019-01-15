# Acquire SCM support utilities
source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','utils.R',sep='/'))
const<-utils.read.constants()

# Main plotting function for tephigrams
tephi <- function(
      pres,
      presm,
      temp,
      spread,
      wind_module=-1,
      wind_dir=-1,
      zoom='undef',
      titre='',
      titre2='',
      titre3='',
      barbs_thick=1.5,
      barbs_position_at_1000_hPa=40,
      barbs_length=0.05,
      col='black',
      plot.type='l',
      add=FALSE
) {

#==============================================================================
#
# Thepigramme en R, pour iMeteo.ca
#
# Autor : Andre Plante et Claude Girard juin 2006
#         Andre Plante fevrier 2014 ajout de vents
#
#==============================================================================
# Load libraries
library(grid)

#------------------------------------------------------------------------------
# Prepare the "device"

# Pour echel du graphe

if ( zoom == 'undef' ){
   tmin=-90
   tmax=16
}
if ( zoom == 'chaud' ){
   tmin=-10
   tmax=15
}

#------------------------------------------------------------------------------
p0theta=1000.
cw=4218.
ew0=6.1058
c2=-(const$CPV-cw)/const$RGASV
c1=const$CHLC/const$RGASV+c2*const$TCDK
c0=log(ew0)+c1/const$TCDK+c2*log(const$TCDK)

# Pour tracer isotermes et theta
trace_tmin=-70
trace_tmax=50

trace_pmax=1050
trace_pmin=100

#==============================================================================
# Praparation de l'espace de tracage
# Pour changer la taille jouer avec width, height, et/ou xscale, yscale

xscale=log(c(230,400))
yscale=c(tmax,tmin)
angle=45

vp <- viewport(xscale=xscale,yscale=yscale,width=1.,height=1.,angle=angle)
#@@@@@@@@@@@@@@@
pushViewport(vp)
#@@@@@@@@@@@@@@@
#grid.rect()
#grid.xaxis()
#grid.yaxis()

# Only plot background if not adding to existing plot
if (!add){

    #==============================================================================
    # Tracer les pseudo-adiabatiques saturees [Iribarne, p143, formule (85)]

    ts_liste=(0:22)*4.-40.
    nb=length(ts_liste)
    nmax=100

    courbes <- array(0., dim=c(nmax,2,nb,2))
    count   <- array(0 , dim=c(nb,2))
        
    for (i in 1:length(ts_liste)) {
        ts=ts_liste[i]+const$TCDK
        ews=exp(c0-c1/ts-c2*log(ts))
        rws=const$EPS1*ews/(p0theta-ews)
        lvs=const$CHLC+(const$CPV-cw)*(ts-const$TCDK)
        Const=const$CPD*log(ts)-const$RGASD*log(p0theta-ews)+rws*lvs/ts
        for (dir in c(1,-1)){
            step=dir*20
            if (step > 0) {
                pfin=trace_pmax
                
            } else {
                pfin=trace_pmin
            }
            t=ts
            srw=0.
            t1=t-const$TCDK
            theta1=t1+const$TCDK
            stop=0      
            pm=1000
            
            for (p in seq(1000+step,pfin,step)) {
                if (stop == 0) {
                    tm=t
                    ewm=exp(c0-c1/tm-c2*log(tm))
                    rwm=const$EPS1*ewm/(p-ewm)
                    for ( j in 1:2) {
                        ew=exp(c0-c1/t-c2*log(t))
                        rw=const$EPS1*ew/(p-ew)
                        lv=const$CHLC+(const$CPV-cw)*(t-const$TCDK)
                        dew=ew*(c1/t^2-c2/t)
                        drw=const$EPS1*dew*p/(p-ew)^2
                        dlv=const$CPV-cw
                        f_t=const$CPD*log(t)+cw*(srw+rwm*log(t/tm))-const$RGASD*log(p-ew)+rw*lv/t-Const
                        fp_t=(const$CPD+cw*rwm)/t+const$RGASD/(p-ew)*dew+rw*lv/t*(drw/rw+dlv/lv-1/t)
                        t=t-f_t/fp_t
                    }
                    srw=srw+rwm*log(t/tm)
                    t2=t-const$TCDK
                    theta2=(t2+const$TCDK)*(p0theta/p)^const$CAPPA
                    if (t2 < -50) {
                        aa=(theta1-theta2)/(t1-t2)
                        bb=theta1-aa*t1
                        t2=-50
                        theta2=aa*t2+bb
                        stop=1
                    }
                    if (dir == -1){
                            k=1
                            courbes[count[i,k]  ,1,i,k]=log(theta1)
                            courbes[count[i,k]  ,2,i,k]=t1
                            courbes[count[i,k]+1,1,i,k]=log(theta2)
                            courbes[count[i,k]+1,2,i,k]=t2
                            count[i,k]=count[i,k]+1
                            if (count[i,k]+1 > nmax){
                                "augmenter nmax"
                                stop
                            }
                        }
                    if (dir == 1) {
                        k=1
                        courbes[count[i,k]  ,1,i,k]=log(theta1)
                        courbes[count[i,k]  ,2,i,k]=t1
                        courbes[count[i,k]+1,1,i,k]=log(theta2)
                        courbes[count[i,k]+1,2,i,k]=t2
                        count[i,k]=count[i,k]+1
                        if (count[i,k]+1 > nmax){
                            "augmenter nmax"
                            stop
                        }
                    }
                    t1=t2
                    theta1=theta2
                    pm=p
                }
            }	
        }
        count[i,1]=count[i,1]+1
        count[i,2]=count[i,2]+1
    
    }
    
    k=1
    for (i in seq(1,nb-1,2)) {
        grid.lines(x=courbes[1:(count[i  ,k]-1),1,i  ,k],y=courbes[1:(count[i  ,k]-1),2,i  ,k],gp=gpar(col='orange'),default.units ="native")
        grid.lines(x=courbes[1:(count[i+1,k]-1),1,i+1,k],y=courbes[1:(count[i+1,k]-1),2,i+1,k],gp=gpar(col='orange'),default.units ="native")
    }

    #==============================================================================
    # Tracer les rs eau avec formule empirique pour ew Iribarne pp 68 eq. (53)

    if ( 1 == 1 ){

        rs_liste=c(.02,.03,.05,.15,.2,.3,.4,.6,.8,1.,1.5,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.,25.,30.,35.,40.,50.)

        n=length(rs_liste)
        ei0=ew0
        c1i=(const$CHLC+const$CHLF)/const$RGASV
        c0i=log(ei0)+c1i/const$TCDK

        for (i in 1:length(rs_liste)){

            rs=rs_liste[i]*1.e-3

            p=trace_pmax
            const.lnp=log(p/(1.+const$EPS1/rs))
            
            t=const$TCDK
            for ( j in 1:10){
                f_t=const.lnp-c0+c1/t+c2*log(t)
                fp_t=-c1/t^2+c2/t
                t=t-f_t/fp_t
            }

            tm=t-const$TCDK
            lnthetam=log(t)+const$CAPPA*log(p0theta/p)
            
            for (tp in seq(50,-40,-5)){
                
                if ( tp < tm ){
                    
                    t=tp+const$TCDK
                    ew=exp(c0-c1/t-c2*log(t))
                    p=ew*(1.+const$EPS1/rs)
                    lnthetap=log(t)+const$CAPPA*log(p0theta/p)
                    grid.lines(x=c(lnthetam,lnthetap),y=c(tm,tp),gp=gpar(col='orange',lwd=1),default.units ="native")
                    
                    tm=tp
                    lnthetam=lnthetap
                    
                }
                
            }
            
            p=1080.
            const.lnp=log(p/(1.+const$EPS1/rs))
            
            t=const$TCDK
            for ( j in 1:10){
                f_t=const.lnp-c0+c1/t+c2*log(t)
                fp_t=-c1/t^2+c2/t
                t=t-f_t/fp_t
            }
            
            tm=t-const$TCDK
            lnthetam=log(t)+const$CAPPA*log(p0theta/p)
            grid.text(paste(rs*1000.),x=lnthetam,y=tm,gp=gpar(col='orange'),default.units ="native",just = "center",rot = -45)
            
        }
    }
    
    #==============================================================================
    # Tracer les theta
    for (vtheta in seq(200,400,5)) {
        lntheta=log(vtheta)
        tt=vtheta*(trace_pmax/p0theta)^const$CAPPA
        grid.lines(x=c(lntheta,lntheta),y=c(trace_tmin,tt-const$TCDK),gp=gpar(col='orange'),default.units ="native")
        
        # Metre les valeurs de theta le long de l'isotherme tt
        tt=-25
        if(p0theta*((tt+const$TCDK)/vtheta)^(1/const$CAPPA) <= trace_pmax){
            grid.text(paste(vtheta),x=lntheta,y=tt,gp=gpar(col='orange'),default.units ="native",rot = -90,just = "bottom")
        }
    }

    #==============================================================================
    # Tracer les isothermes
    for (ii in seq(trace_tmin,trace_tmax,1)){
        linethick=.3
        label=""
        if ( ii %% 5 == 0 ) {
            linethick=1
            label=paste(ii)
        }
        if ( ii == 0 ) { linethick=4 }
        lnthetam=log(ii+const$TCDK)+const$CAPPA*log(p0theta/trace_pmax)
        grid.lines(x=c(lnthetam,log(500)),y=c(ii,ii),gp=gpar(col='green',lwd=linethick),default.units ="native")
        
        # Tracer les valeurs de T le long de theta=vtheta
        vtheta=305
        if(p0theta*((ii+const$TCDK)/vtheta)^(1/const$CAPPA) <= trace_pmax){
            grid.text(label,x=log(vtheta),y=ii,gp=gpar(col='green'),default.units ="native",just = "bottom")
        }
   
        # Tracer les valeurs de T le long de p trace_pmax
        lntheta=log(ii+const$TCDK)+const$CAPPA*log(p0theta/trace_pmax)
        grid.text(paste(label,"  ",sep=""),x=lntheta,y=ii,gp=gpar(col='green'),default.units ="native",just="right")
   
    }

    #==============================================================================
    # Tracer p	
    # Equation 1 pp 100 Iribarne

    for (p in seq(trace_pmax,trace_pmin,-50.)){
        tm=-70.
        lnthetam=log(tm+const$TCDK)+const$CAPPA*log(p0theta/p)
        for (tp in seq(-65.,50.,5.)){
            lnthetap=log(tp+const$TCDK)+const$CAPPA*log(p0theta/p)
            grid.lines(x=c(lnthetam,lnthetap),y=c(tm,tp),gp=gpar(col='green',lwd=linethick),default.units ="native")
            lnthetam=lnthetap
            tm=tp
        }

        tt=-15
        lntheta=log(tt+const$TCDK)+const$CAPPA*log(p0theta/p)
        grid.text(paste(p),x=lntheta,y=tt,gp=gpar(col='green'),default.units ="native",rot=-45,just = "bottom")
    }
} # end of background plotting
    
#==============================================================================
# Tracer les profiles

nb=min(length(temp),length(spread))

my_lntheta=log(temp+const$TCDK)+const$CAPPA*log(p0theta/pres)
grid.lines(x=my_lntheta,y=temp,gp=gpar(col=col,lwd=3),default.units ="native")
if (plot.type == 'b'){
    grid.points(x=my_lntheta,y=temp,gp=gpar(col=col,lwd=3,pch=1),default.units ="native")
}

my_lntheta=log(temp[1:nb]-spread[1:nb]+const$TCDK)+const$CAPPA*log(p0theta/pres[1:nb])
grid.lines(x=my_lntheta,y=temp[1:nb]-spread[1:nb],gp=gpar(col=col,lwd=3,lty=2),default.units ="native")
if (plot.type == 'b'){
    grid.points(x=my_lntheta,y=temp[1:nb]-spread[1:nb],gp=gpar(col=col,lwd=3,pch=1),default.units ="native")
}

#==============================================================================
# Tracer les vents

if ( wind_dir[1] != -1 ){

   #
   # 
   # 
   # 
   #(0,wy0) \   \
   #         \   \
   #          \wxd\
   #           -------------
   #       (wx0,0)        (1,0)
   ###########################

   # Define a 45 degree slop on graph (delat ln(tetha))/delta(T)
   # to plot the wind barbs
   slop=(-70-0)/(log(350/270))

   lntho_1000=log(barbs_position_at_1000_hPa+const$TCDK)+const$CAPPA*log(p0theta/1000)

   ordonee=barbs_position_at_1000_hPa-slop*lntho_1000

   my_t=slop*log(400)+ordonee
   #grid.lines(x=c(lntho_1000,log(400)),y=c(barbs_position_at_1000_hPa,my_t),gp=gpar(col='blue',lwd=3),default.units ="native")

   theta=320
   for (k in seq(1,length(wind_dir))) {
      exner=(presm[k]/p0theta)^const$CAPPA
      # Methode de Raphson-Newton, converge en 3 iterations au centieme
      for (i in seq(1,100)) {
         ff=slop*log(theta)-exner*theta+ordonee+const$TCDK
         ffp=slop/theta -exner
         thetaP=theta-ff/ffp
	 dif=abs(thetaP-theta)
	 theta=thetaP
	 #print(theta)
	 if(dif < .01){break}
      }
      tt=exner*theta-const$TCDK
      #print(paste(pres[k],tt,theta))
      #grid.points(x=log(theta),y=tt,gp=gpar(col='red',lwd=3),default.units ="native")

      px=convertX(unit(log(theta),"native"),"npc",valueOnly=TRUE)
      py=convertY(unit(tt,"native"),"npc",valueOnly=TRUE)
      #print(paste(px,py))
      #grid.points(x=px,y=py,gp=gpar(col='red',lwd=2),default.units ="npc")
      #dx=sin((wind_dir[k]+angle)*pi/180.)*2*barbs_length/aspect
      #dy=cos((wind_dir[k]+angle)*pi/180.)*2*barbs_length
      #grid.lines(x=c(px,px+dx),y=c(py,py+dy),gp=gpar(col='red',lwd=2),default.units ="npc")
      #grid.text(wind_module[k],x=px+barbs_length,y=py-barbs_length,default.units ="npc",rot=-angle)

      vpw <- viewport(just=c('right','bottom'),x=unit(px,"npc"),y=unit(py,"npc"),xscale=c(0,1),yscale=c(0,1),width=barbs_length/aspect,height=barbs_length,angle=-wind_dir[k]+270-angle)
      #@@@@@@@@@@@@@@@
      pushViewport(vpw)
      #@@@@@@@@@@@@@@@
      #grid.polygon(x=c(0,0,1,1),y=c(0,1,1,0),gp=gpar(fill=1:5))
      wx0=0.15
      wy0=0.25
      wxd=.11
      xx=wx0
      total=0
      # draw 50 knots flags

      #print(floor(wind_module[k]/50))
      nb=floor(wind_module[k]/50)
      if(nb>0){
         for ( i in seq(1,nb)){
            grid.polygon(x=c(xx,xx-wx0/3,xx+wxd*.85),y=c(0,wy0,0),gp=gpar(fill=col,col=col))
            xx=xx+wxd
            total=total+50
	    if(i==nb){xx=xx+wxd}
         }
      }
      # draw 10 knots barbs                  
      nb=floor((wind_module[k]-total)/10)
      if(nb>0){
         for ( i in seq(1,nb)){
            grid.lines(x=c(xx,xx-wx0),y=c(0,wy0),gp=gpar(col=col,lwd=barbs_thick),default.units ="npc")  
            xx=xx+wxd
            total=total+10
         }
      }
      if(wind_module[k]-total>=2.5){
         if(xx==wx0){xx=xx+wxd}
	 grid.lines(x=c(xx,xx-wx0/2),y=c(0,wy0/2),gp=gpar(col=col,lwd=barbs_thick),default.units ="npc")
      }
      if(wind_module[k]<2.5){
         grid.points(x=1,y=0,gp=gpar(col=col,lwd=barbs_thick),default.units ="npc")
      } else {
         # Corp de la barbule
         grid.lines(x=c(wx0,1),y=c(0,0),gp=gpar(col=col,lwd=barbs_thick),default.units ="npc")      
      }
      #@@@@@@@@@@@
      upViewport()
      #@@@@@@@@@@@

   }

}

#==============================================================================
#@@@@@@@@@@@
upViewport()
#@@@@@@@@@@@

grid.text(titre           ,x=.02,y=.98,gp=gpar(col='black',fontsize=18),default.units ="npc",just="left")
grid.text(titre2          ,x=.02,y=.94,gp=gpar(col='black',fontsize=16),default.units ="npc",just="left")
grid.text(titre3          ,x=.02,y=.90,gp=gpar(col='black',fontsize=16),default.units ="npc",just="left")

#==============================================================================
}


########## Main Program ###########

# Read and parse command line arguments
args<-commandArgs()
files<-strsplit(args[5],',')[[1]]
fcst.list<-args[6]
type<-args[7]
col.list.user<-strsplit(args[8],',')[[1]]
pch.user<-as.numeric(args[9])
paths<-args[10:length(args)]

# Common setup
fcst.vec<-as.numeric(unlist(strsplit(fcst.list,',')))
fld.cnt<-length(paths)*length(fcst.vec)
ext=strsplit(files[1],'_')[[1]][length(strsplit(files[1],'_')[[1]])]
pfile.t<-paste('PREST_',ext,sep='')
pfile.m<-paste('PRESM_',ext,sep='')
date.init<-strptime(strsplit(ext,'.txt',fixed=TRUE)[[1]][1],format="%Y%m%d.%H%M%S")
date.valid<-date.init+fcst.vec[1]
oname<-paste('tephi_',sub(' ','T',format(date.valid,format='%Y%m%d.%H%M%S')),'.',type,sep='')
write(oname,file='.tephi_outname')
if (col.list.user[1] == 'R_colours'){
  col.list<-seq(1,fld.cnt)
} else {
  col.list<-rep(col.list.user,fld.cnt)
}
plot.type<-'l'
if (pch.user > 0){plot.type<-'b'}
f<-list("tt"=1,"hu"=2,"uu"=3,"vv"=4)

# Prepare the plot device
width=700
height=500
aspect=width/height
if (type == 'jpeg'){
    jpeg(oname,quality=90,width=width,height=height,units="px",pointsize=10,bg="white")
} else if (type == 'ps'){
    postscript(oname,width=width,height=height,units="px",pointsize=10,bg="white")
} else {
    png(filename=oname,width=width,height=height,units="px",pointsize=10,bg="white")
}

# Initialize data elements
exp<-c()

# Loop over requested times
col.index<-0
add.to.plot<-FALSE
for (fcst.sec in fcst.vec){

    # Retrieve data and times
    date.valid<-date.init+fcst.sec

    # Loop over inputs to plot
    for (i in 1:length(paths)){
        prog<-difftime(date.valid,date.init,units="hours")
        exp.this<-capwords(basename(paths[i])[1])
        col.index<-col.index+1
        
        # Experiment and data setup
        if (length(fcst.vec) > 1){exp.this<-paste(exp.this,' (',prog,'h)',sep='')}
        exp<-c(exp,exp.this)
        pfld.t<-read.table(paste(paths[i],'coord',pfile.t,sep='/'),header=TRUE)
        pfld.m<-read.table(paste(paths[i],'coord',pfile.m,sep='/'),header=TRUE)
        ttfld<-read.table(paste(paths[i],'series',files[f$tt],sep='/'),header=TRUE)
        hufld<-read.table(paste(paths[i],'series',files[f$hu],sep='/'),header=TRUE)        
        uufld<-read.table(paste(paths[i],'series',files[f$uu],sep='/'),header=TRUE)
        vvfld<-read.table(paste(paths[i],'series',files[f$vv],sep='/'),header=TRUE)
        pfld.date<-as.POSIXct(strptime(pfld.t[,1],format="%Y%m%dT%H%M%S"))
        pfld.t.values<-as.numeric(pfld.t[pfld.date==date.valid,2:dim(pfld.t)[2]])/100.
        pfld.m.values<-as.numeric(pfld.m[pfld.date==date.valid,2:dim(pfld.m)[2]])/100.
        fld.date<-as.POSIXct(strptime(ttfld[,1],format="%Y%m%dT%H%M%S"))
        ttfld.values<-as.numeric(ttfld[fld.date==date.valid,2:ncol(pfld.t)])
        hufld.values<-as.numeric(hufld[fld.date==date.valid,2:ncol(pfld.t)])
        uufld.values<-as.numeric(uufld[fld.date==date.valid,2:ncol(pfld.m)])
        vvfld.values<-as.numeric(vvfld[fld.date==date.valid,2:ncol(pfld.m)])

        # Compute derived fields
        wind.spd<-sqrt(uufld.values^2+vvfld.values^2)
        wind.dir<-atan(uufld.values/abs(vvfld.values))*180/pi + 180 + 180*(vvfld.values<0)
        tdk<-5.42e3/(log(2.53e9*0.622/(sapply(hufld.values,function(x) max(x,1e-6))*pfld.t.values)))
        dwpt.depr<-ttfld.values-tdk
        dwpt.depr[dwpt.depr<0]<-0
        ttfld.values.c<-ttfld.values-const$TCDK      

        # Call tephigram plotting
        status<-tephi(pfld.t.values,pfld.m.values,ttfld.values.c,dwpt.depr,wind.spd,wind.dir,
                      col=col.list[col.index],
                      plot.type=plot.type,
                      add=add.to.plot)
        add.to.plot<-TRUE

    }
}

# Plot legend
for (i in seq(1,fld.cnt)){
    grid.text(exp[i],x=0.02,y=0.95-0.035*(i-1),gp=gpar(col=col.list[i],cex=1.6),hjust=0)
}

# Finalize plotting
dev.off()
