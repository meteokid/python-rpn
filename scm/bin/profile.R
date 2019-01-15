# Read and parse command line arguments
args<-commandArgs()
files<-strsplit(args[5],',')[[1]]
fcst.list<-args[6]
levtype<-args[7]
coord<-args[8]
toplevel<-as.numeric(args[9])
botlevel<-as.numeric(args[10])
xlow<-as.numeric(args[11])
xhigh<-as.numeric(args[12])
legend.x<-args[13]
type<-args[14]
zero<-as.logical(args[15])
nodiag<-as.logical(args[16])
col.list.user<-strsplit(args[17],',')[[1]]
pch.user<-as.numeric(args[18])
paths<-args[19:length(args)]

# Basic configuration
lwd<-3
pch.diag<-19

# Acquire SCM support utilities
source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','utils.R',sep='/'))

# Common setup
fcst.vec<-as.numeric(unlist(strsplit(fcst.list,',')))
fld.name.default<-fldname(files[1])
fld.cnt<-length(paths)*length(files)*length(fcst.vec)
ext=strsplit(files[1],'_')[[1]][length(strsplit(files[1],'_')[[1]])]
pfile<-paste(coord,levtype,'_',ext,sep='')
dfile<-paste('DIAG',levtype,'_',ext,sep='')
date.init<-strptime(strsplit(ext,'.txt',fixed=TRUE)[[1]][1],format="%Y%m%d.%H%M%S")
date.valid<-date.init+fcst.vec[1]
oname<-paste(fld.name.default,'_',sub(' ','T',format(date.valid,format='%Y%m%d.%H%M%S')),'.',type,sep='')
write(oname,file='.prof_outname')
zero.plot<-NA
if (zero){zero.plot<-0}
if (col.list.user[1] == 'R_colours'){
  col.list<-seq(1,fld.cnt)
} else {
  col.list<-rep(col.list.user,fld.cnt)
}
plot.type<-'l'
if (pch.user > 0){plot.type<-'b'}
log.y<-''
coord.mult<-1.
coord.unit<-''
if (coord=='PRES'){
    log.y<-'y'
    coord.mult<-0.01
    coord.unit<-'(hPa)'
} else if (coord=='HGHT'){
    tmp<-toplevel
    toplevel<-botlevel
    botlevel<-tmp
    coord.unit<-'(m)'
}
coord.name<-paste(coord,'Level',coord.unit)

# Initialize data elements
exp<-c()
fld.x<-list()
fld.y<-list()
diag.x<-list()
diag.y<-list()
fld.xmax<--Inf
fld.xmin<-Inf
fld.ymax<--Inf
fld.ymin<-Inf

# Loop over requested times
for (fcst.sec in fcst.vec){

  # Retrieve data and times
  date.valid<-date.init+fcst.sec
  
  # Loop over inputs to plot
  for (i in 1:length(paths)){
    prog<-difftime(date.valid,date.init,units="hours")
    exp.this<-capwords(basename(paths[i])[1])

    # Loop over fields to plot
    for (j in 1:length(files)){
      fld.name.this<-fldname(files[j])

      # Experiment and data setup
      expfld.this<-paste(exp.this,fld.name.this)
      if (length(fcst.vec) > 1){expfld.this<-paste(expfld.this,' (',prog,'h)',sep='')}
      exp<-c(exp,expfld.this)
      fname<-paste(paths[i],'series',files[j],sep='/')
      fld<-read.table(fname,header=TRUE)
      pfld<-read.table(paste(paths[i],'coord',pfile,sep='/'),header=TRUE)
      fld.date<-as.POSIXct(strptime(fld[,1],format="%Y%m%dT%H%M%S"))
      pfld.date<-as.POSIXct(strptime(pfld[,1],format="%Y%m%dT%H%M%S"))
      fld.values<-as.numeric(fld[fld.date==date.valid,2:dim(fld)[2]])
      if (coord == 'HYB'){
          pfld.values<-as.numeric(pfld[,2:dim(pfld)[2]])*coord.mult
      } else {
          pfld.values<-as.numeric(pfld[pfld.date==date.valid,2:dim(pfld)[2]])*coord.mult
      }

      # Retrieve diagnostic values and reshape fields if necessary
      diag.x[[expfld.this]]<-NA
      diag.y[[expfld.this]]<-NA
      if (coord == 'PRES'){
          if (length(fld.values) < length(pfld.values)){
              stop(paste('there are an insufficient number of levels in',fname))
          } else if (length(fld.values) == (length(pfld.values)+1)) {
              if (! nodiag){
                  dfld<-read.table(paste(paths[i],'coord',dfile,sep='/'),header=TRUE)
                  dfld.date<-as.POSIXct(strptime(dfld[,1],format="%Y%m%dT%H%M%S"))
                  diag.values<-as.numeric(dfld[dfld.date==date.valid,2])
                  if (diag.values>=toplevel & diag.values<=botlevel){
                      diag.x[[expfld.this]]<-fld.values[length(fld.values)]
                      diag.y[[expfld.this]]<-diag.values
                  }
              }
          }
      } else {
          diag.x[[expfld.this]]<-fld.values[length(fld.values)]
          diag.y[[expfld.this]]<-pfld.values[length(fld.values)]
          fld.values<-fld.values[1:(length(fld.values)-1)]
          pfld.values<-pfld.values[1:(length(pfld.values)-1)]
      }
      if (coord == 'HGHT'){
          fld.values<-rev(fld.values)
          pfld.values<-rev(pfld.values)
      }

      # Accumulate field x/y values for plots
      pmask<-pfld.values>=toplevel & pfld.values<=botlevel    
      pmask.ext<-pmask | c(pmask[1],pmask[1:(length(pmask)-1)]) | c(pmask[2:length(pmask)],pmask[length(pmask)])
      pmask.ext[(length(pmask.ext)+1)]<-FALSE
      fld.x[[expfld.this]]<-fld.values[pmask.ext]
      fld.y[[expfld.this]]<-pfld.values[pmask.ext]
      fld.xmax<-max(fld.xmax,max(c(fld.x[[expfld.this]],diag.x[[expfld.this]]),na.rm=TRUE))
      fld.xmin<-min(fld.xmin,min(c(fld.x[[expfld.this]],diag.x[[expfld.this]]),na.rm=TRUE))
      fld.ymax<-max(fld.ymax,max(c(fld.y[[expfld.this]],diag.y[[expfld.this]]),na.rm=TRUE))
      fld.ymin<-max(fld.ymin,min(c(fld.y[[expfld.this]],diag.y[[expfld.this]]),na.rm=TRUE))
    }
  }
}

# Override plot ranges on user request
if (xlow > -Inf){fld.xmin<-xlow}
if (xhigh < Inf){fld.xmax<-xhigh}
yrange<-c(min(fld.ymax,botlevel),toplevel)
if (coord=='HGHT'){yrange<-rev(yrange)}

# Select plot device type
if (type == 'png'){
  png(oname)
} else if (type == 'jpeg'){
  jpeg(oname,quality=90)
} else if (type == 'ps'){
  postscript(oname)
} else {
  png(oname)
}

# Generate plot
plot(x=fld.x[[exp[1]]],y=fld.y[[exp[1]]],
     type=plot.type,
     xlim=c(fld.xmin,fld.xmax),
     ylim=yrange,
     log=log.y,
     xlab=fld.name.default,
     ylab=coord.name,
     main=paste(fld.name.default,' at ',format(date.valid,format='%H%M UTC %d %b %Y ('),
       as.numeric(difftime(date.valid,date.init,units="hours")),'h)',sep=' '),
     panel.first=abline(v=zero.plot),
     cex.main=1.2,
     cex.lab=1.,
     cex.axis=1.,
     lwd=lwd,
     pch=pch.user,
     col=col.list[1])
points(x=diag.x[[exp[1]]],y=diag.y[[exp[1]]],
       pch=pch.diag,
       col=col.list[1])
for (i in 2:length(exp)){  
  lines(x=fld.x[[exp[i]]],y=fld.y[[exp[i]]],
        type=plot.type,
        pch=pch.user,
        lwd=lwd,
        col=col.list[i])
  points(x=diag.x[[exp[i]]],y=diag.y[[exp[i]]],
       pch=pch.diag,
       col=col.list[i])
}

# Add legend to plot
legend(x=legend.x,legend=exp,
       inset=0.01,
       col=col.list,
       lwd=rep(lwd,length(exp)))
dev.off()
