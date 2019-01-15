# Read and parse command line arguments
args<-commandArgs(trailingOnly=TRUE)
files<-strsplit(args[1],',')[[1]]
lev.list<-as.integer(strsplit(args[2],',')[[1]])
legend.x<-args[3]
type<-args[4]
col.list.user<-strsplit(args[5],',')[[1]]
pch.user<-as.numeric(args[6])
xlow<-as.numeric(args[7])
xhigh<-as.numeric(args[8])
ylow<-as.numeric(args[9])
yhigh<-as.numeric(args[10])
paths<-args[11:length(args)]

# Basic configuration
lwd<-3

# Acquire SCM support utilities
source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','utils.R',sep='/'))

# Common setup
fld.name.default<-fldname(files[1])
fld.cnt<-length(paths)*length(files)
ext=strsplit(files[1],'_')[[1]][length(strsplit(files[1],'_')[[1]])]
date.init<-strptime(strsplit(ext,'.txt',fixed=TRUE)[[1]][1],format="%Y%m%d.%H%M%S")
oname<-paste(fld.name.default,'_',lev.list[1],'.',type,sep='')
write(oname,file='.series_outname')
if (col.list.user[1] == 'R_colours'){
  col.list<-seq(1,fld.cnt)
} else {
  col.list<-rep(col.list.user,length=fld.cnt)
}
plot.type<-'l'
if (pch.user > 0){plot.type<-'b'}

# Loop over inputs to plot
exp<-c()
fld.x<-list()
fld.y<-list()
fld.xmax<--Inf
fld.xmin<-Inf
fld.ymax<--Inf
fld.ymin<-Inf
for (i in 1:length(paths)){
    exp.this<-capwords(basename(paths[i])[1])
    for (j in 1:length(files)){
        fld.name.this<-fldname(files[j])
        
        # Experiment and data setup
        expfld.this<-paste(exp.this,fld.name.this)
        if (length(unique(lev.list)) > 1){
            expfld.this<-paste(expfld.this,' (L',lev.list[i],')',sep='')
        }
        exp<-c(exp,expfld.this)
        fname<-paste(paths[i],'series',files[j],sep='/')
        fld<-read.table(fname,header=TRUE)
        fld.date<-as.POSIXct(strptime(fld[,1],format="%Y%m%dT%H%M%S"))
        col<-lev.list[i]+1
        
        # Accumulate field x/y values for plots
        fld.x[[expfld.this]]<-fld.date
        fld.y[[expfld.this]]<-fld[,col]        
        fld.xmax<-max(fld.xmax,max(fld.x[[expfld.this]]),na.rm=TRUE)
        fld.xmin<-min(fld.xmin,min(fld.x[[expfld.this]]),na.rm=TRUE)
        fld.ymax<-max(fld.ymax,max(fld.y[[expfld.this]],na.rm=TRUE),na.rm=TRUE)
        fld.ymin<-min(fld.ymin,min(fld.y[[expfld.this]],na.rm=TRUE),na.rm=TRUE)
    }
}

# Override axis limits on user request
if (xlow > -Inf){fld.xmin<-date.init+xlow}
if (xhigh < Inf){fld.xmax<-date.init+xhigh}
if (ylow > -Inf){fld.ymin<-ylow}
if (yhigh < Inf){fld.ymax<-yhigh}

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
     type='n',
     xlim=c(fld.xmin,fld.xmax),
     ylim=c(fld.ymin,fld.ymax),
     xaxt='n',
     xlab='Valid date/time',
     ylab=fld.name.default,
     main=paste('Time Series of',fld.name.default,'at level',lev.list[1],sep=' '),
     cex.main=1.5,
     cex.lab=1.,
     cex.axis=1.)
axis.POSIXct(1,fld.date,cex.axis=1.)
for (i in 1:length(exp)){
    lines(x=fld.x[[exp[i]]],y=fld.y[[exp[i]]],
          lwd=lwd,
          type=plot.type,
          pch=pch.user,
          col=col.list[i])
}

# Add legend to plot
legend(x=legend.x,legend=exp,
       inset=0.01,
       col=col.list,
       lwd=rep(lwd,length(exp)))
dev.off()
