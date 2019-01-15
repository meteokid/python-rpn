# Process command line arguments
args<-commandArgs(trailingOnly=TRUE)
var<-args[1]
date.str<-args[2]
run<-args[3]

# Acquire SCM support utilities
source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','utils.R',sep='/'))

# Acquire R-SCM API
source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','api.R',sep='/'))

# Read common inputs required for all calculations
t<-read.scm('T8',date.str,run)
p0<-read.scm('DIAGT',date.str,run,type='coord')
p<-read.scm('PREST',date.str,run,type='coord')
p.all<-cbind(p$v,p0$v*100)

# Retrieve constants for calculations
const<-utils.read.constants()

# Compute potential temperature on request
if (var == 'THETA') {
    th<-c()
    for (i in seq(1,ncol(p.all))){
   	th<-cbind(th,t$v[,i]*(100000./p.all[,i])^(const$RGASD/const$CPD))
    }
    theta<-new.scm('THETA',th,clone=t)
    write.scm(theta)
}
