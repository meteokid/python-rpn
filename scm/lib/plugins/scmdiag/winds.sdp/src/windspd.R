# Process command line arguments
args<-commandArgs(trailingOnly=TRUE)
var<-args[1]
date.str<-args[2]
run<-args[3]

# Acquire R-SCM API
source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','api.R',sep='/'))

# Read common inputs required for all calculations
u<-read.scm('U8',date.str,run)
v<-read.scm('V8',date.str,run)

# Compute wind speed on request
if (var == 'WSPD') {
    wspd<-new.scm('WSPD',sqrt(u$v^2+v$v^2),clone=u)
    write.scm(wspd)
}

# Compute wind direction on request
if (var == 'WDIR'){
    wdir<-new.scm('WDIR',(270.-atan2(u$v,v$v)*180/pi )%%360,clone=u)
    write.scm(wdir)
}
