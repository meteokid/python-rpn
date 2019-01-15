# This set of R function implements a basic API for SCM outputs.  Acquire these
# functions using something like source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','api.R',sep='/'))

# Example (computing wind speed from components):
#   source(paste(Sys.getenv('SCM_SCRIPTS_LIBPATH'),'R','api.R',sep='/'))
#   u<-read.scm('U8',mydate,myrun)
#   v<-read.scm('V8',mydate,myrun)
#   wspd<-new.scm('WSPD',sqrt(u$v^2+v$v^2),clone=u)
#   write.scm(wspd) 

read.scm<-function(name,dateo,run,type='series'){
    # Read a field from an SCM output file
    fn<-fname._scm(name,dateo,run,type)
    d<-read.table(fn,header=TRUE)
    self<-list("date"=strptime(d$date,format='%Y%m%dT%H%M%SZ'),"v"=as.matrix(d[,2:(ncol(d))]),
               "dateo"=dateo,"run"=run,"type"=type,"name"=name)
    colnames(self$v)<-seq(1,ncol(self$v))
    return(self)
}

write.scm<-function(self){
    # Write a field to an SCM output file
    fn<-fname._scm(self$name,self$dateo,self$run,self$type)
    d<-as.data.frame(cbind(strftime(self$date,format='%Y%m%dT%H%M%SZ'),self$v))
    colnames(d)[1]<-'date'
    write.table(d,fn,row.names=FALSE,quote=FALSE)
}

new.scm<-function(name,v,clone){
    # Create a new SCM field with metadata copied from an existing field
    self<-clone
    self$name<-name
    self$v<-v
    colnames(self$v)<-seq(1,ncol(v))
    return(self)
}

update.scm<-function(key,value){
    # Adjust the data or metadata of an SCM field
    self[key]<-value
}

fname._scm<-function(name,dateo,run,type){
    # Internal function for file name construction
    return(paste(run,'/',type,'/',name,'_',dateo,'.txt',sep=''))
}

