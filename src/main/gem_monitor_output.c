#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

main_gem_monitor_output(int argc, char **argv){
int i, fd;
char buffer[32768];
pid_t pp=getppid();

if(argc-1 & 1) { printf("argument count must be even \n"); exit(1); }
if(fork()) exit(0);
while(1){
  if(kill(pp,0)) exit(0);
  for(i=1 ; i<argc ; i+=2){
    if( (fd=open(argv[1],O_RDONLY )) >= 0 ) {
      close(fd);
      printf("file=%s,cmd=%s\n",argv[i],argv[i+1]);
      snprintf(buffer,sizeof(buffer)-1,"%s %s",argv[i+1],argv[i]);
      printf("Executing:%s\n",buffer);
      system(buffer);
      printf("Deleting:%s\n",argv[i]);
      unlink(argv[i]);
      }
    }
  sleep(2);
  }
}
