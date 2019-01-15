#include <stdio.h>

#include <unistd.h>
#include <alloca.h>

#include <stdlib.h>
#include "qstdir.h"
#include "convert_ip.h"

#include "proto.h"
//#include <fnom.h>
#include <rmnlib.h>
#include <string.h>

int c_fnom(int *iun,char *nom,char *type,int lrec);

void main() {
  int iunit,ni,nj,nk,key,istat;
  word *data;

  iunit = 0;
  istat = c_fnom(&iunit,"/cnfs/ops/production/gridpt/dbase/prog/gsloce/2015070706_042","RND+OLD+R/O",0);
  istat = c_fstouv(iunit,"RND");
  key = c_fstinf(iunit,&ni,&nj,&nk,-1,"            ", -1,-1,-1,"P@","TM  ");
  data = (word *) alloca(ni*nj*nk*4);
  //data = (word *) calloc(ni*nj*nk*4);
  istat = c_fstluk(data,key,&ni,&nj,&nk);
  printf("%f\n",data[0]);
  istat = c_fstfrm(iunit);
  istat = c_fclos(iunit);
  //free(data);
}
