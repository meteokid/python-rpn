/* libdescrip - Vertical grid descriptor library for FORTRAN programming
 * Copyright (C) 2016  Direction du developpement des previsions nationales
 *                     Centre meteorologique canadien
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation,
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vgrid.h"
#include "armnlib.h"

char *filenames[] = {
    "../tests/data/dm_1001_from_model_run",
    "../tests/data/dm_1002_from_model_run",
    "../tests/data/2001_from_model_run",
    "../tests/data/dm_5001_from_model_run",
    "../tests/data/dm_5002_from_model_run",
    "../tests/data/dm_5003_from_model_run",
    "../tests/data/dm_5004_from_model_run",
    "../tests/data/dm_5005_from_model_run",
    "../tests/data/dm_5100_from_model_run",
    "../tests/data/dm_5999_from_model_run",
};

#define n_file (sizeof (filenames) / sizeof (const char *))

static int similar_vec_float(float *vec1, int n1, float *vec2, int n2) {
  int i;
  if(vec1) {
    if (vec2) {
      if ( n1 == n2 ) {
	for(i = 0; i < n1; i++) {
	  if( fabs(vec1[i]) < 1.e-37 ){
	    if( fabs(vec2[i]) > 1.e-37 ){
	      printf("Vector differs: val1=%f, val2=%f\n", vec1[i], vec2[i]);
	      return(-1);
	    }
	  } else {
	    if ( fabs(vec1[i]-vec2[i])/fabs(vec1[i]) > 1.e-5 ){
	      printf("Vector differs: val1=%f, val2=%f\n", vec1[i], vec2[i]);
	      return(-1);
	    }
	  }
	}
      } else {
	// Vectors are not the same size.
	return(-2);
      }
    } else {
      // vec2 not allocated
      return(-3);
    }
  }
  // Vector are the same or are not allocated.
  return(0);
}

static int comp_pres(char *filename, int ind) {
  int ier, iun, key, ni, nj, nk, nijk, k, ni2, nj2, nk2, ij, ijk, *ip1_list = NULL;
  iun = 10 + ind;
  float *rfld_2d = NULL, *rfls_2d = NULL, *levels = NULL, *levels2 = NULL;
  vgrid_descriptor *vgd = NULL;
  char rfld_S[VGD_LEN_RFLD];
  char rfls_S[VGD_LEN_RFLS];
  
  ier = c_fnom(&iun,filename,"RND+R/O",0);
  if( ier < 0 ) {
    printf("ERROR with c_fnom on iun, file %s\n", filename);
    return(VGD_ERROR);
  }
  ier = c_fstouv(iun,"RND","");
  if( ier < 0 ) {
    printf("ERROR with c_fstouv on iun, file %s\n", filename);
    return(VGD_ERROR);
  }
  if( Cvgd_new_read(&vgd, iun, -1, -1, -1, -1) == VGD_ERROR ) {
    printf("ERROR with Cvgd_new_read on iun\n");
    return(VGD_ERROR);
  }

  // Read RFLD ?
  ier = Cvgd_get_char(vgd, "RFLD", rfld_S, 1);
  if( strcmp(rfld_S, VGD_NO_REF_NOMVAR) == 0 ){
    printf("   The current Vcode has no RFLD field\n");
    // Get grid size from TT
    key = c_fstinf( iun, &ni, &nj, &nk, -1, " ", -1, -1, -1, " ", "TT");
    if(key < 0){
      printf("Problem getting info for TT");
      return(VGD_ERROR);
    }
    // Allocate the surface field rfld_2d since it will be used to get the horizontal problem size
    // in the vgrid library. But the value in this surface field will not be used.
    rfld_2d = malloc(ni*nj * sizeof(float));
    if(! rfld_2d){
      printf("Problem allocating rfld_2d of size %d\n",ni*nj);
      return(VGD_ERROR);
    }
    for( ij = 0; ij < ni*nj; ij++){
      rfld_2d[ij] = 0.f;
    }
  } else {
    printf("   RFLD='%s'\n", rfld_S);
    key = c_fstinf( iun, &ni, &nj, &nk, -1, " ", -1, -1, -1, " ", rfld_S);
    if(key < 0){
      printf("Problem getting info for %s", rfld_S);
      return(VGD_ERROR);
    }
    rfld_2d = malloc(ni*nj * sizeof(float));
    if(! rfld_2d){
      printf("Problem allocating rfld_2d of size %d\n",ni*nj);
      return(VGD_ERROR);
    }
    ier = c_fstluk( rfld_2d, key, &ni, &nj, &nk );
    if(ier < 0){
      printf("Problem with fstluk for %s\n",rfld_S);
      return(VGD_ERROR);
    }
    if( strcmp(rfld_S,"P0  ") == 0 ){
      for( ij = 0; ij < ni*nj; ij++){
	rfld_2d[ij] = rfld_2d[ij]*100.f;
      }
    }    
  }

  // Read RFLS (large scale ef field for SLEVE) ?
  ier = Cvgd_get_char(vgd, "RFLS", rfls_S, 1);
  if( strcmp(rfls_S, VGD_NO_REF_NOMVAR) == 0 ){
    printf("   The current Vcode has no RFLS (large scale SLEVE) field\n");
  } else {
    printf("   RFLS='%s'\n", rfls_S);
    key = c_fstinf( iun, &ni2, &nj2, &nk2, -1, " ", -1, -1, -1, " ", rfls_S);
    if(key < 0){
      printf("Problem getting info for %s", rfls_S);
      return(VGD_ERROR);
    }
    if(ni2 != ni || nj2 != nj || nk2 != nk){
      printf("Size problem with %s\n", rfls_S);
      return(VGD_ERROR);
    }
    rfls_2d = malloc(ni*nj * sizeof(float));
    if(! rfls_2d){
      printf("Problem allocating rfls_2d of size %d\n",ni*nj);
      return(VGD_ERROR);
    }
    ier = c_fstluk( rfls_2d, key, &ni, &nj, &nk );
    if(ier < 0){
      printf("Problem with fstluk for %s\n",rfls_S);
      return(VGD_ERROR);
    }
    if( strcmp(rfls_S,"P0LS") == 0 ){
      for( ij = 0; ij < ni*nj; ij++){
	rfls_2d[ij] = rfls_2d[ij]*100.f;
      }
    }
  }

  // Compute pressure for momentum level
  if( Cvgd_get_int_1d(vgd, "VIPM", &ip1_list, &nk, 0) ){
    printf("Error with Cvgd_get_int on VIPM\n");
    return(VGD_ERROR);
  }
  nijk=ni*nj*nk;
  levels = malloc(nijk * sizeof(float));
  if(! levels){
    printf("Problem allocating levels of size %d\n",ni*nj);
    return(VGD_ERROR);
  }
  if( Cvgd_levels_2ref(vgd, ni, nj, nk, ip1_list, levels, rfld_2d, rfls_2d, 0) ){
    printf("Problem with Cvgd_levels_2ref\n");
    return(VGD_ERROR);
  }

  // Compare computed pressure with PX in file
  levels2 = malloc(nijk * sizeof(float));
  if(! levels2){
    printf("Problem allocating levels2 of size %d\n",ni*nj);
    return(VGD_ERROR);
  }
  for( k=0; k < nk; k++){
    key = c_fstinf( iun, &ni2, &nj2, &nk2, -1, " ", ip1_list[k], -1, -1, " ", "PX");
    if(key < 0){
      printf("Problem getting info for PX for ip1=%d\n", ip1_list[k]);
      return(VGD_ERROR);
    }
    if(ni2 != ni || nj2 != nj){
      printf("Size problem with PX for ip1=%d\n", ip1_list[k] );
      return(VGD_ERROR);
    }
    ier = c_fstluk(levels2+ni*nj*k, key, &ni2, &nj2, &nk2 );
  }
  for( ijk=0; ijk < nijk; ijk++){
    levels2[ijk] = levels2[ijk]*100.f;
  }
  if( similar_vec_float(levels, 1, levels2, 1) == 0 ){
    printf(">>>> pressure is the same\n");
  } else {
    printf(">>>> pressure differs\n");
  }

  free(rfld_2d);
  free(rfls_2d);
  free(levels);
  free(levels2);
  ier = c_fstfrm(iun);
  return(VGD_OK);
}

int c_pressure_for_all_vcode() {
  int i, ier, status = VGD_OK;
  if( Cvgd_putopt_int("ALLOW_SIGMA",1) == VGD_ERROR ){
    printf ("ERROR with option ALLOW_SIGMA\n");
    return(1);
  }
  for (i = 0; i < (int) n_file; i++) {
    printf ("Computing pressure for %s\n", filenames[i]);
    if(comp_pres(filenames[i],i) == VGD_ERROR) {
      status = VGD_ERROR;
      exit(1);
    }
  }
  return(status);
}
