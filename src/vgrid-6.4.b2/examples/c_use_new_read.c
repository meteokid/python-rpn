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
#include "vgrid.h"
#include "armnlib.h"

int c_use_new_read () {

  int ier, iun = 10, iun2 = 11;
  int *i_val = NULL;
  int nl_t, nt, ni, nj, nk, k;
  char filename[]="../tests/data/dm_5005_from_model_run";
  char mode[]="RND+R/O";
  float *f_val = NULL;
  double *a_8_t = NULL, *b_8_t = NULL, *table = NULL;
  vgrid_descriptor *vgd = NULL, *vgd2 = NULL;
#include "vgrid_version.hc"
  
  printf("%s\n", vgrid_descriptors_version);

  printf("c_use_new_read\n");

  ier = c_fnom(&iun,filename,mode,0);
  if( ier < 0 ) {
    printf("ERROR with c_fnom on iun, file %s\n", filename);
    return 1;
  }
  ier = c_fstouv(iun,"RND","");  
  if( ier < 0 ) {
    printf("ERROR with c_fstouv on iun, file %s\n", filename);
    return 1;
  }
  
  if( Cvgd_new_read(&vgd, iun, -1, -1, -1, -1) == VGD_ERROR ) {
    printf("ERROR with Cvgd_new_read on iun\n");
    return 1;
  }
  if( Cvgd_get_int_1d(vgd, "VIPT", &i_val, &nt, -1) ==  VGD_ERROR ) {
    printf("ERROR with Cvgd_get_int for VIPT\n");
    return 1;
  }
  if( Cvgd_get_float_1d(vgd, "VCDT", &f_val, &nt, -1) ==  VGD_ERROR ) {
    printf("ERROR with Cvgd_get_float_1d for VCDT\n");
    return 1;
  }
  if( Cvgd_get_double_1d(vgd, "CA_T", &a_8_t, &nt, -1) ==  VGD_ERROR ) {
    printf("ERROR with Cvgd_get_double_1d for CA_T\n");
    return 1;
  }
  if( Cvgd_get_double_1d(vgd, "CB_T", &b_8_t, &nt, -1) ==  VGD_ERROR ) {
    printf("ERROR with Cvgd_get_double_1d for CB_T\n");
    return 1;
  }

  // Size of thermo may also be obtained by this:
  ier = Cvgd_get_int(vgd, "NL_T", &nl_t, -1);
  if(nl_t != nt ) {
    printf("ERROR: nt and nl_t should be equal, got %d, %d\n",nt, nl_t);
    return(-1);
  }
  printf("nl_t = %d\n", nl_t);
  
  for( k = 0; k < nl_t; k++) {
    printf("k = %d, ip1 = %d, val = %f, A = %f, B = %f\n",
  	   k, i_val[k], f_val[k], a_8_t[k], b_8_t[k]);
  }

  // Load table (this is the actual data in fst record !! which may also be
  // obtained with fstlir, but why do it if vgd already contains it!)
  if ( Cvgd_get_double_3d(vgd, "VTBL", &table, &ni, &nj, &nk, -1) ==  VGD_ERROR ) {
    printf("ERROR with Cvgd_double_3d for VTBL\n");
    return 1;
  }
  
  // Constructing new vgd with this table
  if ( Cvgd_new_from_table(&vgd2, table, ni, nj, nk) ==  VGD_ERROR ) {
    printf("ERROR with Cvgd_new_from_table for VTBL\n");
    return 1;
  }

  // Comparing new table with original table, must be the same.
  if( Cvgd_vgdcmp(vgd, vgd2) != 0 ){
    printf("ERROR, vgd and vgd2 should be the same\n");
    return(1);
  }

  // Write descriptor in new file
  ier = c_fnom(&iun2,"to_erase","RND",0);
  if( ier < 0 ) {
    printf("ERROR with c_fnom on iun2\n");
    return 1;
  }
  ier = c_fstouv(iun2,"RND","");  
  if( ier < 0 ) {
    printf("ERROR with c_fstouv on iun2\n");
    return 1;
  }
  if( Cvgd_write_desc(vgd, iun2) == VGD_ERROR ){
    printf("ERROR with Cvgd_write_desc on iun2\n");
    return 1;
  }
  ier = c_fstfrm(iun2);
  ier = c_fclos(iun2);

  // Re open, read and compare
  ier = c_fnom(&iun2,"to_erase",mode,0);
  if( ier < 0 ) {
    printf("ERROR with c_fnom on iun2\n");
    return 1;
  }
  ier = c_fstouv(iun2,"RND","");  
  if( ier < 0 ) {
    printf("ERROR with c_fstouv on iun2\n");
    return 1;
  }
  if( Cvgd_new_read(&vgd2, iun, -1, -1, -1, -1) == VGD_ERROR ) {
    printf("ERROR with Cvgd_new_read vgd2\n");
    return 1;
  }
  if( Cvgd_vgdcmp(vgd, vgd2) != 0 ){
    printf("ERROR, vgd and vgd2 shouldne the same after write in file, read from file\n");
    return(1);
  }

  Cvgd_free(&vgd);
  Cvgd_free(&vgd2);
  free(table);
  free(i_val);
  free(f_val);
  free(a_8_t);
  free(b_8_t);

  return(0);
}
