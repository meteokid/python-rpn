/* #define DEBUG On */
#include <rpnmacros.h>
#include <Python.h>
#include <numpy/arrayobject.h>

int c_fst_data_length(int length_type);

static void imprime_ca(char *varname, float *array, int nb);

/* # include "numpy/oldnumeric.h" */

#define FTN_Style_Array 1
static PyObject *ErrorObject;

static int datyps[32]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                       -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
static int lentab[32]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

/* initialize the table giving the length of the array element types */
static init_lentab()
{
	lentab[NPY_CHAR]=sizeof(char);
	lentab[NPY_UBYTE]=sizeof(unsigned char);
	lentab[NPY_BYTE]=sizeof(signed char);
	lentab[NPY_SHORT]=sizeof(short);
	lentab[NPY_INT]=sizeof(int);
	lentab[NPY_LONG]=sizeof(long);
	lentab[NPY_FLOAT]=sizeof(float);
	lentab[NPY_DOUBLE]=sizeof(double);
	lentab[NPY_CFLOAT]=2*sizeof(float);
	lentab[NPY_CDOUBLE]=2*sizeof(double);
	lentab[NPY_OBJECT]=sizeof(PyObject *);

	datyps[NPY_CHAR]=-1;
	datyps[NPY_UBYTE]=-1;
	datyps[NPY_BYTE]=-1;
	datyps[NPY_SHORT]=-1;
	datyps[NPY_INT]=2;
	datyps[NPY_LONG]=2;
	datyps[NPY_FLOAT]=1;
	datyps[NPY_DOUBLE]=-1;
	datyps[NPY_CFLOAT]=-1;
	datyps[NPY_CDOUBLE]=-1;
	datyps[NPY_OBJECT]=-1;
}
/* ----------------------------------------------------- */
#if defined (mips) || defined (__mips)
void __dshiftr4()
{
printf("__dshiftr4 called\n");
exit(1);
}
void __mask4()
{
printf("__mask4 called\n");
exit(1);
}
void __dshiftl4()
{
printf("__dshiftl4 called\n");
exit(1);
}
#endif

void imprime_ca(char *varname, float *tabl, int nb)
{
  int i;

  for (i=0; i < nb; i++)
     printf("%s[%d]=%f\n",varname,i,tabl[i]);
}

static char Fstdc_fstouv__doc__[] =
"Interface to fstouv and fnom to open a RPN 2000 Standard File"
;

static PyObject *
Fstdc_fstouv(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int iun=0;
	char *filename="None";
	char *options="RND";

	if (!PyArg_ParseTuple(args, "iss",&iun,&filename,&options))
		return NULL;
	if( c_fnom(&iun,filename,options,0) < 0 ){  /* fnom failed */
	  Py_INCREF(Py_None);
	  return Py_None;
	}
	if( c_fstouv(iun,filename,options) < 0 ){  /* fstouv failed */
	  Py_INCREF(Py_None);
	  return Py_None;
	}
	printf("Opened file %s, unit=%d, with options %s\n",filename,iun,options);
	return Py_BuildValue("i",iun);   /* return unit number if O.K.  */
}

static char Fstdc_fstvoi__doc__[] =
"Interface to fstvoi and fnom and fstouv view a RPN Standard File"
;
static PyObject *
Fstdc_fstvoi(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	char *options="NEWSTYLE";
	int iun, ier;

	if (!PyArg_ParseTuple(args, "is",&iun,&options))
		return NULL;

	ier = c_fstvoi(iun,options);
	return Py_None;
}

static char Fstdc_fstfrm__doc__[] =
"Interface to fclos to close a RPN 2000 Standard File"
;

static PyObject *
Fstdc_fstfrm(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int iun=0;

	if (!PyArg_ParseTuple(args, "i",&iun))
		return NULL;
	printf("Closed file %d\n",iun);
	c_fstfrm(iun);
	c_fclos(iun);
	Py_INCREF(Py_None);
	return Py_None;
}

static char Fstdc_fstsui__doc__[] =
"Interface to fstsui, returns record handle + record keys in a dictionary"
;

static PyObject *
Fstdc_fstsui(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int iun, ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0;
	char TYPVAR[3]={' ',' ','\0'};
	char NOMVAR[5]={' ',' ',' ',' ','\0'};
	char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
	char GRTYP[2]={' ','\0'};
	int handle, junk;
	int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
	int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;


	if (!PyArg_ParseTuple(args, "i",&iun))
		return NULL;
	handle=c_fstsui(iun,&ni,&nj,&nk);
	if(handle >= 0) {
          junk =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,&nbits,&datyp,&ip1,&ip2,&ip3,
                  TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
                  &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
	  }
	if(junk < 0 || handle < 0) {
	  Py_INCREF(Py_None);
	  return Py_None;
	}
	return Py_BuildValue("(i{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:s,s:s,s:s,s:s,s:i,s:i,s:i,s:i})",
               handle,"ni",ni,"nj",nj,"nk",nk,"dateo",dateo,"ip1",ip1,"ip2",ip2,"ip3",ip3,
	       "deet",deet,"npas",npas,"datyp",datyp,"nbits",nbits,
	       "type",TYPVAR,"nom",NOMVAR,"etiket",ETIKET,"grtyp",GRTYP,
	       "ig1",ig1,"ig1",ig2,"ig3",ig3,"ig4",ig4);
}

static char Fstdc_fstinf__doc__[] =
"Interface to fstinf, dsfsui, fstinfx, returns a record handle and record keys in a dictionary"
;

static PyObject *
Fstdc_fstinf(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int iun, inhandle=-2, ni=0, nj=0, nk=0, datev=0, ip1=0, ip2=0, ip3=0;
	char *typvar, *nomvar, *etiket;
	char TYPVAR[3]={' ',' ','\0'};
	char NOMVAR[5]={' ',' ',' ',' ','\0'};
	char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
	char GRTYP[2]={' ','\0'};
	int handle, junk;
	int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
	int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;

	if (!PyArg_ParseTuple(args, "isssiiiii",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev,&inhandle))
		return NULL;
#if defined(DEBUG)
printf("Looking for iun=%d,nomvar=:%s:,typvar=:%s:,etiket=:%s:,ip1=%d,ip2=%d,ip3=%d,datev=%d,handle=%d\n",
        iun,nomvar,typvar,etiket,ip1,ip2,ip3,datev,inhandle);
#endif
	if(inhandle == -2) {
	  handle=c_fstinf(iun,&ni,&nj,&nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar);
	}
	else if(inhandle == -1) {
	  handle=c_fstsui(iun,&ni,&nj,&nk);
	}
	else if(inhandle >= 0) {
#if defined(DEBUG)
printf("c_fstinfx\n");
#endif
	  handle=c_fstinfx(inhandle,iun,&ni,&nj,&nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar);
	}
	else
	  {Py_INCREF(Py_None); return Py_None;} /* error, return None */
#if defined(DEBUG)
printf("Found handle=%d\n",handle);
#endif
	if(handle < 0) {Py_INCREF(Py_None); return Py_None;} /* error, return None */
        junk =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,&nbits,&datyp,&ip1,&ip2,&ip3,
                TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
                &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
#if defined(DEBUG)
printf("c_fstprm return=%d\n",junk);
#endif
	if(junk < 0) {Py_INCREF(Py_None); return Py_None;} /* error, return None */
#if defined(DEBUG)
        printf("Debug fstinf ni=%d nj=%d dateo=%d datev=%d\n",ni,nj,dateo,extra1);
#endif
	return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:s,s:s,s:s,s:s,s:i,s:i,s:i,s:i,s:i}",
               "handle",handle,"ni",ni,"nj",nj,"nk",nk,"dateo",dateo,"ip1",ip1,"ip2",ip2,"ip3",ip3,
	       "deet",deet,"npas",npas,"datyp",datyp,"nbits",nbits,
	       "type",TYPVAR,"nom",NOMVAR,"etiket",ETIKET,"grtyp",GRTYP,
	       "ig1",ig1,"ig2",ig2,"ig3",ig3,"ig4",ig4,"date",extra1);
}

static char Fstdc_fnom__doc__[] =
"Interface to fnom (unused)"
;

static PyObject *
Fstdc_fnom(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{

	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static char Fstdc_fsteff__doc__[] =
"Interface to fsteff"
;

static PyObject *
Fstdc_fsteff(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int handle=0;
	int status=0;

	if (!PyArg_ParseTuple(args, "i",&handle))
		return NULL;

	status=c_fsteff(handle);

	return Py_BuildValue("i",status);
}
static char Fstdc_fstecr__doc__[] =
"Interface to fstecr"
;

static PyObject *
Fstdc_fstecr(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int iun, ip1=0, ip2=0, ip3=0;
	char *typvar, *nomvar, *etiket, *grtyp;
	int dateo=0, deet=0, npas=0, nbits=0, ig1=0, ig2=0, ig3=0, ig4=0;
	int ni=0,nj=0,nk=0,datyp=0,rewrit=0;
        int dtl=4;                 /* default data type length of 4 bytes */
	int strides[3]={0,0,0},strides2[3]={0,0,0};
	PyArrayObject *array;

	if (!PyArg_ParseTuple(args, "Oisssiiiisiiiiiii",
	     &array,&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&dateo,&grtyp,&ig1,&ig2,&ig3,&ig4,&deet,&npas,&nbits))
		return NULL;

#if defined(DEBUG)
        printf("fstecr array->flags=%d \n",array->flags);
        printf("NPY_F_CONTIGUOUS=%d   NPY_FARRAY=%d  NPY_C_CONTIGUOUS=%d\n",NPY_F_CONTIGUOUS,NPY_FARRAY,NPY_C_CONTIGUOUS);
#endif
	if( PyArray_ISCONTIGUOUS(array) || (array->flags & NPY_FARRAY)) {
	  ni=array->dimensions[0]; strides[0]=array->strides[0];
	  if(array->nd > 1){ nj=array->dimensions[1];strides[1]=array->strides[1];}
	  if(array->nd > 2){ nk=array->dimensions[2];strides[2]=array->strides[2];}
          switch (array->descr->type_num) {

            case NPY_INT:
              datyp=4;
              dtl=4;
              break;

	    case NPY_LONG:
              if (sizeof(long)!=4) printf("PyFstecr: warning sizeof(long)=%d\n",sizeof(long));
              datyp=4;
              dtl=4;
              break;

            case NPY_SHORT:
              datyp=4;
              dtl=2;
              break;

	    case NPY_FLOAT:
              datyp=134;
              dtl=4;
              break;

	    case NPY_DOUBLE:
              datyp=1;
              dtl=8;
              break;

	    default:
	      printf("PyFstecr: unsupported data type :%c\n",array->descr->type);
	      Py_INCREF(Py_None);
	      return Py_None;
              break;

	  }
	}
	else {
	   printf("array is not CONTIGUOUS in memory\n");
	  Py_INCREF(Py_None);
	  return Py_None;
	}

#if defined(DEBUG)
printf("datyp=%d dtl=%d\n",datyp,dtl);
printf("writing array, nd=%d,ni=%d,nj=%d,nk=%d,datyp=%d,element length=%d,fstdtyp=%d\n",array->nd,ni,nj,nk,datyp,lentab[array->descr->type_num],datyps[array->descr->type_num]);
printf("writing array with strides=%d,%d,%d\n",strides[0],strides[1],strides[2]);
printf("writing iun=%d,nomvar=:%s:,typvar=:%s:,etiket=:%s:,ip1=%d,ip2=%d,ip3=%d,\
dateo=%d,grtyp=:%s:,ig1=%d,ig2=%d,ig3=%d,ig4=%d,deet=%d,npas=%d,nbits=%d\n",
        iun,nomvar,typvar,etiket,ip1,ip2,ip3,dateo,grtyp,ig1,ig2,ig3,ig4,deet,npas,nbits);
/* printf("%x %x %x\n",array->data[0],array->data[4],array->data[8]); */
#endif
	c_fstecr(array->data,array->data,-nbits,iun,dateo,deet,npas,ni,nj,nk,ip1,ip2,ip3,
	         typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp+c_fst_data_length(dtl),rewrit);
/*	         typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp,rewrit); */
/*
*/
	Py_INCREF(Py_None);
	return Py_None;
}

static char Fstdc_fstluk__doc__[] =
"Interface to fstluk, returns record keys in a dictionary + Numeric.array object"
;

static PyObject *
Fstdc_fstluk(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	PyArrayObject *newarray;
	int dimensions[3]={1,1,1}, strides[3]={0,0,0}, ndimensions=3;
	int type_num=NPY_FLOAT;

	int iun, ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0, temp;
	char TYPVAR[3]={' ',' ','\0'};
	char NOMVAR[5]={' ',' ',' ',' ','\0'};
	char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
	char GRTYP[2]={' ','\0'};
	int handle, junk;
	int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
	int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;

	if (!PyArg_ParseTuple(args, "i",&handle))
		return NULL;
#if defined(DEBUG)
printf("handle = %d \n",handle);
#endif
	if(handle >= 0) {
          junk =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,&nbits,&datyp,&ip1,&ip2,&ip3,
                  TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
                  &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
	  }
	if(junk < 0 || handle < 0) {
	  Py_INCREF(Py_None);
	  return Py_None;
	}
	if (datyp == 0 || datyp == 2 || datyp == 4) type_num=NPY_INT;
	else if (datyp == 1 || datyp == 5 || datyp == 134) type_num=NPY_FLOAT;
	else if (datyp == 3 ) type_num=NPY_CHAR;
	else {
	  printf("PyFstluk: unrecognized data type : %d\n",datyp);
	  Py_INCREF(Py_None);
	  return Py_None;
	}

	dimensions[0] = (ni>1) ? ni : 1  ;
	dimensions[1] = (nj>1) ? nj : 1 ;
	dimensions[2] = (nk>1) ? nk : 1 ;
	strides[0]=4 ; strides[1]=strides[0]*dimensions[0] ; strides[2]=strides[1]*dimensions[1];
	if(nk>1) ndimensions=3;
	else if(nj>1) ndimensions=2;
	else ndimensions=1;

        newarray = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndimensions,dimensions,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);
#if defined(DEBUG)
	if(nk>1){
	  printf("Creating newarray with dimensions %d %d %d\n", (ni>1) ? ni : 1 ,(nj>1) ? nj : 1, nk );
	  printf("Strides= %d %d %d \n",newarray->strides[0],newarray->strides[1],newarray->strides[2]);
          }
        else if(nj>1){
	  printf("Creating newarray with dimensions %d %d\n", (ni>1) ? ni : 1,(nj>1) ? nj : 1);
	  printf("Strides= %d %d \n",newarray->strides[0],newarray->strides[1]);
          }
        else {
	  printf("Creating newarray with dimension %d\n",(ni>1) ? ni : 1);
	  printf("Strides= %d \n",newarray->strides[0]);
          }
#endif
	junk = c_fstluk(newarray->data,handle,&ni,&nj,&nk);
	if( junk >= 0 )
	  return Py_BuildValue("O",newarray);
	else {
	  Py_DECREF(newarray);
	  Py_INCREF(Py_None);
	  return Py_None;
	  }

}

static char Fstdc_fst_edit_dir__doc__[] =
"Interface to fst_edit_dir"
;

static PyObject *
Fstdc_fst_edit_dir(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int handle=0;
	int status=0;
	int ip1=-1, ip2=-1, ip3=-1;
	char *typvar, *nomvar, *etiket, *grtyp;
	int date=-1, deet=-1, npas=-1, nbits=-1, ig1=-1, ig2=-1, ig3=-1, ig4=-1;
	int ni=-1,nj=-1,nk=-1,datyp=-1;

	if (!PyArg_ParseTuple(args, "iiiiiiiiiissssiiiii",
                &handle,&date,&deet,&npas,&ni,&nj,&nk,&ip1,&ip2,&ip3,&typvar,&nomvar,&etiket,&grtyp,&ig1,&ig2,&ig3,&ig4,&datyp))
		return NULL;

	status=c_fst_edit_dir(handle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp);

	return Py_BuildValue("i",status);
}

static char Fstdc_newdate__doc__[] =
"Interface to newdate"
;

static PyObject *
Fstdc_newdate(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int date1=0,date2=0,date3=0,mode=0;
	int status=0;
	wordint fdat1,fdat2,fdat3,fmode;

	if (!PyArg_ParseTuple(args, "iiii",&date1,&date2,&date3,&mode)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}

	fdat1 = date1; fdat2 = date2; fdat3=date3;
	fmode = mode;
	status=f77name(newdate)(&fdat1,&fdat2,&fdat3,&fmode);
#if defined(DEBUG)
	printf("Fstdc_newdate fdat1=%d,fdat2=%d,fdat3=%d\n",fdat1,fdat2,fdat3);
#endif
	return Py_BuildValue("(iii)",fdat1,fdat2,fdat3);
}
static char Fstdc_difdatr__doc__[] =
"Interface to difdatr"
;

static PyObject *
Fstdc_difdatr(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
        int date1=0,date2=0;
	int status=0;
	double nhours;
	wordint fdat1,fdat2;

	if (!PyArg_ParseTuple(args, "ii",&date1,&date2)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
	fdat1 = date1; fdat2 = date2;
	status=f77name(difdatr)(&fdat1,&fdat2,&nhours);
#if defined(DEBUG)
	printf("Fstdc_difdatr date1=%d,date2=%d,nhours=%f\n",date1,date2,nhours);
#endif
	return Py_BuildValue("d",nhours);
}

static char Fstdc_incdatr__doc__[] =
"Interface to incdatr"
;

static PyObject *
Fstdc_incdatr(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
        int date1=0,date2=0;
	int status=0;
	double nhours;
	wordint fdat1,fdat2;

	if (!PyArg_ParseTuple(args, "id",&date2,&nhours)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
	fdat2 = date2;
	status=f77name(incdatr)(&fdat1,&fdat2,&nhours);
	date1 = fdat1;
#if defined(DEBUG)
	printf("Fstdc_incdatr date1=%d,date2=%d,nhours=%f\n",date1,date2,nhours);
#endif
	return Py_BuildValue("i",date1);
}

static char Fstdc_datematch__doc__[] =
"Determine if date stamp match search crieterias"
;

static PyObject *
Fstdc_datematch(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
        int datelu, debut, fin;
	float delta;
        double nhours,modulo,ddelta=delta;
	float toler=.00023;            /* tolerance d'erreur de 5 sec */

	if (!PyArg_ParseTuple(args, "iiif",&datelu,&debut,&fin,&delta)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
	printf("Debug datelu=%d debut=%d fin=%d delta=%f\n",datelu,debut,fin,delta);
	if ((fin != -1) && (datelu > fin)) return Py_BuildValue("i",0);
	if (debut != -1) {
	  if (datelu < debut) return Py_BuildValue("i",0);
	  f77name(difdatr)(&datelu,&debut,&nhours);
	}
	else {
	  if (fin == -1) return Py_BuildValue("i",1);          /* debut et fin = -1 */
	  f77name(difdatr)(&fin,&datelu,&nhours);
	}
	printf("Debug nhours=%f\n",nhours);
	modulo = fmod(nhours,ddelta);
	printf("Debug modulo=%f\n",modulo);
	if ((modulo < toler) || ((delta - modulo) < toler))
	  return Py_BuildValue("i",1);
	else
	  return Py_BuildValue("i",0);
}

static char Fstdc_level_to_ip1__doc__[] =
"Interface to convip: encode level value to ip1\n @param level_list list of level values (list of float)\n @param kind type of level (int)\n @return [(ip1new,ip1old),...] (list of tuple of int)"
;

static PyObject *
Fstdc_level_to_ip1(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
        int i,kind, nelm, status;
	float level;
        long ipnew, ipold;
	wordint fipnew, fipold, fmode, flag=0, fkind;
	char strg[30];
	PyObject *level_list, *ip1_list=Py_None, *item, *ipnew_obj, *ipnewold_obj;
	int convip();

        printf("Debug Fstdc_level_to_ip1 [Begin]\n");
	if (!PyArg_ParseTuple(args, "Oi",&level_list,&kind)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
	fkind = (wordint) kind;
	nelm = PyList_Size(level_list);
        printf("Debug Fstdc_level_to_ip1 kind=%d nelm=%d\n",kind,nelm);
	ip1_list = PyList_New(0);
	Py_INCREF(ip1_list);
	for (i=0; i < nelm; i++) {
	  item = PyList_GetItem(level_list,i);
	  level = (float) PyFloat_AsDouble(item);
          printf("Debug Fstdc_level_to_ip1 level=%f\n",level);
	  fmode = 2;
	  status=f77name(convip)(&fipnew,&level,&fkind,&fmode,&strg,&flag,30);
	  fmode = 3;
	  status=f77name(convip)(&fipold,&level,&fkind,&fmode,&strg,&flag,30);
            printf("Debug Fstdc_level_to_ip1 level=%f kind=%d ipold=%d ipnew=%d \n",level,kind,ipold,ipnew);
            ipnew = (long) fipnew;
            ipold = (long) fipold;
            ipnewold_obj = Py_BuildValue("(l,l)",ipnew,ipold);
            Py_INCREF(ipnewold_obj);
            PyList_Append(ip1_list,ipnewold_obj);
	}
	return (ip1_list);
}

static char Fstdc_ip1_to_level__doc__[] =
"Interface to convip: decode ip1 to level type,value, return list of tuple (level,kind)"
;

static PyObject *
Fstdc_ip1_to_level(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
        int i,kind, nelm, status;
	float level;
	wordint fip1, fmode, flag=0, fkind;
	char strg[30];
	PyObject *ip1_list=Py_None, *level_list, *item, *level_kind_obj;
	int convip();

	if (!PyArg_ParseTuple(args, "O",&ip1_list)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
        fmode = -1;
	nelm = PyList_Size(ip1_list);
	level_list = PyList_New(0);
	Py_INCREF(level_list);
	for (i=0; i < nelm; i++) {
	  item  = PyList_GetItem(ip1_list,i);
	  fip1  = (wordint) PyLong_AsLong(item);
	  status=f77name(convip)(&fip1,&level,&fkind,&fmode,&strg,&flag,30);
          kind  = (int) fkind;
          level_kind_obj = Py_BuildValue("(f,i)",level,kind);
          Py_INCREF(level_kind_obj);
	  PyList_Append(level_list,level_kind_obj);
	}
	return (level_list);
}

static char Fstdc_mapdscrpt__doc__[] =
"Interface to get map descriptors for use with PyNGL"
;

static PyObject *
Fstdc_mapdscrpt(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int ig1, ig2, ig3, ig4, one=1, ni, nj, proj;
        char *cgrtyp;
        float x1,y1, x2,y2, polat,polong, rot, lat1,lon1, lat2,lon2;

	if (!PyArg_ParseTuple(args, "ffffiisiiii",&x1,&y1,&x2,&y2,&ni,&nj,&cgrtyp,&ig1,&ig2,&ig3,&ig4)) {
	  Py_INCREF(Py_None);
	  return Py_None;
        }
        printf("Debug apres parse tuple cgrtyp[0]=%c\n",cgrtyp[0]);
        printf("Debug appel Mapdesc_PyNGL\n");
        f77name(mapdesc_pyngl)(cgrtyp,&one,&ig1,&ig2,&ig3,&ig4,&x1,&y1,&x2,&y2,&ni,&nj,\
                               &proj,&polat,&polong,&rot,&lat1,&lon1,&lat2,&lon2,1);
#if defined(DEBUG)
	printf("Fstdc_mapdscrpt ig1=%d ig2=%d ig3=%d ig4=%d\n",ig1,ig2,ig3,ig4);
	printf("Fstdc_mapdscrpt polat=%f polong=%f rot=%f, lat1=%f lon1=%f lat2=%f, lon2=%f\n",polat,polong,rot,lat1,lon1,lat2,lon2);
#endif
        return Py_BuildValue("{s:f,s:f,s:f,s:f,s:f,s:f,s:f}","polat",polat,"polong",polong,\
                             "rot",rot,"lat1",lat1,"lon1",lon1,"lat2",lat2,"lon2",lon2);
/*        return Py_BuildValue("f",rot); */
}



static char Fstdc_ezinterp__doc__[] =
"Interface to interpolate from one grid to another"
;

static PyObject *
Fstdc_ezinterp(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int ig1S, ig2S, ig3S, ig4S, niS, njS, gdid_src;
	int ig1D, ig2D, ig3D, ig4D, niD, njD, gdid_dst;
/*        int ni_ps=199, nj_ps=155, ig1p=200, ig2p=0, ig3p=23640, ig4p=37403, gdps; */
        int srcaxis,dstaxis,vecteur,ier;
        char *grtypS,*grtypD,*grrefS,*grrefD;
	int dimensions[3]={1,1,1}, strides[3]={0,0,0}, ndimensions=3;
	int type_num=NPY_FLOAT;
	PyArrayObject *arrayin,*arrayin2,*newarray,*newarray2,*xsS,*ysS,*xsD,*ysD;

	if (!PyArg_ParseTuple(args, "OO(ii)s(siiii)OOi(ii)s(siiii)OOii",&arrayin,&arrayin2,&niS,&njS,&grtypS,&grrefS,&ig1S,&ig2S,&ig3S,&ig4S,&xsS,&ysS,&srcaxis,&niD,&njD,&grtypD,&grrefD,&ig1D,&ig2D,&ig3D,&ig4D,&xsD,&ysD,&dstaxis,&vecteur)) {
          printf("\n\n *********\n Fstdc_ezinterp error parsing arguments\n *********\n\n");
	  Py_INCREF(Py_None);
	  return Py_None;
        }

#if defined(DEBUG)
        printf("Debug Fstdc_ezinterp grtypS[0]=%c grrefS[0]=%c\n",grtypS[0],grrefS[0]);
	printf("Debug Fstdc_ezinterp ig1S=%d ig2S=%d ig3S=%d ig4S=%d\n",ig1S,ig2S,ig3S,ig4S);
	printf("Debug Fstdc_ezinterp niS=%d njS=%d\n",niS,njS);
        printf("Debug Fstdc_ezinterp grtypD[0]=%c grrefD[0]=%c\n",grtypD[0],grrefD[0]);
	printf("Debug Fstdc_ezinterp ig1D=%d ig2D=%d ig3D=%d ig4D=%d\n",ig1D,ig2D,ig3D,ig4D);
	printf("Debug Fstdc_ezinterp niD=%d njD=%d\n",niD,njD);
/*
	imprime_ca("xs",xs->data,10);
	imprime_ca("ys",ys->data,10);
*/
	imprime_ca("arrayin",arrayin->data,10);
#endif

	dimensions[0] = (niD>1) ? niD : 1  ;
	dimensions[1] = (njD>1) ? njD : 1 ;
	dimensions[2] = 1 ;
	strides[0]=4 ; strides[1]=strides[0]*dimensions[0] ; strides[2]=strides[1]*dimensions[1];
	if(njD>1) ndimensions=2;
	else ndimensions=1;
#if defined(DEBUG)
        printf("Creating newarray with dimensions %d %d\n", niD, njD);
#endif

        newarray = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndimensions,dimensions,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);
        if (vecteur)
          newarray2 = PyArray_NewFromDescr(&PyArray_Type,
                                          PyArray_DescrFromType(type_num),
                                          ndimensions,dimensions,
                                          NULL, NULL, FTN_Style_Array,
                                          NULL);
        if (srcaxis)
          gdid_src = c_ezgdef_fmem(niS,njS,grtypS,grrefS,ig1S,ig2S,ig3S,ig4S,xsS->data,ysS->data);
        else
          gdid_src = c_ezqkdef(niS,njS,grtypS,ig1S,ig2S,ig3S,ig4S,0);
#if defined(DEBUG)
	printf("Debug Fstdc_ezinterp gdid_src=%d \n",gdid_src);
#endif

        if (dstaxis)
          gdid_dst = c_ezgdef_fmem(niD,njD,grtypD,grrefD,ig1D,ig2D,ig3D,ig4D,xsD->data,ysD->data);
        else
          gdid_dst = c_ezqkdef(niD,njD,grtypD,ig1D,ig2D,ig3D,ig4D,0);
#if defined(DEBUG)
	printf("Debug Fstdc_ezinterp gdid_dst=%d \n",gdid_dst);
	printf("Debug Fstdc_ezinterp vecteur=%d \n",vecteur);
#endif

        ier = c_ezdefset(gdid_dst,gdid_src);
        if (vecteur) {
          ier = c_ezuvint(newarray->data,newarray2->data,arrayin->data,arrayin2->data);
          return Py_BuildValue("OO",newarray,newarray2);
        }
        else {
          ier = c_ezsint(newarray->data,arrayin->data);
#if defined(DEBUG)
          imprime_ca("newarray",newarray->data,10);
#endif
          return Py_BuildValue("O",newarray);
       }

}

static char Fstdc_cxgaig__doc__[] =
"Interface to cxgaig"
;

static PyObject *
Fstdc_cxgaig(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int ig1,ig2,ig3,ig4;
        float xg1,xg2,xg3,xg4;
        char *grtyp;
        int status;

	if (!PyArg_ParseTuple(args, "sffff",&grtyp,&xg1,&xg2,&xg3,&xg4)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}

	status=f77name(cxgaig)(grtyp,&ig1,&ig2,&ig3,&ig4,&xg1,&xg2,&xg3,&xg4);
#if defined(DEBUG)
	printf("Fstdc_cxgaig grtyp=%s xg1=%f,xg2=%f,xg3=%f xg4=%f\n",grtyp,xg1,xg2,xg3,xg4);
	printf("Fstdc_cxgaig ig1=%d,ig2=%d,ig3=%d ig4=%d\n",ig1,ig2,ig3,ig4);
#endif
	return Py_BuildValue("(iiii)",ig1,ig2,ig3,ig4);
}

static char Fstdc_cigaxg__doc__[] =
"Interface to cigaxg"
;

static PyObject *
Fstdc_cigaxg(self, args)
	PyObject *self;	/* Not used */
	PyObject *args;
{
	int ig1,ig2,ig3,ig4;
        float xg1,xg2,xg3,xg4;
        char *grtyp;
        int status;

	if (!PyArg_ParseTuple(args, "siiii",&grtyp,&ig1,&ig2,&ig3,&ig4)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}

	status=f77name(cigaxg)(grtyp,&xg1,&xg2,&xg3,&xg4,&ig1,&ig2,&ig3,&ig4);
#if defined(DEBUG)
	printf("Fstdc_cigaxg ig1=%d,ig2=%d,ig3=%d ig4=%d\n",ig1,ig2,ig3,ig4);
	printf("Fstdc_cigaxg grtyp=%s xg1=%f,xg2=%f,xg3=%f xg4=%f\n",grtyp,xg1,xg2,xg3,xg4);
#endif
	return Py_BuildValue("(ffff)",xg1,xg2,xg3,xg4);
}

/* List of methods defined in the module */

static struct PyMethodDef Fstdc_methods[] = {
 {"fstouv",	(PyCFunction)Fstdc_fstouv,	METH_VARARGS,	Fstdc_fstouv__doc__},
 {"fstvoi",	(PyCFunction)Fstdc_fstvoi,	METH_VARARGS,	Fstdc_fstvoi__doc__},
 {"fstfrm",	(PyCFunction)Fstdc_fstfrm,	METH_VARARGS,	Fstdc_fstfrm__doc__},
 {"fstsui",	(PyCFunction)Fstdc_fstsui,	METH_VARARGS,	Fstdc_fstsui__doc__},
 {"fstinf",	(PyCFunction)Fstdc_fstinf,	METH_VARARGS,	Fstdc_fstinf__doc__},
 {"fnom",	(PyCFunction)Fstdc_fnom,	METH_VARARGS,	Fstdc_fnom__doc__},
 {"fstecr",	(PyCFunction)Fstdc_fstecr,	METH_VARARGS,	Fstdc_fstecr__doc__},
 {"fsteff",	(PyCFunction)Fstdc_fsteff,	METH_VARARGS,	Fstdc_fsteff__doc__},
 {"fstluk",	(PyCFunction)Fstdc_fstluk,	METH_VARARGS,	Fstdc_fstluk__doc__},
 {"fst_edit_dir",(PyCFunction)Fstdc_fst_edit_dir,METH_VARARGS,	Fstdc_fst_edit_dir__doc__},
 {"newdate",	(PyCFunction)Fstdc_newdate,	METH_VARARGS,	Fstdc_newdate__doc__},
 {"difdatr",	(PyCFunction)Fstdc_difdatr,	METH_VARARGS,	Fstdc_difdatr__doc__},
 {"incdatr",	(PyCFunction)Fstdc_incdatr,	METH_VARARGS,	Fstdc_incdatr__doc__},
 {"datematch",	(PyCFunction)Fstdc_datematch,	METH_VARARGS,	Fstdc_datematch__doc__},
 {"level_to_ip1",(PyCFunction)Fstdc_level_to_ip1,METH_VARARGS,	Fstdc_level_to_ip1__doc__},
 {"ip1_to_level",(PyCFunction)Fstdc_ip1_to_level,METH_VARARGS,	Fstdc_ip1_to_level__doc__},
 {"mapdscrpt",	(PyCFunction)Fstdc_mapdscrpt,	METH_VARARGS,	Fstdc_mapdscrpt__doc__},
 {"ezinterp",	(PyCFunction)Fstdc_ezinterp,	METH_VARARGS,	Fstdc_ezinterp__doc__},
 {"cxgaig",	(PyCFunction)Fstdc_cxgaig,	METH_VARARGS,	Fstdc_cxgaig__doc__},
 {"cigaxg",	(PyCFunction)Fstdc_cigaxg,	METH_VARARGS,	Fstdc_cigaxg__doc__},

	{NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFstdc) */

static char Fstdc_module_documentation[] =
""
;


void
initFstdc()
{
	PyObject *m, *d;

	/* Create the module and add the functions */
	m = Py_InitModule4("Fstdc", Fstdc_methods,
		Fstdc_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);

	/* Import the array object */
	import_array();

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyString_FromString("Fstdc.error");
	PyDict_SetItemString(d, "error", ErrorObject);

	/* XXXX Add constants here */

	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module Fstdc");
	printf("RPN (2000) Standard File module V-0.18 initialized\n");
	init_lentab();
}

