#include <stdint.h>
#define f77name(a) a##_
#define f77_name(a) a##_
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define Little_Endian
#elif (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#else
#error "endianness not defined"
#endif
#if (INTPTR_MAX == INT32_MAX)
#define PTR_AS_INT int
#elif (INTPTR_MAX == INT64_MAX)
#define PTR_AS_INT long long
#endif
#define INT_32 int
#define INT_64 long long
// Windows
#ifdef WIN32
#define _int64 INT_64
#define open64 open
#define tell64 tell
#endif
//#define tell(fdesc) lseek(fdesc,0,1)
#define FORTRAN_loc_delta           4
#define wordint INT_32
#define ftnword INT_32
#define ftnfloat float
#define wordfloat float
#define bytesperword 4
#define D77MULT            4
#define F2Cl int
