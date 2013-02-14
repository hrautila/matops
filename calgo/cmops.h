
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef _CMOPS_H
#define _CMOPS_H

#include <stdio.h>

enum {
  MTX_NOTRANS = 0,
  MTX_TRANSA = 0x1,
  MTX_TRANSB = 0x2,
  MTX_LOWER  = 0x4,
  MTX_UPPER  = 0x8,
  MTX_LEFT   = 0x10,
  MTX_RIGHT  = 0x20
};
  
// multiples of 4 but not powers of 2
#define MAX_VP_ROWS 196
#define MAX_VP_COLS 68


// max values of block sizes for unaligned cases.
#define MAX_UA_MB 130
#define MAX_UA_NB 130
#define MAX_UA_VP 66

#define MAX_MB_DDOT 94
#define MAX_NB_DDOT 94
#define MAX_VP_DDOT 194

#define MAX_MB 254
#define MAX_NB 254
#define MAX_VP 194

#define OFFSET(a,b) ((unsigned int)(a-b))

// simple structure to hold column-major matrix data; 'md' points to first element
// in matrix at index (0, 0); 'step' is the row stride for data i.e. how many elements 
// between A[i,j] and A[i,j+1]
typedef struct mdata {
  double *md;
  int step;
} mdata_t;

typedef struct mvec {
  double *md;
  int inc;
} mvec_t;

extern void *memcpy(void *, const void *, size_t);




extern void print_tile(const double *D, int ldD, int nR, int nC);



extern double
ddot_vec(const double *X, const double *Y, int incX, int incY, int N);

extern void
dscale_vec(double *X, int incX, double f0, int N);

extern void
dscale_tile(double *X, int ldX, double f0, int M, int N);

extern void
vpur_ddot(double *Cc, const double *Aroot, const double *Bc, double alpha,
          int ldC, int ldA, int ldB, int nSL, int nRE, int nVP);

extern void
vpur_daxpy(double *Cc, const double *Aroot, const double *Bc, double alpha,
                int ldC, int ldA, int ldB, int nSL, int nRE, int nVP);




extern void
dmult_mm_blocked(mdata_t *C, const mdata_t *A, const mdata_t *B,
                 double alpha, double beta, int flags,
                 int P, int S, int L, int R, int E, 
                 int vlen, int NB, int MB);


extern void 
dmult_symm_blocked(mdata_t *C, const mdata_t *A, const mdata_t *B,
                   double alpha, double beta, int flags,
                   int P, int S, int L, int R, int E,
                   int vlen, int NB, int MB);


// matrix-vector: Y = alpha*A*X + beta*Y
extern void
dmult_gemv_blocked(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                   double alpha, double beta, int flags, 
                   int S, int L, int R, int E,
                   int vlen, int MB);


// A = A + alpha * x * y.T; A is M*N, x is M*1, Y is N*1
extern void
drank_mv(mdata_t *A, const mvec_t *X, const mvec_t *Y, double alpha, 
         int S, int L, int R, int E, int vlen, int NB, int MB);

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
