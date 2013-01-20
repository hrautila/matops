
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef _CMOPS_H
#define _CMOPS_H

#include <stdio.h>


// max values of block sizes for unaligned cases.
#define MAX_UA_MB 128
#define MAX_UA_NB 128
#define MAX_UA_VP 64

#define MAX_MB 256
#define MAX_NB 256
#define MAX_VP 192

#define OFFSET(a,b) ((unsigned int)(a-b))

// simple structure to hold column-major matrix data; 'md' points to first element
// in matrix at index (0, 0); 'step' is the row stride for data i.e. how many elements 
// between A[i,j] and A[i,j+1]
typedef struct mdata {
  double *md;
  int step;
} mdata_t;

extern void *memcpy(void *, const void *, size_t);

// Copy nC columns of length nL from source to dest. Source row stride is nS and destination
// row stride is nD.
extern inline
void colcpy(double *dst, int nD, const double *src, int nS, int nL, int nC)
{
  register int i;
  for (i = 0; i < nC; i++) {
    memcpy(dst, src, nL*sizeof(double));
    dst += nD;
    src += nS;
  }
}

extern inline
void colcpy_trans(double *dst, int ldD, const double *src, int ldS, int nL, int nC)
{
  register double *Dc, *Dr;
  register const double *Sc, *Sr;
  register int j, i;
  Dc = dst; Sc = src;
  for (j = 0; j < nC; j++) {
    Dr = Dc;
    Sr = Sc;
    __builtin_prefetch(Sr+ldS, 0, 1);
    for (i = 0; i <nL-3; i += 4) {
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
    }
    if (i == nL) {
      goto increment;
    }
    if (i < nL-1) {
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
      i += 2;
    }
    if (i < nL) {
      *Dr = *Sr;
      Dr += ldD;
      Sr++;
      i += 2;
    }
  increment:
    Dc++;
    Sc += ldS;
  }
}

extern double
ddot_vec(const double *X, const double *Y, int incX, int incY, int N);

extern void
dscale_vec(double *X, int ldX, double f0, int N);

extern void
dscale_tile(double *X, int ldX, double f0, int M, int N);

extern void
vpur_ddot(double *Cc, const double *Aroot, const double *Bc, double alpha,
          int ldC, int ldA, int ldB, int nSL, int nRE, int nVP);

extern void
vpur_daxpy(double *Cc, const double *Aroot, const double *Bc, double alpha,
                int ldC, int ldA, int ldB, int nSL, int nRE, int nVP);



// C = alpha*A*B + beta*C for data not aligned at 16 bytes.
extern void
dmult_aligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      double alpha, double beta,
                      int P, int S, int L, int R, int E,
                      int vlen, int NB, int MB);

// for data not aligned at 16 bytes.
extern void
dmult_unaligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        double alpha, double beta, 
                        int P, int S, int L, int R, int E,
                        int vlen, int NB, int MB);

// C = alpha*A.T*B + beta*C for data not aligned at 16 bytes.
extern void
dmult_unaligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       double alpha, double beta, 
                       int P, int S, int L, int R, int E,
                       int vlen, int NB, int MB);

extern void
dmult_aligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
                     double alpha, double beta, 
                     int P, int S, int L, int R, int E,
                     int vlen, int NB, int MB);


// C = alpha*A*B.T + beta*C for data not aligned at 16 bytes.
extern void
dmult_unaligned_transb(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       double alpha, double beta, 
                       int P, int S, int L, int R, int E,
                       int vlen, int NB, int MB);

extern void
dmult_aligned_transb(mdata_t *C, const mdata_t *A, const mdata_t *B,
                     double alpha, double beta, 
                     int P, int S, int L, int R, int E,
                     int vlen, int NB, int MB);


// C = alpha*A.T*B.T + beta*C for data not aligned at 16 bytes.
extern void
dmult_unaligned_transab(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        double alpha, double beta, 
                        int P, int S, int L, int R, int E,
                        int vlen, int NB, int MB);

extern void
dmult_aligned_transab(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      double alpha, double beta, 
                      int P, int S, int L, int R, int E,
                      int vlen, int NB, int MB);


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
