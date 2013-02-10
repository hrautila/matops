
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef _CMOPS_H
#define _CMOPS_H

#include <stdio.h>


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

/*
extern void _inner_vec_daxpy(double *y0, int incY, const double *a0,
                             const double *x0, int incX, double alpha, int nRE);

extern void _inner_vec2_daxpy(double *y0, int incY, const double *a0, const double *a1,
                              const double *x0, int incX, double alpha, int nRE);

extern void _inner_vec_daxpy_sse(double *y0, const double *a0, const double *x0,
                                 int incX, double alpha, int nRE, int oddStart);

extern void _inner_vec2_daxpy_sse(double *y0, const double *a0, const double *a1,
                                  const double *x0,
                                  int incX, double alpha, int nRE, int oddStart);

extern void _inner_daxpy(double *Cr, const double *Ar, const double *Br,
                         double alpha, int m);

extern void _inner_daxpy2(double *c0, double *c1, const double *Ar,
                          const double *b0, const double *b1, double alpha, int m);

extern void _inner_daxpy_sse(double *Cr, const double *Ar, const double *Br,
                             double alpha, int m);

extern void _inner_daxpy2_sse(double *c0, double *c1, const double *Ar,
                              const double *b0, const double *b1, double alpha, int m);

extern void _inner_daxpy4_sse(double *c0, double *c1, double *c2, double *c3,
                              const double *Ar, const double *b0, const double *b1,
                              const double *b2, const double *b3,
                              double alpha, int m);

extern void _inner_ddot(double *Cr, const double *Ar, const double *Br, double alpha, int nVP);

extern void _inner_ddot4_sse(double *c0, double *c1, double *c2, double *c3,
                             const double *Ar, const double *b0, const double *b1,
                             const double *b2, const double *b3, double alpha, int nVP);

extern void _inner_ddot2_sse(double *c0, double *c1,
                             const double *Ar, const double *b0, const double *b1, 
                             double alpha, int nVP);

extern void _inner_ddot_sse(double *Cr, const double *Ar, const double *Br, double alpha, int nVP);

extern void _inner_ddot4_trans_sse(double *c0, double *c1, double *c2, double *c3,
                                   const double *Ar, const double *b0, const double *b1,
                                   const double *b2, const double *b3, double alpha,
                                   int nVP, int ldB);

extern void _inner_ddot2_trans_sse(double *c0, double *c1,
                                   const double *Ar, const double *b0, const double *b1, 
                                   double alpha, int nVP, int ldB);

extern void _inner_ddot_trans_sse(double *Cr, const double *Ar, const double *Br,
                                  double alpha, int nVP, int ldB);

extern void _inner_ddot_trans(double *Cr, const double *Ar, const double *Br,
                              double alpha, int nVP, int ldB);

*/



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


// C = alpha*A*B + beta*C, A is symmetric, upper matrix, unaligned
extern void
dmult_symm_ua_up_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                         double alpha, double beta,
                         int P, int S, int L, int R, int E,
                         int vlen, int NB, int MB);

// C = alpha*A*B + beta*C, A is symmetric, lower matrix, unaligned
extern void
dmult_symm_ua_low_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          double alpha, double beta,
                          int P, int S, int L, int R, int E,
                          int vlen, int NB, int MB);

// matrix-vector: Y = alpha*A*X + beta*Y
extern void
dmult_mv_notrans(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                 double alpha, double beta,
                 int S, int L, int R, int E,
                 int vlen, int MB);

extern void
dmult_mv_transa(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                double alpha, double beta,
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
