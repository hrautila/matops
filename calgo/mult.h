
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef __MULT_H
#define __MULT_H 1


#include "matcpy.h"

#if defined(__AVX__) //&& defined(USE_AVX)
// AVX1 256bit vectorization
#include "mult_avx.h"

#elif defined(__SSE3__) //&& defined(USE_SSE)
// SSE 128bit vectorization
#include "mult_sse.h"

#else
// unvectorized version
#include "mult_nosimd.h"

#endif


// update 4 columns of C;
static inline
void __CMULT4(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, int col, int nI, int nP)
{
  register int i;
  double *c0, *c1, *c2, *c3;
  const double *a0, *a1;
  const double *b0 = &B->md[(col+0)*B->step];
  const double *b1 = &B->md[(col+1)*B->step];
  const double *b2 = &B->md[(col+2)*B->step];
  const double *b3 = &B->md[(col+3)*B->step];
  for (i = 0; i < nI-1; i += 2) {
    c0 = &C->md[i+(col+0)*C->step];
    c1 = &C->md[i+(col+1)*C->step];
    c2 = &C->md[i+(col+2)*C->step];
    c3 = &C->md[i+(col+3)*C->step];
    a0 = &A->md[(i+0)*A->step];      
    a1 = &A->md[(i+1)*A->step];      
    __mult2c4(c0, c1, c2, c3, a0, a1, b0, b1, b2, b3, alpha, nP);
  }
  if (i == nI)
    return;
  c0 = &C->md[i+(col+0)*C->step];
  c1 = &C->md[i+(col+1)*C->step];
  c2 = &C->md[i+(col+2)*C->step];
  c3 = &C->md[i+(col+3)*C->step];
  a0 = &A->md[(i+0)*A->step];      
  __mult1c4(c0, c1, c2, c3, a0, b0, b1, b2, b3, alpha, nP);
}


//  update two columns of C
static inline
void __CMULT2(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, int col, int nI, int nP)
{
  register int i;
  double *c0, *c1, *a0, *a1;
  const double *b0 = &B->md[(col+0)*B->step];
  const double *b1 = &B->md[(col+1)*B->step];
  for (i = 0; i < nI-1; i += 2) {
    c0 = &C->md[i+(col+0)*C->step];
    c1 = &C->md[i+(col+1)*C->step];
    a0 = &A->md[(i+0)*A->step];      
    a1 = &A->md[(i+1)*A->step];      
    __mult2c2(c0, c1, a0, a1, b0, b1, alpha, nP);
  }
  if (i == nI)
    return;
  c0 = &C->md[i+(col+0)*C->step];
  c1 = &C->md[i+(col+1)*C->step];
  a0 = &A->md[(i+0)*A->step];      
  __mult1c2(c0, c1, a0, b0, b1, alpha, nP);
}

// update one column of C;
static inline
void __CMULT1(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, int col, int nI, int nP)
{
  register int i;
  double *c0;
  const double *a0;
  const double *b0 = &B->md[(col+0)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->md[i+(col+0)*C->step];
    a0 = &A->md[i*A->step];
    __mult1c1(c0, a0, b0, alpha, nP);
  }
}


// update 4 rows of C;
static inline
void __RMULT4(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, int row, int nJ, int nP)
{
  register int k;
  double *c0, *c1;
  const double *b0, *b1;
  const double *a0 = &A->md[(row+0)*A->step];
  const double *a1 = &A->md[(row+1)*A->step];
  const double *a2 = &A->md[(row+2)*A->step];
  const double *a3 = &A->md[(row+3)*A->step];
  for (k = 0; k < nJ-1; k += 2) {
    c0 = &C->md[row+(k+0)*C->step];
    c1 = &C->md[row+(k+1)*C->step];
    b0 = &B->md[(k+0)*B->step];      
    b1 = &B->md[(k+1)*B->step];      
    __mult4c2(c0, c1, a0, a1, a2, a3, b0, b1, alpha, nP);
  }
  if (k == nJ)
    return;
  c0 = &C->md[row+(k+0)*C->step];
  b0 = &B->md[(k+0)*B->step];      
  __mult4c1(c0, a0, a1, a2, a3, b0, alpha, nP);
}

// update 2 rows of C;
static inline
void __RMULT2(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, int row, int nJ, int nP)
{
  register int k;
  double *c0, *c1;
  const double *b0, *b1;
  const double *a0 = &A->md[(row+0)*A->step];
  const double *a1 = &A->md[(row+1)*A->step];
  for (k = 0; k < nJ-1; k += 2) {
    c0 = &C->md[row+(k+0)*C->step];
    c1 = &C->md[row+(k+1)*C->step];
    b0 = &B->md[(k+0)*B->step];      
    b1 = &B->md[(k+1)*B->step];      
    __mult2c2(c0, c1, a0, a1, b0, b1, alpha, nP);
  }
  if (k == nJ)
    return;
  c0 = &C->md[row+(k+0)*C->step];
  b0 = &B->md[(k+0)*B->step];      
  __mult2c1(c0, a0, a1, b0, alpha, nP);
}

// update 1row of C;
static inline
void __RMULT1(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, int row, int nJ, int nP)
{
  register int k;
  double *c0;
  const double *b0;
  const double *a0 = &A->md[(row+0)*A->step];
  for (k = 0; k < nJ; k += 1) {
    c0 = &C->md[row+(k+0)*C->step];
    b0 = &B->md[(k+0)*B->step];      
    __mult1c1(c0, a0, b0, alpha, nP);
  }
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
