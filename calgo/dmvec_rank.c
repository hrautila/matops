
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "cmops.h"

// update 1 column of A matrix (a0) with vector X scaled with elements y0
void _inner_mv_daxpy(double *a0, const double *x, int incX,
		     const double *y0, double alpha, int nRE)
{
  register int i;
  register double cf0, cf1;

  cf0 = alpha * y0[0];
  
  for (i = 0; i < nRE-3; i += 4) {
    a0[0] += x[0] * cf0;
    x += incX;
    a0[1] += x[0] * cf0;
    x += incX;
    a0[2] += x[0] * cf0;
    x += incX;
    a0[3] += x[0] * cf0;
    x += incX;
    a0 += 4;
  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    a0[0] += x[0] * cf0;
    x += incX;
    a0[1] += x[0] * cf0;
    x += incX;
    a0 += 2;
    i += 2;
  }
  if (i < nRE) {
    a0[0] += x[0] * cf0;
    x += incX;
  }
}

// update 2 columns of A matrix (a0, a1) with vector X scaled with elements y0, y1
void _inner_mv_daxpy2(double *a0, double *a1, const double *x, int incX,
		      const double *y0, const double *y1, double alpha, int nRE)
{
  register int i;
  register double cf0, cf1;

  cf0 = alpha * y0[0];
  cf1 = alpha * y1[0];
  
  for (i = 0; i < nRE-3; i += 4) {
    a0[0] += x[0] * cf0;
    a1[0] += x[0] * cf1;
    x += incX;
    a0[1] += x[0] * cf0;
    a1[1] += x[0] * cf1;
    x += incX;
    a0[2] += x[0] * cf0;
    a1[2] += x[0] * cf1;
    x += incX;
    a0[3] += x[0] * cf0;
    a1[3] += x[0] * cf1;
    x += incX;
    a0 += 4;
    a1 += 4;
  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    a0[0] += x[0] * cf0;
    a1[0] += x[0] * cf1;
    x += incX;
    a0[1] += x[0] * cf0;
    a1[1] += x[0] * cf1;
    x += incX;
    a0 += 2;
    a1 += 2;
    i += 2;
  }
  if (i < nRE) {
    a0[0] += x[0] * cf0;
    a1[0] += x[0] * cf1;
    x += incX;
    i++;
  }
}

void dmvec_vpur_rank(mdata_t *A, const mvec_t *X, const mvec_t *Y,  double alpha, 
		     int S, int L, int R, int E, int vlen)
{
  register int i, j, nRE, nSL;
  register double *Ac, *Ar, cf;
  register const double *y, *x;
  
  Ac = &A->md[S*A->step + R];
  y  = &Y->md[S*Y->inc];
  nRE = E - R;
  nSL = L - S;

  for (j = 0; j < nSL; j += 2) {
    x  = &X->md[R*X->inc];
    Ar = Ac;
    _inner_mv_daxpy2(Ac, Ac+A->step, x, X->inc, y, y+Y->inc, alpha, nRE);
    y += 2*Y->inc;
    Ac += 2*A->step;
  }
  if (j == nSL)
    return;
  if (j < nSL) {
    _inner_mv_daxpy(Ac, x, X->inc, y, alpha, nRE);
  }
}

// A = A + alpha * x * y.T; A is M*N, x is M*1, Y is N*1
void drank_mv(mdata_t *A, const mvec_t *X,  const mvec_t *Y, double alpha, 
	      int S, int L, int R, int E,
	      int vlen, int NB, int MB)
{
  int i, j, nI, nJ, x_aligned, y_aligned, lda_even;

  if (MB <= 0) {
    MB = E - R;
  }

  x_aligned = ((uintptr_t)X->md & 0xF);
  y_aligned = ((uintptr_t)Y->md & 0xF);
  lda_even = (A->step & 0x1) == 0;

  if (NB <= 0) {
    NB = L - S;
  }
  if (MB <= 0) {
    MB = E - R;
  }
  if (vlen <= 0) {
    vlen = MAX_VP_DDOT;
  }

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dmvec_vpur_rank(A, X, Y, alpha, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:

