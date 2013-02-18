
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_vec_axpy.h"

// solves backward a diagonal block and updates Xc values.  Yc vector
// contains per row accumulated values for already solved X's. 
static void
_dmvec_solve_backward(double *Xc, const double *Ac, double *Yc,
                      int incX, int ldA, int incY, int nRE)
{
  // Y is 
  register int i;
  double *xr, *yr, xtmp;
  const double *Ar, *Acl;

  // upper diagonal matrix of nRE rows/cols and vector X, Y of length nRE
  // move to point to last column and last entry of X.
  Acl = Ac + (nRE-1)*ldA;
  Ar = Acl;
  xr = Xc + (nRE-1)*incX;
  yr = Yc + (nRE-1)*incY;

  // xr is the current X element, Ar is row in current A column.
  for (i = nRE-1; i >= 0; i--) {
    Ar = Acl + i;                // move on diagonal
    xtmp = (xr[0] - yr[0]) / Ar[0];
    xr[0] = xtmp;

    // update all y-values with in current column (i is the count above current row)
    _inner_vec_daxpy(Yc, incY, Acl, xr, incX, 1.0, i);
    //printf("Y:\n"); print_tile(Yc, 1, nRE, 1);
    //printf("X:\n"); print_tile(Xc, 1, nRE, 1);
    // previous X, previous Y,  previous column in A 
    xr  -= incX;
    yr  -= incY;
    Acl -= ldA;
  }
}

// solves forward a diagonal block and updates Xc values.  Yc vector
// contains per row accumulated values for already solved X's. 
static void
_dmvec_solve_forward(double *Xc, const double *Ac, double *Yc,
                     int incX, int ldA, int incY, int nRE)
{
  // Y is 
  register int i;
  double *xr, *yr, xtmp;
  const double *Ar;

  // lower diagonal matrix of nRE rows/cols and vector X, Y of length nRE
  Ar = Ac;
  xr = Xc;
  yr = Yc;

  // xr is the current X element, Ar is row in current A column.
  for (i = 0; i < nRE; i++) {
    Ar = Ac + i;                // move on diagonal
    xtmp = (xr[0] - yr[0]) / Ar[0];
    xr[0] = xtmp;
    // update all y-values with in current column
    yr += incY;
    Ar ++;
    _inner_vec_daxpy(yr, incY, Ar, xr, incX, 1.0, nRE-i-1);
    //printf("Y:\n"); print_tile(Yc, 1, nRE, 1);
    //printf("X:\n"); print_tile(Xc, 1, nRE, 1);
    // next X, next column in A 
    xr += incX;
    Ac += ldA;
  }
}

extern void memset(void *, int, size_t);

#define MAX_VEC_NB 256

// X = A(-1)*X
void dmvec_solve_unb(mvec_t *X, const mdata_t *A, int flags, int N)
{
  int i, nI;
  mvec_t Y;
  double cB[MAX_VEC_NB] __attribute__((aligned(64)));
  double *ybuf = (double *)0;
  Y.md = cB;
  Y.inc = 1;

  if (N > MAX_VEC_NB) {
    Y.md = (double *)calloc(N, sizeof(double));
    if (! Y.md) {
      return;
    }
  }

  memset(Y.md, 0, N*sizeof(double));
  if (flags & MTX_LOWER) {
    _dmvec_solve_forward(&X->md[i], &A->md[i*A->step+i], Y.md, X->inc, A->step, 1, N);
  } else {
    _dmvec_solve_backward(&X->md[i], &A->md[i*A->step+i], Y.md, X->inc, A->step, 1, N);
  }
  if (N > MAX_VEC_NB) {
    free(Y.md);
  }
}


void dmvec_solve_blocked(mvec_t *X, const mdata_t *A, int flags, int N, int NB)
{
  int i, nI;
  mvec_t Y;
  double cB[MAX_VEC_NB];
  Y.md = cB;
  Y.inc = 1;

  if (NB <= 0) {
    NB = 68;
  }
  //memset(cB, 0, sizeof(cB));

  if (flags & MTX_LOWER) {
    for (i = 0; i < N; i += NB) {
      nI = N - i < NB ? N - i : NB;
      // solve forward using Y values 
      _dmvec_solve_forward(&X->md[i], &A->md[i*A->step+i], Y.md, X->inc, A->step, 1, nI);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
