
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_vec_axpy.h"

// calculates backward a diagonal block and updates Xc values. (lower tridiagonal)
static void
_dmvec_trid_backward(double *Xc, const double *Ac, double alpha,
                     int incX, int ldA, int nRE)
{
  // Y is 
  register int i;
  double *xr, *yr, xtmp;
  const double *Ar, *Acl;

  // lower diagonal matrix of nRE rows/cols and vector X of length nRE
  // move to point to last column and last entry of X.
  Acl = Ac + (nRE-1)*ldA;
  Ar = Acl;
  xr = Xc + (nRE-1)*incX;

  // xr is the current X element, Ar is row in current A column.
  for (i = nRE; i > 0; i--) {
    Ar = Acl + i - 1; // move on diagonal

    // update all x-values below with the current A column and current X
    _inner_vec_daxpy(xr+incX, incX, Ar+1, xr, incX, alpha, nRE-i);
    xtmp = xr[0] * Ar[0];
    //printf("i: %d, xr[0]: %.1f, Ar[0]: %.1f\n", i, xr[0], Ar[0]);
    xr[0] = xtmp * alpha;
    //printf("X:\n"); print_tile(Xc, 1, nRE, 1);

    // previous X, previous column in A 
    xr  -= incX;
    Acl -= ldA;
  }
}

// calculate forward a diagonal block and updates Xc values. (upper tridiagonal)
static void
_dmvec_trid_forward(double *Xc, const double *Ac, double alpha,
                    int incX, int ldA, int nRE)
{
  // Y is 
  register int i;
  double *xr, *yr, xtmp;
  const double *Ar;

  // upper diagonal matrix of nRE rows/cols and vector X, Y of length nRE
  Ar = Ac;
  xr = Xc;

  // xr is the current X element, Ar is row in current A column.
  // Xc is start of X;
  for (i = 0; i < nRE; i++) {
    // update all previous x-values with current A column and current X
    _inner_vec_daxpy(Xc, incX, Ac, xr, incX, alpha, i);
    Ar = Ac + i;
    xtmp = xr[0] * Ar[0];
    //printf("i: %d, xr[0]: %.1f, Ar[0]: %.1f\n", i, xr[0], Ar[0]);
    xr[0] = alpha * xtmp;
    //printf("X:\n"); print_tile(Xc, 1, nRE, 1);
    // next X, next column in A 
    xr += incX;
    Ac += ldA;
  }
}

//extern void memset(void *, int, size_t);

#define MAX_VEC_NB 256

// X = A(-1)*X
void dmvec_trid_unb(mvec_t *X, const mdata_t *A, double alpha, int flags, int N)
{
  int i, nI;

  if (flags & MTX_UPPER) {
    _dmvec_trid_forward(X->md, A->md, alpha, X->inc, A->step, N);
  } else {
    _dmvec_trid_backward(X->md, A->md, alpha, X->inc, A->step, N);
  }
}


void dmvec_trid_blocked(mvec_t *X, const mdata_t *A, double alpha, int flags, int N, int NB)
{
  int i, nI;
  mvec_t Y;

  if (NB <= 0) {
    NB = 68;
  }
  //memset(cB, 0, sizeof(cB));

  if (flags & MTX_UPPER) {
    for (i = 0; i < N; i += NB) {
      nI = N - i < NB ? N - i : NB;
      // solve forward using Y values 
      _dmvec_trid_forward(&X->md[i], &A->md[i*A->step+i], alpha, X->inc, A->step, nI);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
