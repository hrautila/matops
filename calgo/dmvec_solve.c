
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_vec_axpy.h"

// solves forward a diagonal block and updates Xc values.  Yc vector
// contains values for already solved X's 
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

    // next X, next column in A 
    xr += incX;
    Ac += ldA;
  }
}

extern void memset(void *, int, size_t);

#define MAX_VEC_NB 256

// X = A(-1)*X
void dmvec_solve(mvec_t *X, const mdata_t *A, int flags, int N, int NB)
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
      // calculate A[0:i,i:i+nI]*X[i:i+nI]
      //dmult_gemv_blocked(&Y, A, X, 1.0, 0.0, MTX_NOTRANS, 0, i, i, i+nI, NB, NB);
      // solve forward using Y values 
      _dmvec_solve_forward(&X->md[i], &A->md[i*A->step+i], Y.md, X->inc, A->step, 1, nI);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
