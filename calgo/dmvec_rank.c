
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "cmops.h"

void dmvec_vpur_rank(mdata_t *A, const mvec_t *Y,  const mvec_t *X, double alpha, 
		     int S, int L, int R, int E, int vlen)
{
  register int i, j;
  register double *Ac, *Ar, cf;
  register const double *y, *x;
  
  Ac = &A->md[S*A->step + R];
  x  = &X->md[S*X->inc];

  for (j = S; j < L; j++) {
    y  = &X->md[R*Y->inc];
    Ar = Ac;
    cf = alpha * x[0];
    for (i = R; i < E; i++) {
      Ar[0] += y[0] * cf;
      y += Y->inc;
      Ar++;
    }
    x += X->inc;
    Ac += A->step;
  }
}

void drank_mv(mdata_t *A, const mvec_t *Y, const mvec_t *X,  double alpha, 
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
      dmvec_vpur_rank(A, Y, X, alpha, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:

