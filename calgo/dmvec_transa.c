
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_vec_dot.h"

static
void dmvec_ddot(double *Yc, const double *Aroot, const double *Xc, double alpha,
                 int incY, int ldA, int incX, int nRE, int nC)
{
  register int j, k;
  register const double *a0, *a1, *a2, *a3, *x0;
  register double *y0;
  const double *Ac;

  Ac = Aroot;
  x0 = Xc;
  // 4 columns of A
  for (j = 0; j < nRE-3; j += 4) {
    y0 = Yc;
    a0 = Ac;
    a1 = a0 + ldA;
    a2 = a1 + ldA;
    a3 = a2 + ldA;
    _inner_vec2_ddot(y0, incY, a0, a1, Xc, incX, alpha, nC);
    y0 += 2*incY;
    _inner_vec2_ddot(y0, incY, a2, a3, Xc, incX, alpha, nC);
    Ac += 4*ldA;
    Yc += 4*incY;
  }
  // Here if j == nRE --> nRE mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nRE)
    return;

  // do the not-multiples of 4 cases....
  if (j < nRE-1) {
    y0 = Yc;
    a0 = Ac;
    a1 = a0 + ldA;
    _inner_vec2_ddot(y0, incY, a0, a1, Xc, incX, alpha, nC);
    y0 += incY;
    Yc += 2*incY;
    Ac += 2*ldA;
    j += 2;
  }

  if (j < nRE) {
    // not multiple of 2
    y0 = Yc;
    a0 = Ac;
    _inner_vec_ddot(y0, incY, a0, Xc, incX, alpha, nC);
    Yc += incY;
    Ac += ldA;
    j++;
  }
}

static
void dmvec_vpur_unaligned_transa(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                                 double alpha, double beta,
                                 int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL;
  const double *Xc, *Ac, *AvpS;
  double *Yc;

  //printf("R=%d, E=%d\n", R, E);
  vpS = S;
  vpL = vlen < L-S ? S + vlen : L;

  // Y element 
  Yc = &Y->md[R*Y->inc];

  while (vpS < L) {
    AvpS = &A->md[R*A->step + vpS];
    // X element
    Xc = &X->md[vpS*X->inc];

    //printf("  vpS=%d, vpL=%d\n", vpS, vpL);
    dmvec_ddot(Yc, AvpS, Xc, alpha, Y->inc, A->step, X->inc, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > L) {
      vpL = L;
    }
  }
}


static 
void dmvec_ddot_sse(double *Yc, const double *Aroot, const double *Xc, double alpha,
                    int incY, int ldA, int incX, int nRE, int nC, int oddstart)
{
  register int j, k;
  register double *y0;
  register const double *x0, *a0, *a1, *a2, *a3;
  const double *Ac;

  Ac = Aroot;
  x0 = Xc;
  // 4 columns of A
  for (j = 0; j < nRE-3; j += 4) {
    y0 = Yc;
    a0 = Ac;
    a1 = a0 + ldA;
    a2 = a1 + ldA;
    a3 = a2 + ldA;
    _inner_vec2_ddot_sse(y0, incY, a0, a1, Xc, alpha, nC, oddstart);
    y0 += 2*incY;
    _inner_vec2_ddot_sse(y0, incY, a2, a3, Xc, alpha, nC, oddstart);
    Ac += 4*ldA;
    Yc += 4*incY;
  }
  // Here if j == nC --> nC mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nRE)
    return;

  // do the not-multiples of 4 cases....
  if (j < nRE-1) {
    y0 = Yc;
    a0 = Ac;
    a1 = a0 + ldA;
    _inner_vec2_ddot_sse(y0, incY, a0, a1, Xc, alpha, nC, oddstart);
    y0 += incY;
    Yc += 2*incY;
    Ac += 2*ldA;
    j += 2;
  }

  if (j < nRE) {
    // not multiple of 2
    y0 = Yc;
    a0 = Ac;
    _inner_vec_ddot_sse(y0, incY, a0, Xc, alpha, nC, oddstart);
    Yc += incY;
    Ac += ldA;
    j++;
  }
}

// here we have a chance for SSE, ldA is even and incY is one and Y and A
// data arrays have same alignment.
static
void dmvec_vpur_aligned_transa(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                                double alpha, double beta,
                                int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, oddStart;
  const double *Xc, *Ac, *AvpS;
  double *Yc;

  //printf("R=%d, E=%d\n", R, E);
  vpS = S;
  vpL = vlen < L-S ? S + vlen : L;

  // Y element 
  Yc = &Y->md[R];

  while (vpS < L) {
    AvpS = &A->md[R*A->step + vpS];
    // X element
    Xc = &X->md[vpS*X->inc];
    oddStart = ((uintptr_t)Xc & 0xF) != 0;

    //printf("  vpS=%d, vpL=%d\n", vpS, vpL);
    dmvec_ddot_sse(Yc, AvpS, Xc, alpha, Y->inc, A->step, X->inc, E-R, vpL-vpS, oddStart);

    vpS = vpL;
    vpL += vlen;
    if (vpL > L) {
      vpL = L;
    }
  }
}

// if A, X == aligned(16) and incX == 1 and ldA == even
//      --> we can use SSE with _mm_load() for A, X
//
// other cases 
//      --> use the non-SSE version 

// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
void dmult_mv_transa(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                     double alpha, double beta,
                     int S, int L, int R, int E,
                     int vlen, int MB)
{
  int i, j, nI, nJ, a_aligned, x_aligned, lda_even;

  if (MB <= 0) {
    MB = L - S;
  }
  if (vlen <= 0) {
    vlen = 1024;
  }

  a_aligned = ((uintptr_t)A->md & 0xF);
  x_aligned = ((uintptr_t)X->md & 0xF);
  lda_even = (A->step & 0x1) == 0;

  // we can work it out if with SSE if A and Y alignment is same.
  if (lda_even && Y->inc == 1 && a_aligned == x_aligned) {
    //printf("SSE version ...\n");
    for (i = S; i < L; i += MB) {
      nI = L - i < MB ? L - i : MB;
      if (beta != 1.0) {
        // scaling with beta ...
        dscale_vec(&Y->md[R*Y->inc], Y->inc, beta, E-R);
      }
      dmvec_vpur_aligned_transa(Y, A, X, alpha, beta, i, i+nI, R, E, vlen);
    }
  } else {
    for (i = S; i < L; i += MB) {
      nI = L - i < MB ? L - i : MB;
      if (beta != 1.0) {
        // scaling with beta ...
        dscale_vec(&Y->md[R*Y->inc], Y->inc, beta, E-R);
      }
      dmvec_vpur_unaligned_transa(Y, A, X, alpha, beta, i, i+nI, R, E, vlen);
    }
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:
