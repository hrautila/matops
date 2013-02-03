
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "cmops.h"
#include <x86intrin.h>
#include <emmintrin.h>

//#include "inner_axpy.h"

// Update Y with 1 column of A
void _inner_vec_daxpy(double *y0, int incY, const double *a0,
                      const double *x0, int incX, double alpha, int nRE)
{
  register int i;
  register double cf0 = alpha * x0[0];

  for (i = 0; i < nRE-3; i += 4) {
    y0[0] += a0[0] * cf0;
    y0 += incY;
    y0[0] += a0[1] * cf0;
    y0 += incY;
    y0[0] += a0[2] * cf0;
    y0 += incY;
    y0[0] += a0[3] * cf0;
    y0 += incY;

    a0 += 4;
  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    y0[0] += a0[0] * cf0;
    y0 += incY;
    y0[0] += a0[1] * cf0;
    y0 += incY;
    a0 += 2;
    i += 2;
  }
  if (i < nRE) {
    y0[0] += a0[0] * cf0;
    y0 += incY;
    i++;
  }
}

// Update Y with 2 columns of A
void _inner_vec2_daxpy(double *y0, int incY, const double *a0, const double *a1,
                      const double *x0, int incX, double alpha, int nRE)
{
  register int i;
  register double cf0, cf1, ytmp;

  cf0 = alpha * x0[0];
  cf1 = alpha * x0[incX];

  for (i = 0; i < nRE-3; i += 4) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[1] * cf0;
    ytmp += a1[1] * cf1;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[2] * cf0;
    ytmp += a1[2] * cf1;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[3] * cf0;
    ytmp += a1[3] * cf1;
    y0[0] += ytmp;
    y0 += incY;

    a0 += 4;
    a1 += 4;
  }
  if (i == nRE)
    return;
      
  if (i < nRE-1) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[1] * cf0;
    ytmp += a1[1] * cf1;
    y0[0] += ytmp;
    y0 += incY;

    a0 += 2;
    a1 += 2;
    i += 2;
  }

  if (i < nRE) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    y0[0] += ytmp;
  }
}


void dmvec_daxpy(double *Yc, const double *Aroot, const double *Xc, double alpha,
                 int incY, int ldA, int incX, int nRE, int nC)
{
  register int j, k;
  register const double *a0, *a1, *a2, *a3;
  const double *Ac;

  Ac = Aroot;
  // 4 columns of A
  for (j = 0; j < nC-3; j += 4) {
    a0 = Ac;
    a1 = a0 + ldA;
    a2 = a1 + ldA;
    a3 = a2 + ldA;
    _inner_vec2_daxpy(Yc, incY, a0, a1, Xc, incX, alpha, nRE);
    Xc += 2*incX;
    _inner_vec2_daxpy(Yc, incY, a2, a3, Xc, incX, alpha, nRE);
    Xc += 2*incX;
    //_inner_vec4_daxpy(Yc, incY, a0, a1, a2, a3, Xc, incX, alpha, nRE);
    //Xc += 4*incX;
    Ac += 4*ldA;
  }
  // Here if j == nC --> nC mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nC)
    return;

  // do the not-multiples of 4 cases....
  if (j < nC-1) {
    a0 = Ac;
    a1 = a0 + ldA;
    _inner_vec2_daxpy(Yc, incY, a0, a1, Xc, incX, alpha, nRE);
    Xc += 2*incX;
    Ac += 2*ldA;
    j += 2;
  }

  if (j < nC) {
    // not multiple of 2
    a0 = Ac;
    _inner_vec_daxpy(Yc, incY, a0, Xc, incX, alpha, nRE);
    Xc += incX;
    Ac += ldA;
    j++;
  }
}

void dmvec_vpur_unaligned_notrans(mvec_t *Y, const mdata_t *A, const mvec_t *X,
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
  Yc = &Y->md[R*X->inc];

  while (vpS < L) {
    AvpS = &A->md[vpS*A->step + R];
    // X element
    Xc = &X->md[vpS*Y->inc];

    //printf("  vpS=%d, vpL=%d\n", vpS, vpL);
    dmvec_daxpy(Yc, AvpS, Xc, alpha, Y->inc, A->step, X->inc, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > L) {
      vpL = L;
    }
  }
}

// Update Y with 1 column of A
void _inner_vec_daxpy_sse(double *y0, const double *a0, const double *x0,
                          int incX, double alpha, int nRE, int oddStart)
{
  register int i;
  register double cf0;
  register __m128d C0, A0, CF;

  cf0 = alpha * x0[0];
  CF = _mm_set1_pd(cf0);

  // if start with unaligned address do it normal way.
  if (oddStart) {
    y0[0] += a0[0] * cf0;
    y0++;
    a0++;
    nRE--;
  }

  // here the vectorized  loop
  for (i = 0; i < nRE-3; i += 4) {
    C0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A0 = A0 * CF;
    C0 = C0 + A0;
    _mm_store_pd(y0, C0);
    y0 += 2;
    a0 += 2;

    C0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A0 = A0 * CF;
    C0 = C0 + A0;
    _mm_store_pd(y0, C0);
    y0 += 2;
    a0 += 2;
  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    C0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A0 = A0 * CF;
    C0 = C0 + A0;
    _mm_store_pd(y0, C0);
    y0 += 2;
    a0 += 2;
    i += 2;
  }
  // the remaining entry
  if (i < nRE) {
    y0[0] += a0[0] * cf0;
    i++;
  }
}

// Update Y with 2 columns of A
void _inner_vec2_daxpy_sse(double *y0, const double *a0, const double *a1,
                           const double *x0,
                          int incX, double alpha, int nRE, int oddStart)
{
  register int i;
  register double cf0, cf1, ytmp;
  register __m128d CF0, CF1, Y0, A0, A1;

  cf0 = alpha * x0[0];
  cf1 = alpha * x0[incX];

  CF0 = _mm_set1_pd(cf0);
  CF1 = _mm_set1_pd(cf1);

  if (oddStart) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    y0[0] += ytmp;
    y0++;
    a0++;
    a1++;
    nRE--;
  }

  for (i = 0; i < nRE-3; i += 4) {
    Y0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    A0 = A0 * CF0;
    Y0 = Y0 + A0;
    A1 = A1 * CF1;
    Y0 = Y0 + A1;
    _mm_store_pd(y0, Y0);
    y0 += 2;
    a0 += 2;
    a1 += 2;

    Y0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    A0 = A0 * CF0;
    Y0 = Y0 + A0;
    A1 = A1 * CF1;
    Y0 = Y0 + A1;
    _mm_store_pd(y0, Y0);
    y0 += 2;
    a0 += 2;
    a1 += 2;
  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    Y0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    A0 = A0 * CF0;
    Y0 = Y0 + A0;
    A1 = A1 * CF1;
    Y0 = Y0 + A1;
    _mm_store_pd(y0, Y0);
    y0 += 2;
    a0 += 2;
    a1 += 2;
    i += 2;
  }

  if (i < nRE) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    y0[0] += ytmp;
    i++;
  }
}


void dmvec_daxpy_sse(double *Yc, const double *Aroot, const double *Xc, double alpha,
                     int ldA, int incX, int nRE, int nC, int oddStart)
{
  register int j, k;
  register double *y0;
  register const double *x0, *a0, *a1, *a2, *a3;
  const double *Ac;

  Ac = Aroot;
  // 4 columns of A
  for (j = 0; j < nC-3; j += 4) {
    x0 = Xc;
    y0 = Yc;
    a0 = Ac;
    a1 = a0 + ldA;
    a2 = a1 + ldA;
    a3 = a2 + ldA;
    //_inner_vec4_daxpy_sse(y0, a0, a1, a2, a3, x0, incX, alpha, nRE, oddStart);
    _inner_vec2_daxpy_sse(y0, a0, a1, x0, incX, alpha, nRE, oddStart);
    x0 += 2*incX;
    _inner_vec2_daxpy_sse(y0, a2, a3, x0, incX, alpha, nRE, oddStart);
    // forward to start of next column in C, B
    Ac += 4*ldA;
    Xc += 4*incX;
  }
  // Here if j == nC --> nC mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nC)
    return;

  // do the not-multiples of 4 cases....
  if (j < nC-1) {
    x0 = Xc;
    y0 = Yc;
    a0 = Ac;
    a1 = a0 + ldA;
    //_inner_vec2_daxpy(y0, 1, a0, a1, x0, incX, alpha, nRE);
    _inner_vec2_daxpy_sse(y0, a0, a1, x0, incX, alpha, nRE, oddStart);
    Xc += 2*incX;
    Ac += 2*ldA;
    j += 2;
  }

  if (j < nC) {
    // not multiple of 2
    x0 = Xc;
    y0 = Yc;
    a0 = Ac;
    //_inner_vec_daxpy(y0, 1, a0, x0, incX, alpha, nRE);
    _inner_vec_daxpy_sse(y0, a0, x0, incX, alpha, nRE, oddStart);
    Xc += incX;
    Ac += ldA;
    j++;
  }
}

// here we have a chance for SSE, ldA is even and incY is one and Y and A
// data arrays have same alignment.
void dmvec_vpur_aligned_notrans(mvec_t *Y, const mdata_t *A, const mvec_t *X,
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
    
  oddStart = ((uintptr_t)Yc & 0xF) != 0;

  while (vpS < L) {
    AvpS = &A->md[vpS*A->step + R];
    // X element
    Xc = &X->md[vpS*Y->inc];

    //printf("  vpS=%d, vpL=%d\n", vpS, vpL);
    dmvec_daxpy_sse(Yc, AvpS, Xc, alpha, A->step, X->inc, E-R, vpL-vpS, oddStart);

    vpS = vpL;
    vpL += vlen;
    if (vpL > L) {
      vpL = L;
    }
  }
}

// if A, Y == aligned(16) and incY == 1 and ldA == even
//      --> we can use SSE with _mm_load() for A, Y and _mm_store() for Y
//
// other cases 
//      --> use the non-SSE version 

// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
void dmult_mv_notrans(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                      double alpha, double beta,
                      int S, int L, int R, int E,
                      int vlen, int MB)
{
  int i, j, nI, nJ, a_aligned, y_aligned, lda_even;

  if (MB <= 0) {
    MB = E - R;
  }

  a_aligned = ((uintptr_t)A->md & 0xF);
  y_aligned = ((uintptr_t)Y->md & 0xF);
  lda_even = (A->step & 0x1) == 0;

  // we can work it out if with SSE if A and Y alignment is same.
  if (lda_even && Y->inc == 1 && a_aligned == y_aligned) {
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      if (beta != 1.0) {
        // scaling with beta ...
        dscale_vec(&Y->md[i*Y->inc], Y->inc, beta, nI);
      }
      dmvec_vpur_aligned_notrans(Y, A, X, alpha, beta, S, L, i, i+nI, vlen);
    }
  } else {
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      if (beta != 1.0) {
        // scaling with beta ...
        dscale_vec(&Y->md[i*Y->inc], Y->inc, beta, nI);
      }
      dmvec_vpur_unaligned_notrans(Y, A, X, alpha, beta, S, L, i, i+nI, vlen);
    }
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:
