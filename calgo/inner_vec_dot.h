
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef __INNER_VEC_DOT_H
#define __INNER_VEC_DOT_H 1

#include <x86intrin.h>
#include <emmintrin.h>

// Update Y one cell with A[i:]*X
static inline
void _inner_vec_ddot(double *y0, int incY, const double *a0,
                     const double *x0, int incX, double alpha, int nC)
{
  register int i;
  register double t0, t1, t2, t3;

  t0 = t1 = t2 = t3 = 0.0;
  for (i = 0; i < nC-3; i += 4) {
    t0 += a0[0] * x0[0];
    //x0 += incX;
    t1 += a0[1] * x0[incX];
    //x0 += incX;
    t2 += a0[2] * x0[2*incX];
    //x0 += incX;
    t3 += a0[3] * x0[3*incX];
    //x0 += incX;
    x0 += incX << 2;
    a0 += 4;
  }
  if (i == nC) 
    goto update;

  if (i < nC-1) {
    t0 += a0[0] * x0[0];
    //x0 += incX;
    t1 += a0[1] * x0[incX];
    //x0 += incX;
    x0 += incX << 1;
    a0 += 2;
    i += 2;
  }
  if (i < nC) {
    t2 += a0[0] * x0[0];
    x0 += incX;
    i++;
  }

 update:
  t0 += t1; t2 += t3;
  y0[0] += (t0 + t2) * alpha;
}

static inline
void _inner_vec2_ddot(double *y0, int incY, const double *a0, const double *a1,
                      const double *x0, int incX, double alpha, int nC)
{
  register int i;
  register double t0, t1, t2, t3, t4, t5, t6, t7;

  t0 = t1 = t2 = t3 = 0.0;
  t4 = t5 = t6 = t7 = 0.0;
  for (i = 0; i < nC-3; i += 4) {
    t0 += a0[0] * x0[0];
    t1 += a1[0] * x0[0];
    //x0 += incX;
    t2 += a0[1] * x0[incX];
    t3 += a1[1] * x0[incX];
    //x0 += incX;
    t4 += a0[2] * x0[2*incX];
    t5 += a1[2] * x0[2*incX];
    //x0 += incX;
    t6 += a0[3] * x0[3*incX];
    t7 += a1[3] * x0[3*incX];
    //x0 += incX;
    x0 += incX << 2;
    a0 += 4;
    a1 += 4;
  }
  if (i == nC) 
    goto update;

  if (i < nC-1) {
    t0 += a0[0] * x0[0];
    t1 += a1[0] * x0[0];
    //x0 += incX;
    t2 += a0[1] * x0[incX];
    t3 += a1[1] * x0[incX];
    //x0 += incX;
    x0 += incX << 1;
    a0 += 2;
    a1 += 2;
    i += 2;
  }
  if (i < nC) {
    t4 += a0[0] * x0[0];
    t5 += a1[0] * x0[0];
    x0 += incX;
    i++;
  }
 update:
  t0 += t2 + t4 + t6;
  t1 += t3 + t5 + t7;
  y0[0] += t0 * alpha;
  y0[incY] += t1 * alpha;
}

static inline
void _inner_vec_ddot_sse(double *y0, int incY, const double *a0,
                         const double *x0, double alpha, int nC, int oddstart)
{
  register int i;
  register double ytmp;
  register __m128d Y0, Y1, A0, X0, A1, X1;

  ytmp = 0.0;
  Y0 = _mm_set1_pd(0.0);
  Y1 = Y0;

  if (oddstart) {
    ytmp += a0[0] * x0[0];
    x0++;
    a0++;
    nC--;
  }

  for (i = 0; i < nC-3; i += 4) {
    A0 = _mm_load_pd(a0);
    X0 = _mm_mul_pd(A0, _mm_load_pd(x0));
    Y0 += X0;
    x0 += 2;
    a0 += 2;

    A1 = _mm_load_pd(a0);
    X1 = _mm_mul_pd(A1, _mm_load_pd(x0));
    Y1 += X1;
    x0 += 2;
    a0 += 2;
  }
  if (i == nC) 
    goto update;

  if (i < nC-1) {
    A0 = _mm_load_pd(a0);
    X0 = _mm_mul_pd(A0, _mm_load_pd(x0));
    Y0 += X0;
    x0 += 2;
    a0 += 2;
    i += 2;
  }
  if (i < nC) {
    ytmp += a0[0] * x0[0];
    i++;
  }
 update:
  Y0 = _mm_hadd_pd(Y0, Y1);
  ytmp += Y0[0];
  ytmp += Y0[1];
  ytmp *= alpha;
  y0[0] += ytmp;
  
}

static inline
void _inner_vec2_ddot_sse(double *y0, int incY, const double *a0, const double *a1,
                         const double *x0, double alpha, int nC, int oddstart)
{
  register int i;
  register double ytmp0, ytmp1;
  register __m128d Y0, Y1, A0, A1, X0, TMP0, TMP1;

  ytmp0 = 0.0;
  ytmp1 = 0.0;
  Y0 = _mm_set1_pd(0.0);
  Y1 = _mm_set1_pd(0.0);

  if (oddstart) {
    ytmp0 += a0[0] * x0[0];
    ytmp1 += a1[0] * x0[0];
    x0++;
    a0++;
    a1++;
    nC--;
  }

  for (i = 0; i < nC-3; i += 4) {
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    X0 = _mm_load_pd(x0);
    TMP0 = A0 * X0;
    TMP1 = A1 * X0;
    Y0 = Y0 + TMP0;
    Y1 = Y1 + TMP1;
    x0 += 2;
    a0 += 2;
    a1 += 2;

    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    X0 = _mm_load_pd(x0);
    TMP0 = A0 * X0;
    TMP1 = A1 * X0;
    Y0 = Y0 + TMP0;
    Y1 = Y1 + TMP1;
    x0 += 2;
    a0 += 2;
    a1 += 2;
  }
  if (i == nC) 
    goto update;

  if (i < nC-1) {
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    X0 = _mm_load_pd(x0);
    TMP0 = A0 * X0;
    TMP1 = A1 * X0;
    Y0 = Y0 + TMP0;
    Y1 = Y1 + TMP1;
    x0 += 2;
    a0 += 2;
    a1 += 2;
    i += 2;
  }
  if (i < nC) {
    ytmp0 += a0[0] * x0[0];
    ytmp1 += a1[0] * x0[0];
    i++;
  }
 update:
  TMP1 = _mm_hadd_pd(Y0, Y1);
  ytmp0 += TMP1[0];
  ytmp1 += TMP1[1];
  y0[0]    += ytmp0 * alpha;
  y0[incY] += ytmp1 * alpha;
  
}

#endif


// Local Variables:
// indent-tabs-mode: nil
// End:
