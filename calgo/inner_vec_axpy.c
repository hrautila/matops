
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <x86intrin.h>

// Update Y with 4 columns of A
void _inner_vec4_daxpy(double *y0, int incY, const double *a0, const double *a1,
                       const double *a2, const double *a3,
                       const double *x0, int incX, double alpha, int nRE)
{
  register int i, ix;
  register double cf0, cf1, cf2, cf3;
  register double ytmp, t0, t1;

  ix = 0;
  cf0 = alpha * x0[ix];
  ix += incX;
  cf1 = alpha * x0[ix];
  ix += incX;
  cf2 = alpha * x0[ix];
  ix += incX;
  cf3 = alpha * x0[ix];

  for (i = 0; i < nRE-3; i += 4) {
    ytmp = a0[0] * cf0;
    t0 = a1[0] * cf1;
    ytmp += t0;
    t1 = a2[0] * cf2;
    ytmp += t1;
    t0 = a3[0] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[1] * cf0;
    t0 = a1[1] * cf1;
    ytmp += t0;
    t1 = a2[1] * cf2;
    ytmp += t1;
    t0 = a3[1] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[2] * cf0;
    t0 = a1[2] * cf1;
    ytmp += t0;
    t1 = a2[2] * cf2;
    ytmp += t1;
    t0 = a3[2] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[3] * cf0;
    t0 = a1[3] * cf1;
    ytmp += t0;
    t1 = a2[3] * cf2;
    ytmp += t1;
    t0 = a3[3] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
    y0 += incY;

    a0 += 4;
    a1 += 4;
    a2 += 4;
    a3 += 4;
  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    ytmp = a0[0] * cf0;
    t0 = a1[0] * cf1;
    ytmp += t0;
    t1 = a2[0] * cf2;
    ytmp += t1;
    t0 = a3[0] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
    y0 += incY;

    ytmp = a0[1] * cf0;
    t0 = a1[1] * cf1;
    ytmp += t0;
    t1 = a2[1] * cf2;
    ytmp += t1;
    t0 = a3[1] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
    y0 += incY;

    a0 += 2;
    a1 += 2;
    a2 += 2;
    a3 += 2;
    i += 2;
  }

  if (i < nRE) {
    ytmp = a0[0] * cf0;
    t0 = a1[0] * cf1;
    ytmp += t0;
    t1 = a2[0] * cf2;
    ytmp += t1;
    t0 = a3[0] * cf3;
    ytmp += t0;
    y0[0] += ytmp;
  }
}


// Update Y with 4 columns of A
void _inner_vec4_daxpy_sse(double *y0, const double *a0, const double *a1,
                           const double *a2, const double *a3, const double *x0,
                           int incX, double alpha, int nRE, int oddStart)
{
  register int i, ix;
  register double cf0, cf1, cf2, cf3, ytmp;
  register __m128d CF0, CF1, CF2, CF3, Y0, A0, A1, A2, A3;

  ix = 0;
  cf0 = alpha * x0[ix];
  ix += incX;
  cf1 = alpha * x0[ix];
  ix += incX;
  cf2 = alpha * x0[ix];
  ix += incX;
  cf3 = alpha * x0[ix];

  //cf0 = alpha * x0[0];
  //cf1 = alpha * x0[incX];

  CF0 = _mm_set1_pd(cf0);
  CF1 = _mm_set1_pd(cf1);
  CF2 = _mm_set1_pd(cf2);
  CF3 = _mm_set1_pd(cf3);

  if (oddStart) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    ytmp += a2[0] * cf2;
    ytmp += a3[0] * cf3;
    y0[0] += ytmp;
    y0++;
    a0++;
    a1++;
    a2++;
    a3++;
    nRE--;
  }

  for (i = 0; i < nRE-3; i += 4) {
    Y0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    A2 = _mm_load_pd(a2);
    A3 = _mm_load_pd(a3);
    A0 = A0 * CF0;
    Y0 = Y0 + A0;
    A1 = A1 * CF1;
    Y0 = Y0 + A1;
    A2 = A2 * CF2;
    Y0 = Y0 + A2;
    A3 = A3 * CF3;
    Y0 = Y0 + A3;
    _mm_store_pd(y0, Y0);
    y0 += 2;
    a0 += 2;
    a1 += 2;
    a2 += 2;
    a3 += 2;

    Y0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    A2 = _mm_load_pd(a2);
    A3 = _mm_load_pd(a3);
    A0 = A0 * CF0;
    Y0 = Y0 + A0;
    A1 = A1 * CF1;
    Y0 = Y0 + A1;
    A2 = A2 * CF2;
    Y0 = Y0 + A2;
    A3 = A3 * CF3;
    Y0 = Y0 + A3;
    _mm_store_pd(y0, Y0);
    y0 += 2;
    a0 += 2;
    a1 += 2;
    a2 += 2;
    a3 += 2;

  }
  if (i == nRE)
    return;

  if (i < nRE-1) {
    Y0 = _mm_load_pd(y0);
    A0 = _mm_load_pd(a0);
    A1 = _mm_load_pd(a1);
    A2 = _mm_load_pd(a2);
    A3 = _mm_load_pd(a3);
    A0 = A0 * CF0;
    Y0 = Y0 + A0;
    A1 = A1 * CF1;
    Y0 = Y0 + A1;
    A2 = A2 * CF2;
    Y0 = Y0 + A2;
    A3 = A3 * CF3;
    Y0 = Y0 + A3;
    _mm_store_pd(y0, Y0);
    y0 += 2;
    a0 += 2;
    a1 += 2;
    a2 += 2;
    a3 += 2;
    i += 2;
  }

  if (i < nRE) {
    ytmp = a0[0] * cf0;
    ytmp += a1[0] * cf1;
    ytmp += a2[0] * cf2;
    ytmp += a3[0] * cf3;
    y0[0] += ytmp;
    i++;
  }
}

