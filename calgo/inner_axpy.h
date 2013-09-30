
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <x86intrin.h>


// This will do efectively AXPY C[:,i] = w * A[:,k] + C[:,i] where w = alpha * B[k,i] 
static inline
void _inner_daxpy(double *Cr, const double *Ar, const double *Br, double alpha, int m)
{
  register int i, k;
  register double cf;
  cf = Br[0] * alpha;

  for (i = 0; i < m-3; i += 4) {
    Cr[0] += Ar[0] * cf;
    Cr[1] += Ar[1] * cf;
    Cr[2] += Ar[2] * cf;
    Cr[3] += Ar[3] * cf;
    Cr += 4;
    Ar += 4;
  }
  if (i == m)
    return;
  k = 0;
  switch (m-i) {
  case 3:
    Cr[k] += Ar[k] * cf;
    k++;
  case 2:
    Cr[k] += Ar[k] * cf;
    k++;
  case 1:
    Cr[k] += Ar[k] * cf; 
  }
}

static inline
void _inner_daxpy2(double *Cr0, double *Cr1, const double *Ar,
                   const double *Br0, const double *Br1, double alpha, int m)
{
  register int i, k;
  register double cf0, cf1;
  cf0 = Br0[0] * alpha;
  cf1 = Br1[0] * alpha;

  for (i = 0; i < m-3; i += 4) {
    Cr0[0] += Ar[0] * cf0;
    Cr0[1] += Ar[1] * cf0;
    Cr0[2] += Ar[2] * cf0;
    Cr0[3] += Ar[3] * cf0;

    Cr1[0] += Ar[0] * cf1;
    Cr1[1] += Ar[1] * cf1;
    Cr1[2] += Ar[2] * cf1;
    Cr1[3] += Ar[3] * cf1;

    Cr0 += 4;
    Cr1 += 4;
    Ar += 4;
  }
  if (i == m)
    return;
  k = 0;
  switch (m-i) {
  case 3:
    Cr0[k] += Ar[k] * cf0;
    Cr1[k] += Ar[k] * cf1;
    k++;
  case 2:
    Cr0[k] += Ar[k] * cf0;
    Cr1[k] += Ar[k] * cf1;
    k++;
  case 1:
    Cr0[k] += Ar[k] * cf0;
    Cr1[k] += Ar[k] * cf1;
  }
}

// This will do efectively AXPY C[:,i] = w * A[:,k] + C[:,i] where w = alpha * B[k,i] 
static inline
void _inner_daxpy_sse(double *Cr, const double *Ar, const double *Br, double alpha, int nR)
{
  register int i, k;
  register double cf;
  register __m128d A0, A1, B0, C0, C1, ALP;

  ALP = _mm_set1_pd(alpha);
  B0 = _mm_set1_pd(Br[0]);
  B0 = B0 * ALP;

  for (i = 0; i < nR-3; i += 4) {
    A0 = _mm_mul_pd(B0, _mm_load_pd(Ar));
    C0 = _mm_add_pd(A0, _mm_load_pd(Cr));
    _mm_store_pd(Cr, C0);

    A1 = _mm_mul_pd(B0, _mm_load_pd(&Ar[2]));
    C1 = _mm_add_pd(A1, _mm_load_pd(&Cr[2]));
    _mm_store_pd(&Cr[2], C1);

    Cr += 4;
    Ar += 4;
  }
  if (i == nR)
    return;

  k = 0;
  cf = B0[0];
  switch (nR-i) {
  case 3:
    Cr[k] += Ar[k] * cf;
    k++;
  case 2:
    Cr[k] += Ar[k] * cf;
    k++;
  case 1:
    Cr[k] += Ar[k] * cf; 
  }
}

static inline
void _inner_daxpy2_sse(double *c0, double *c1, const double *Ar,
                       const double *b0, const double *b1, double alpha, int nR)
{
  register int i, k;
  register __m128d A0, A1, A2, A3;
  register __m128d C0, C1, C2, C3,  B0, B1;
  register double cf0, cf1;

  A0 = _mm_set1_pd(alpha);
  B0 = _mm_set1_pd(b0[0]);
  B1 = _mm_set1_pd(b1[0]);
  B0 = B0 * A0;
  B1 = B1 * A0;

  for (i = 0; i < nR-3; i += 4) {
    A0 = _mm_mul_pd(B0, _mm_load_pd(Ar));
    A1 = _mm_mul_pd(B1, _mm_load_pd(Ar));
    C0 = _mm_add_pd(A0, _mm_load_pd(c0));
    C1 = _mm_add_pd(A1, _mm_load_pd(c1));
    _mm_store_pd(c0, C0);
    _mm_store_pd(c1, C1);

    A2 = _mm_mul_pd(B0, _mm_load_pd(&Ar[2]));
    A3 = _mm_mul_pd(B1, _mm_load_pd(&Ar[2]));
    C2 = _mm_add_pd(A2, _mm_load_pd(&c0[2]));
    C3 = _mm_add_pd(A3, _mm_load_pd(&c1[2]));
    _mm_store_pd(&c0[2], C2);
    _mm_store_pd(&c1[2], C3);

    c0 += 4;
    c1 += 4;
    Ar += 4;
  }
  if (i == nR)
    return;
  k = 0;
  cf0 = B0[0];
  cf1 = B1[0];

  switch (nR-i) {
  case 3:
    c0[k] += Ar[k] * cf0;
    c1[k] += Ar[k] * cf1;
    k++;
  case 2:
    c0[k] += Ar[k] * cf0;
    c1[k] += Ar[k] * cf1;
    k++;
  case 1:
    c0[k] += Ar[k] * cf0;
    c1[k] += Ar[k] * cf1;
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:
