
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <x86intrin.h>


// This will do efectively AXPY C[:,i] = w * A[:,k] + C[:,i] where w = alpha * B[k,i] 
void _inner_daxpy(double *Cr, const double *Ar, const double *Br, double alpha, int m)
{
  register int i;
  register double c0, c1, cf;
  cf = Br[0] * alpha;
  for (i = 0; i < m-3; i += 4) {
    c0 = Ar[0] * cf;
    Cr[0] += c0;
    c1 = Ar[1] * cf;
    Cr[1] += c1;
    Cr += 2;
    Ar += 2;
    c0 = Ar[0] * cf;
    Cr[0] += c0;
    c1 = Ar[1] * cf;
    Cr[1] += c1;
    Cr += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    c0 = Ar[0] * cf;
    Cr[0] += c0;
    c1 = Ar[1] * cf;
    Cr[1] += c1;
    Cr += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    c0 = Ar[0] * cf;
    Cr[0] += c0;
  }
}

// This will do efectively AXPY C[:,i] = w * A[:,k] + C[:,i] where w = alpha * B[k,i] 
void _inner_daxpy_sse(double *Cr, const double *Ar, const double *Br, double alpha, int m)
{
  register int i;
  register __m128d Av, Bv, Cv, Tv, Al;
  Al = _mm_set1_pd(alpha);
  Bv = _mm_set1_pd(Br[0]);
  Bv = Bv * Al;
  for (i = 0; i < m-3; i += 4) {
    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
    Cr += 2;
    Ar += 2;

    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
    Cr += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    // next 2
    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
    Cr += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    // the last one and the extra for odd case. 
    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
  }
}

void _inner_daxpy4_sse(double *c0, double *c1, double *c2, double *c3,
                       const double *Ar, const double *b0, const double *b1,
                       const double *b2, const double *b3,
                       double alpha, int m)
{
  register int i;
  register __m128d T0, T1, T2, T3, Av;
  register __m128d C0, C1, C2, C3, B0, B1, B2, B3;

  Av = _mm_set1_pd(alpha);
  B0 = _mm_set1_pd(b0[0]);
  B1 = _mm_set1_pd(b1[0]);
  B2 = _mm_set1_pd(b2[0]);
  B3 = _mm_set1_pd(b3[0]);
  B0 = B0 * Av;
  B1 = B1 * Av;
  B2 = B2 * Av;
  B3 = B3 * Av;
  for (i = 0; i < m-3; i += 4) {
    Av = _mm_load_pd(Ar);
    C0 = _mm_load_pd(c0);
    C1 = _mm_load_pd(c1);
    C2 = _mm_load_pd(c2);
    C3 = _mm_load_pd(c3);
    T0 = Av * B0;
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T2 = Av * B2;
    C2 = C2 + T2;
    _mm_store_pd(c2, C2);

    T3 = Av * B3;
    C3 = C3 + T3;
    _mm_store_pd(c3, C3);

    c0 += 2;
    c1 += 2;
    c2 += 2;
    c3 += 2;
    Ar += 2;

    Av = _mm_load_pd(Ar);
    C0 = _mm_load_pd(c0);
    C1 = _mm_load_pd(c1);
    C2 = _mm_load_pd(c2);
    C3 = _mm_load_pd(c3);
    T0 = Av * B0;
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T2 = Av * B2;
    C2 = C2 + T2;
    _mm_store_pd(c2, C2);

    T3 = Av * B3;
    C3 = C3 + T3;
    _mm_store_pd(c3, C3);

    c0 += 2;
    c1 += 2;
    c2 += 2;
    c3 += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    // next 2
    Av = _mm_load_pd(Ar);
    C0 = _mm_load_pd(c0);
    C1 = _mm_load_pd(c1);
    C2 = _mm_load_pd(c2);
    C3 = _mm_load_pd(c3);
    T0 = Av * B0;
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T2 = Av * B2;
    C2 = C2 + T2;
    _mm_store_pd(c2, C2);

    T3 = Av * B3;
    C3 = C3 + T3;
    _mm_store_pd(c3, C3);

    c0 += 2;
    c1 += 2;
    c2 += 2;
    c3 += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    // the last one and the extra for odd case. 
    Av = _mm_load_pd(Ar);
    C0 = _mm_load_pd(c0);
    C1 = _mm_load_pd(c1);
    C2 = _mm_load_pd(c2);
    C3 = _mm_load_pd(c3);
    T0 = Av * B0;
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T2 = Av * B2;
    C2 = C2 + T2;
    _mm_store_pd(c2, C2);

    T3 = Av * B3;
    C3 = C3 + T3;
    _mm_store_pd(c3, C3);
  }
}

void _inner_daxpy2_sse(double *c0, double *c1, const double *Ar,
                       const double *b0, const double *b1, double alpha, int m)
{
  register int i;
  register __m128d T0, T1, Av;
  register __m128d C0, C1, B0, B1;

  Av = _mm_set1_pd(alpha);
  B0 = _mm_set1_pd(b0[0]);
  B1 = _mm_set1_pd(b1[0]);
  B0 = B0 * Av;
  B1 = B1 * Av;
  for (i = 0; i < m-3; i += 4) {
    Av = _mm_load_pd(Ar);

    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    c0 += 2;
    c1 += 2;
    Ar += 2;

    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    c0 += 2;
    c1 += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    // next 2
    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    c0 += 2;
    c1 += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    // the last one and the extra for odd case. 
    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
