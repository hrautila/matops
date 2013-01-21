
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <x86intrin.h>
#include <emmintrin.h>

#include "cmops.h"

static inline
void _inner_ddot4_trans_sse(double *c0, double *c1, double *c2, double *c3,
                            const double *Ar, const double *b0, const double *b1,
                            const double *b2, const double *b3, double alpha,
                            int nVP, int ldB)
{
  register int k;
  register double f0, f1, cval;
  register __m128d AR, B0, B1, B2, B3, C0, C1, C2, C3, F0, F1, F2, F3, ALP;

  C0 = _mm_set1_pd(0.0);
  C1 = _mm_set1_pd(0.0);
  C2 = _mm_set1_pd(0.0);
  C3 = _mm_set1_pd(0.0);
  ALP = _mm_set1_pd(alpha);

  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    __builtin_prefetch(b0+2*ldB, 0, 1);
    __builtin_prefetch(b1+2*ldB, 0, 1);
    __builtin_prefetch(b2+2*ldB, 0, 1);
    __builtin_prefetch(b3+2*ldB, 0, 1);

    AR = _mm_load_pd(Ar);
    B0 = _mm_set_pd(b0[0], b0[ldB]);
    B1 = _mm_set_pd(b1[0], b1[ldB]);
    B2 = _mm_set_pd(b2[0], b2[ldB]);
    B3 = _mm_set_pd(b3[0], b2[ldB]);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    F2 = AR * B2;
    C2 = C2 + F2;
    F3 = AR * B3;
    C3 = C3 + F3;

    Ar += 2;
    b0 += 2*ldB;
    b1 += 2*ldB;
    b2 += 2*ldB;
    b3 += 2*ldB;

    __builtin_prefetch(b0+2*ldB, 0, 1);
    __builtin_prefetch(b1+2*ldB, 0, 1);
    __builtin_prefetch(b2+2*ldB, 0, 1);
    __builtin_prefetch(b3+2*ldB, 0, 1);

    AR = _mm_load_pd(Ar);
    B0 = _mm_set_pd(b0[0], b0[ldB]);
    B1 = _mm_set_pd(b1[0], b1[ldB]);
    B2 = _mm_set_pd(b2[0], b2[ldB]);
    B3 = _mm_set_pd(b3[0], b2[ldB]);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    F2 = AR * B2;
    C2 = C2 + F2;
    F3 = AR * B3;
    C3 = C3 + F3;

    Ar += 2;
    b0 += 2*ldB;
    b1 += 2*ldB;
    b2 += 2*ldB;
    b3 += 2*ldB;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_set_pd(b0[0], b0[ldB]);
    B1 = _mm_set_pd(b1[0], b1[ldB]);
    B2 = _mm_set_pd(b2[0], b2[ldB]);
    B3 = _mm_set_pd(b3[0], b2[ldB]);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    F2 = AR * B2;
    C2 = C2 + F2;
    F3 = AR * B3;
    C3 = C3 + F3;

    Ar += 2;
    b0 += 2*ldB;
    b1 += 2*ldB;
    b2 += 2*ldB;
    b3 += 2*ldB;
    k += 2;
  }
  if (k < nVP) {
    cval = Ar[0] * alpha;
    f0 = cval * b0[0];
    c0[0] += f0;
    f1 = cval * b1[0];
    c1[0] += f1;
    f0 = cval * b2[0];
    c2[0] += f0;
    f1 = cval * b2[0];
    c3[0] += f1;
    k++;
  }
 update:
  C0 = C0 * ALP;
  C1 = C1 * ALP;
  C2 = C2 * ALP;
  C3 = C3 * ALP;
  c0[0] += C0[0];
  c0[0] += C0[1];
  c1[0] += C1[0];
  c1[0] += C1[1];
  c2[0] += C2[0];
  c2[0] += C2[1];
  c3[0] += C3[0];
  c3[0] += C3[1];
}

static inline void
_inner_ddot2_trans_sse(double *c0, double *c1,
                       const double *Ar, const double *b0, const double *b1, 
                       double alpha, int nVP, int ldB)
{
  register int k;
  register double f0, f1, cval;
  register __m128d AR, B0, B1, C0, C1, F0, F1, ALP;

  C0 = _mm_set1_pd(0.0);
  C1 = _mm_set1_pd(0.0);
  ALP = _mm_set1_pd(alpha);

  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_set_pd(b0[0], b0[ldB]);
    B1 = _mm_set_pd(b1[0], b1[ldB]);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2*ldB;
    b1 += 2*ldB;

    AR = _mm_load_pd(Ar);
    B0 = _mm_set_pd(b0[0], b0[ldB]);
    B1 = _mm_set_pd(b1[0], b1[ldB]);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2*ldB;
    b1 += 2*ldB;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_set_pd(b0[0], b0[ldB]);
    B1 = _mm_set_pd(b1[0], b1[ldB]);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2*ldB;
    b1 += 2*ldB;
    k += 2;
  }
  if (k < nVP) {
    cval = Ar[0] * alpha;
    f0 = cval * b0[0];
    c0[0] += f0;
    f1 = cval * b1[0];
    c1[0] += f1;
    k++;
  }
 update:
  C0 = C0 * ALP;
  c0[0] += C0[0];
  c0[0] += C0[1];
  C1 = C1 * ALP;
  c1[0] += C1[0];
  c1[0] += C1[1];
}

static inline void
_inner_ddot_trans_sse(double *Cr, const double *Ar, const double *Br,
                      double alpha, int nVP, int ldB)
{
  register int k;
  register double f0, cval;
  register __m128d AR, BR, C0, F0, ALP;
  double zero = 0.0;

  C0 = _mm_set1_pd(0.0);
  ALP = _mm_set1_pd(alpha);

  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    AR = _mm_load_pd(Ar);
    BR = _mm_set_pd(Br[0], Br[ldB]);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2*ldB;

    AR = _mm_load_pd(Ar);
    BR = _mm_set_pd(Br[0], Br[ldB]);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2*ldB;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    BR = _mm_set_pd(Br[0], Br[ldB]);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2*ldB;
    k += 2;
  }
  if (k < nVP) {
    cval = Ar[0] * Br[0];
    Cr[0] += cval * alpha;
    Br++;
    Ar++;
    k++;
  }
 update:
  C0 = C0 * ALP;
  Cr[0] += C0[0];
  Cr[0] += C0[1];
}

static inline void
_inner_ddot_trans(double *Cr, const double *Ar, const double *Br,
                  double alpha, int nVP, int ldB)
{
  register int k, iB;
  register double f0, f1, f2, f3, cval;

  cval = 0.0;
  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    iB = 0;
    f0 = Ar[0] * Br[iB];
    cval += f0;
    iB += ldB;
    f1 = Ar[1] * Br[iB];
    cval += f1;
    iB += ldB;
    f2 = Ar[2] * Br[iB];
    cval += f2;
    iB += ldB;
    f3 = Ar[3] * Br[iB];
    cval += f3;
    Br += 4*ldB;
    Ar += 4;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    iB = 0;
    f0 = Ar[0] * Br[0];
    cval += f0;
    iB += ldB;
    f1 = Ar[1] * Br[iB];
    cval += f1;
    Br += 2*ldB;
    Ar += 2;
    k += 2;
  }
  if (k < nVP) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    Br += ldB;
    Ar++;
    k++;
  }
 update:
  f0 = cval * alpha;
  Cr[0] += f0;
}

// Local Variables:
// indent-tabs-mode: nil
// End:
