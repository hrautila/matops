
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <x86intrin.h>

#include "cmops.h"

static inline
void _inner_ddot4_sse(double *c0, double *c1, double *c2, double *c3,
                      const double *Ar, const double *b0, const double *b1,
                      const double *b2, const double *b3, double alpha, int nVP)
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
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    B2 = _mm_load_pd(b2);
    B3 = _mm_load_pd(b3);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    F2 = AR * B2;
    C2 = C2 + F2;
    F3 = AR * B3;
    C3 = C3 + F3;

    Ar += 2;
    b0 += 2;
    b1 += 2;
    b2 += 2;
    b3 += 2;

    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    B2 = _mm_load_pd(b2);
    B3 = _mm_load_pd(b3);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    F2 = AR * B2;
    C2 = C2 + F2;
    F3 = AR * B3;
    C3 = C3 + F3;

    Ar += 2;
    b0 += 2;
    b1 += 2;
    b2 += 2;
    b3 += 2;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    B2 = _mm_load_pd(b2);
    B3 = _mm_load_pd(b3);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    F2 = AR * B2;
    C2 = C2 + F2;
    F3 = AR * B3;
    C3 = C3 + F3;

    Ar += 2;
    b0 += 2;
    b1 += 2;
    b2 += 2;
    b3 += 2;
    k += 2;
  }
  if (k < nVP) {
    //printf("ddot2_sse < nVP   : c0 += %9.2e * %9.2e\n", Ar[0], b0[0]);
    //printf("ddot2_sse < nVP   : c1 += %9.2e * %9.2e\n", Ar[0], b1[0]);
    cval = Ar[0] * alpha;
    f0 = cval * b0[0];
    c0[0] += f0;
    f1 = cval * b1[0];
    c1[0] += f1;
    f0 = cval * b2[0];
    c2[0] += f0;
    f1 = cval * b3[0];
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
_inner_ddot2_sse(double *c0, double *c1,
                 const double *Ar, const double *b0, const double *b1, 
                 double alpha, int nVP)
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
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;

    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
    k += 2;
  }
  if (k < nVP) {
    //printf("ddot2_sse < nVP   : c0 += %9.2e * %9.2e\n", Ar[0], b0[0]);
    //printf("ddot2_sse < nVP   : c1 += %9.2e * %9.2e\n", Ar[0], b1[0]);
    cval = Ar[0] * alpha;
    f0 = cval * b0[0];
    c0[0] += f0;
    cval = Ar[0] * alpha;
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
_inner_ddot_sse(double *Cr, const double *Ar, const double *Br, double alpha, int nVP)
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
    BR = _mm_load_pd(Br);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2;

    AR = _mm_load_pd(Ar);
    BR = _mm_load_pd(Br);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    BR = _mm_load_pd(Br);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2;
    k += 2;
  }
  if (k < nVP) {
    //printf("ddot_sse  < nVP   : %9.2e += %9.2e * %9.2e\n", Cr[0], Ar[0], Br[0]);
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
_inner_ddot(double *Cr, const double *Ar, const double *Br, double alpha, int nVP)
{
  int k;
  double f0, f1, f2, f3, cval;

  cval = 0.0;
  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    f1 = Ar[1] * Br[1];
    cval += f1;
    f2 = Ar[2] * Br[2];
    cval += f2;
    f3 = Ar[3] * Br[3];
    cval += f3;
    Br += 4;
    Ar += 4;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    f1 = Ar[1] * Br[1];
    cval += f1;
    Br += 2;
    Ar += 2;
    k += 2;
  }
  if (k < nVP) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    Br++;
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
